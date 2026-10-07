"""Rephrasing watermark attack: rephrase with the sampling steered by r.

Scrub and spoof are the same attack with the sign of the objective flipped, so
they are one class:

                    scrub (sign -1)               spoof (sign +1)
    input           watermarked text              clean text
    additive bias   -strength * log(r)            +strength * log(r)
    gumbel          full-vocab argmax on 1 - r    nucleus argmax of log(r)/p

Every r comes from the oracle (oracle.py); there is no local PRF and no key.
With the no-box oracle every candidate reads r = 0.5, so the additive bias adds the
same constant to every top-k logit, which cancels in the softmax: the no-box attack
is the unbiased top-k/top-p rephrase with the same prompt, model and temperature.

Adaptive bias (tamper-aware):

    A fixed-strength attack biases every position by the same amount, so a spoof
    overshoots the public detector by far more than it needs and leaves a large
    channel difference T = S_pub - S_priv for the tampering test to find. This
    modulates the additive bias by the running public p-value of the text so far, so
    the attack banks only the surplus it needs:

        aim     = target / overshoot   (spoof: it has to land BELOW the threshold)
        aim     = target * overshoot   (scrub: it has to land ABOVE it)
        ratio_t = p_run / aim       (spoof)   |   aim / p_run   (scrub)
        s_t     = strength * min(1, log(ratio_t)),  0 once past the aim

    Full `strength` while p is far from the aim, easing off as it approaches, 0 once
    past. `overshoot` (the paper's m) is the knob under test: 1 aims exactly at the detection
    threshold, 10 aims a decade past it, buying the samples that would
    otherwise land on the wrong side at the cost of a larger channel difference.
"""
import math

import torch
import torch.nn.functional as F

def prompt_ids(tokenizer, messages) -> list[int]:
    """Token ids for a chat turn list, ready for generation; the turn contents verbatim
    for a tokenizer with no chat template."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode("\n\n".join(m["content"] for m in messages),
                            add_special_tokens=False)


def stop_ids(tokenizer) -> set[int]:
    """Turn/sequence end token ids (handles list-valued eos + chat end tokens)."""
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        ids.update(i for i in eos if isinstance(i, int))
    elif isinstance(eos, int):
        ids.add(eos)
    for t in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return ids


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, stops, pick_batch) -> list[str]:
    """One batched forward per decode step. `prompts` is a list of token-id lists
    (ragged, left-padded here). `pick_batch(logits_rows, texts_rows, rows) ->
    {row: token_id}` decides every still-active row. Rows stop independently at their
    own `stops` token. Returns one decoded string per prompt."""
    device = next(model.parameters()).device
    B = len(prompts)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_id, int):
        eos = getattr(tokenizer, "eos_token_id", None)
        pad_id = eos if isinstance(eos, int) else 0
    L = max(len(p) for p in prompts)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for b, p in enumerate(prompts):
        input_ids[b, L - len(p):] = torch.tensor(p, dtype=torch.long)   # left-pad
        attn[b, L - len(p):] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp_(min=0)
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]              # last col = real last token (left-pad)
    cur_pos = pos[:, -1].clone()               # abs position of each row's last real token
    generated = [[] for _ in range(B)]
    finished = [False] * B
    for _ in range(max_new_tokens):
        texts = [tokenizer.decode(generated[b], skip_special_tokens=True) for b in range(B)]
        next_tok = torch.full((B, 1), pad_id, dtype=torch.long, device=device)
        active = [b for b in range(B) if not finished[b]]
        chosen = pick_batch([logits[b] for b in active], [texts[b] for b in active], active)
        for b in active:
            t = chosen[b]
            if t in stops:
                finished[b] = True
                continue
            generated[b].append(t)
            next_tok[b, 0] = t
        if all(finished):
            break
        attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
        cur_pos = cur_pos + 1
        out = model(input_ids=next_tok, attention_mask=attn,
                    position_ids=cur_pos.unsqueeze(1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return [tokenizer.decode(generated[b], skip_special_tokens=True) for b in range(B)]


PROMPT = ("Rephrase the following text while preserving its meaning. "
          "Output only the rephrased version, nothing else.\n\n"
          "Text: {text}\n\nRephrased:")


class RephrasingAttack:
    """Rephrase `text` with per-step sampling steered toward or away from the mark."""

    def __init__(self, model, tokenizer, oracle, *, goal="scrub", wm_tokenizer=None,
                 strength=1.0, top_k=50, temperature=1.0, top_p=0.95,
                 sampling="additive", adaptive=False, target_pvalue=1e-3, overshoot=1.0):
        if goal not in ("scrub", "spoof"):
            raise ValueError(f"goal must be 'scrub'/'spoof', got {goal!r}")
        if sampling not in ("additive", "gumbel"):
            raise ValueError(f"sampling must be 'additive'/'gumbel', got {sampling!r}")
        if oracle.access == "blackbox":
            raise ValueError("a bit cannot rank candidates: black-box rephrasing is the "
                             "no-box rephrase with the detector as a stopping rule")
        if oracle.access == "nobox" and (sampling != "additive" or adaptive):
            raise ValueError("no-box rephrasing is additive and not adaptive: the gumbel "
                             "rules and the adaptive bias read the detector")
        if adaptive and sampling != "additive":
            raise ValueError("adaptive bias needs sampling='additive'; the gumbel "
                             "rule is a hard argmax with no strength knob")
        if sampling == "gumbel" and (oracle.access != "whitebox" or oracle.r_levels is None):
            raise ValueError("gumbel sampling needs the white-box oracle on uniform r "
                             "(textseal): it reads r for the whole nucleus or vocabulary")
        if sampling == "gumbel" and goal == "scrub" and wm_tokenizer is None:
            raise ValueError("gumbel scrub is a full-vocab rule; it needs wm_tokenizer "
                             "to map surrogate ids into watermark-tokenizer space")
        self.model, self.tokenizer, self.oracle = model, tokenizer, oracle
        self.goal, self.sign = goal, (-1.0 if goal == "scrub" else 1.0)
        self.wm_tokenizer = wm_tokenizer
        self.strength, self.top_k = strength, top_k
        self.temperature, self.top_p = temperature, top_p
        self.sampling = sampling
        self._stops = stop_ids(tokenizer)
        self.adaptive = adaptive
        if adaptive:
            if overshoot < 1.0:
                raise ValueError(f"overshoot must be >= 1, got {overshoot}")
            aim = target_pvalue / overshoot if goal == "spoof" else target_pvalue * overshoot
            self.aim_pvalue = min(aim, 1.0 - 1e-12)   # a scrub aim can pass 1
        self._surrogate_to_wm = (self._build_token_mapping()
                                 if sampling == "gumbel" and goal == "scrub" else None)

    def _adaptive_strength(self, output_text):
        """Attenuated bias strength for this position; self.strength when off."""
        if not self.adaptive:
            return self.strength
        p_run = max(self.oracle.result(output_text), 1e-300)
        ratio = (p_run / self.aim_pvalue if self.goal == "spoof"
                 else self.aim_pvalue / p_run)
        if ratio <= 1.0:                      # past the aim: stop spending
            return 0.0
        return self.strength * min(1.0, math.log(ratio))

    def run_batch(self, texts, max_new_tokens=512):
        """One batched forward per step. One user turn, no system turn: the steering
        lives in the sampler."""
        prompts = [prompt_ids(self.tokenizer, [{"role": "user", "content": PROMPT.format(text=t)}])
                   for t in texts]
        out = generate_batch(self.model, self.tokenizer, prompts, max_new_tokens,
                             self._stops, self.pick_batch)
        return [{"text": t} for t in out]

    def pick_batch(self, logits_rows, texts, rows):
        """{row: token_id} for one decode step. Fixed-strength additive reads r for every
        row in one detector read; the gumbel rules and the adaptive strength read per
        row. Rows are sampled in order, so the RNG draw sequence is the same either way."""
        if self.sampling != "additive" or self.adaptive:
            return {r: self._pick(lg, t) for lg, t, r in zip(logits_rows, texts, rows)}
        tops = [torch.topk(lg / self.temperature, self.top_k) for lg in logits_rows]
        cands = [idx.tolist() for _, idx in tops]
        rmaps = self._r_multi(texts, cands)
        return {row: self._sample(vals, cand, rmap, self.strength)
                for (vals, _), cand, rmap, row in zip(tops, cands, rmaps, rows)}

    # ── r, always from the oracle ──

    def _r(self, output_text, cand_ids):
        return self._r_multi([output_text], [cand_ids])[0]

    def _r_multi(self, texts, cands):
        """[{id: r}] per row for a whole decode step."""
        if self.oracle.access == "nobox":
            return [{} for _ in texts]
        return self.oracle.candidates_after_multi(
            texts, cands, lambda t: self.tokenizer.decode([t]))

    def _build_token_mapping(self):
        """Surrogate vocab id -> its last watermark-tokenizer id (-1 if it has none)."""
        n = self.tokenizer.vocab_size
        mapping = torch.full((n,), -1, dtype=torch.long)
        base = len(self.wm_tokenizer.encode("x", add_special_tokens=False))
        for sid in range(n):
            ids = self.wm_tokenizer.encode("x" + self.tokenizer.decode([sid]),
                                           add_special_tokens=False)
            if len(ids) > base:
                mapping[sid] = ids[-1]
        return mapping

    # ── per-step selection ──

    def _nucleus(self, logits):
        """(candidate ids, renormalized probs): the top-p mass, at most `top_k` ids.

        Without the `top_k` bound the top-p threshold admits the whole vocabulary as
        soon as temperature flattens the distribution -- 2 ids at T=1, 46,745 at T=3.
        Truncating does not change the pick where the unbounded list was affordable:
        `argmax log(r)/p` scores every id negatively and divides by p, so a
        low-probability id can never win, and renormalising over the kept set scales
        all scores by one positive constant.
        """
        probs = F.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        mask = torch.cumsum(sp, dim=-1) - sp > self.top_p
        active = (~mask).nonzero(as_tuple=True)[0][:self.top_k]
        keep = sp[active]
        return si[active].tolist(), keep / keep.sum()

    def _pick(self, logits, output_text):
        l = logits / self.temperature
        if self.sampling == "gumbel":
            if self.goal == "spoof":
                return self._pick_gumbel_spoof(l, output_text)
            return self._pick_gumbel_scrub(l, output_text)
        return self._pick_additive(l, output_text)

    def _pick_additive(self, l, output_text):
        """Soft logit bias by sign * strength * log(r) over the top-k."""
        vals, idx = torch.topk(l, self.top_k)
        cand = idx.tolist()
        return self._sample(vals, cand, self._r(output_text, cand), self._adaptive_strength(output_text))

    def _sample(self, vals, cand, r, s):
        """Top-p sample from the top-k logits `vals` of `cand`, biased by sign * s * log(r)."""
        for i, tok in enumerate(cand):
            vals[i] += self.sign * s * math.log(max(r.get(tok, 0.5), 1e-10))
        probs = F.softmax(vals, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        sp[torch.cumsum(sp, dim=-1) - sp > self.top_p] = 0.0
        sp /= sp.sum()
        return cand[int(si[int(torch.multinomial(sp, 1).item())].item())]

    def _pick_gumbel_spoof(self, l, output_text):
        """TextSeal selection argmax(log(r) / p) over the nucleus."""
        cand, probs = self._nucleus(l)
        r = self._r(output_text, cand)
        best, best_tok = float("-inf"), cand[0]
        for i, tok in enumerate(cand):
            p = probs[i].item()
            if p < 1e-10:
                continue
            score = math.log(max(r.get(tok, 0.5), 1e-30)) / p
            if score > best:
                best, best_tok = score, tok
        return best_tok

    def _pick_gumbel_scrub(self, l, output_text):
        """log p + strength * gumbel(1 - r) over the whole vocabulary: the TextSeal
        selection rule with r -> 1 - r, so it steers away from the mark."""
        n = self.tokenizer.vocab_size
        wm_ids = self._surrogate_to_wm[:n]
        valid = wm_ids >= 0
        r_wm = self.oracle.vocab_after(output_text, wm_ids[valid])
        r = torch.full((n,), 0.5)
        r[valid] = r_wm.to(r.dtype)
        lv = self.oracle.r_levels
        r.clamp_(1.0 / lv, (lv - 1.0) / lv)
        gumbel = -torch.log(-torch.log(1.0 - r))
        scores = F.log_softmax(l[:n], dim=-1).cpu() + self.strength * gumbel
        scores[~valid] = float("-inf")
        return int(torch.argmax(scores).item())
