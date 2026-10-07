# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Word-edit watermark attack: masked-LM word replacement.

Scrub and spoof are the same algorithm with the sign of the objective flipped:

                        scrub (sign -1)              spoof (sign +1)
    input               watermarked text             clean text
    edit sites          highest-signal words first   lowest-signal first
    candidate wins by   removing the most signal     injecting the most
    accepts when        worst_r < r_threshold        worst_r > spoof_r_threshold
    on target when      p_pub >= target*m            p_pub <  target/m

Per pass: choose up to `tokens_per_pass` sites, batch-generate a masked-LM pool for
them on one snapshot, pick one candidate per site, then splice right-to-left so earlier
char offsets stay valid. Passes repeat until `edit_fraction` of the words are edited.

What the access level (oracle.py) changes:

    whitebox  sites ranked by their per-token scores, re-surveyed every pass; the
              candidate whose replaced span moves the score the most. With `adaptive`,
              stop once p_pub is on target.
    blackbox  a bit cannot rank, so sites come in one random order (drawn once, without
              replacement) and the candidate is the masked LM's argmax. One word per
              pass, the bit read after each: stop on the first crossing.
    nobox     the black-box choices with nothing to read: edit to the budget.

`overshoot` is the paper's m. White-box with `adaptive` aims at target*m (scrub) or
target/m (spoof); black-box keeps editing past the first crossing, to m times the edits
the crossing cost. 1 stops on the threshold.
"""
import re

import torch

from textseal.attacks.utils import char_offsets, is_content_word


class MLMProposer:
    """Top-k masked-LM fills for word spans of one text, best first."""

    def __init__(self, model, tokenizer, top_k=50, min_prob=0.01):
        self.model, self.tokenizer = model, tokenizer
        self.top_k, self.min_prob = top_k, min_prob
        self._special = set(getattr(tokenizer, "all_special_ids", None) or [])
        self._max_len = min(getattr(tokenizer, "model_max_length", 512), 512)

    @classmethod
    def load(cls, model_id, device, top_k=50, min_prob=0.01):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        dtype = torch.float32 if device == "cpu" else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        return cls(model, tokenizer, top_k=top_k, min_prob=min_prob)

    @torch.no_grad()
    def candidates(self, text, spans, batch=16):
        """[[fill, ...]] per (char_start, char_end) span. Every span is masked against the
        SAME snapshot of `text`, so the caller must splice right-to-left."""
        mask = self.tokenizer.mask_token
        masked = [text[:cs] + mask + text[ce:] for cs, ce in spans]
        device = next(self.model.parameters()).device
        out = [[] for _ in spans]
        for i in range(0, len(masked), batch):
            enc = self.tokenizer(masked[i:i + batch], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self._max_len).to(device)
            logits = self.model(**enc).logits
            for j in range(enc["input_ids"].shape[0]):
                pos = (enc["input_ids"][j] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if not len(pos):
                    continue
                probs = torch.softmax(logits[j, pos[-1].item(), :], dim=-1)
                tp, tid = torch.topk(probs, self.top_k)
                for prob, t in zip(tp.tolist(), tid.tolist()):
                    if prob < self.min_prob:
                        break
                    if t in self._special:
                        continue
                    s = self.tokenizer.decode([t]).strip()
                    if s and is_content_word(s):
                        out[i + j].append(s)
        return out


def _word_runs(text):
    """Maximal non-whitespace runs. A replacement carries no whitespace, so the number
    of runs never changes and an index into this list names one word for the whole attack."""
    return [m.span() for m in re.finditer(r"\S+", text)]


class WordEditAttack:

    def __init__(self, oracle, mlm_proposer, wm_tokenizer, rng, *, goal="scrub",
                 edit_fraction=1.0, tokens_per_pass=5, adaptive=False, target_pvalue=1e-3,
                 overshoot=1.0, r_threshold=0.3, spoof_r_threshold=0.5,
                 fluency_margin=0.0):
        if goal not in ("scrub", "spoof"):
            raise ValueError(f"goal must be 'scrub'/'spoof', got {goal!r}")
        if adaptive and oracle.access != "whitebox":
            raise ValueError("adaptive stopping reads the public p-value: white-box only")
        if overshoot < 1.0:
            raise ValueError(f"overshoot must be >= 1, got {overshoot}")
        self.oracle, self._mlm, self.wm_tokenizer, self.rng = oracle, mlm_proposer, wm_tokenizer, rng
        self.goal, self.sign = goal, (-1 if goal == "scrub" else +1)
        self.ranked = oracle.access == "whitebox"
        # The black box always stops on its bit.
        self.stops = adaptive or oracle.access == "blackbox"
        m = 1.0 if oracle.access == "blackbox" else overshoot   # white-box: m moves the aim
        aim = target_pvalue * m if goal == "scrub" else target_pvalue / m
        self.aim_pvalue = min(aim, 1.0 - 1e-12)
        self.overshoot = overshoot if oracle.access == "blackbox" else 1.0   # black-box: m multiplies the edits
        self.edit_fraction = edit_fraction
        self.tokens_per_pass = 1 if oracle.access == "blackbox" else tokens_per_pass
        # Accept only a candidate whose worst_r clears the goal's side of the threshold.
        self.r_threshold = r_threshold if goal == "scrub" else spoof_r_threshold
        self.fluency_margin = fluency_margin

    def _on_target(self, result):
        """Public p-value past the aim (white-box), or the decision on the goal's side (black-box)."""
        if isinstance(result, bool):
            return result == (self.goal == "spoof")
        return result >= self.aim_pvalue if self.goal == "scrub" else result < self.aim_pvalue

    # ── sites ──

    def _survey(self, text):
        """(p_pub, [(ws, we)] best-first) from the per-token scores."""
        per = self.oracle.per_token(text)
        offs = char_offsets(text, self.wm_tokenizer)
        runs = _word_runs(text)
        seen, ranked = set(), []
        for t in sorted(per, key=lambda t: self.sign * t["score"]):
            cs, ce = offs[t["pos"]]
            run = next((r for r in runs if r[0] < ce and cs < r[1]), None)  # the word overlapping the token
            if run is None:
                continue
            ws, we = run
            if (ws, we) in seen or not is_content_word(text[ws:we]):
                continue
            seen.add((ws, we))
            ranked.append((ws, we))
        return self.oracle.result(text), ranked

    # ── candidates ──

    def _pool(self, text, ws, we, mlm_cands):
        """Deduped [(candidate, masked-LM rank)], the original word excluded."""
        original = text[ws:we].lower()
        seen, pool = set(), []
        for i, c in enumerate(mlm_cands):
            k = c.lower()
            if k not in seen and k != original:
                seen.add(k)
                pool.append((c, i))
        return pool

    def _pick(self, text, ws, we, mlm_cands):
        """The text with one replacement at [ws, we), or None."""
        pool = self._pool(text, ws, we, mlm_cands)
        if not self.ranked:   # masked-LM argmax
            return text[:ws] + pool[0][0] + text[we:] if pool else None
        scored = []
        for cand, rank in pool:
            trial = text[:ws] + cand + text[we:]
            worst_r, span_ss = self.oracle.span_stats(trial, ws, len(cand))
            scored.append((trial, worst_r, self.sign * span_ss, rank))
        scored.sort(key=lambda s: -s[2])                      # best objective first
        # scrub (sign -1) clears low r, spoof (sign +1) high r; none clearing, no edit
        ok = [s for s in scored if self.sign * s[1] > self.sign * self.r_threshold]
        if not ok:
            return None
        # among those within `fluency_margin` of the best objective, the masked LM's top
        # rank: with one proposer its rank is the only fluency signal
        near = [s for s in ok if s[2] >= ok[0][2] - self.fluency_margin]
        return min(near, key=lambda s: s[3])[0]

    # ── main loop ──

    def run(self, text):
        if self.ranked:
            p, ranked = self._survey(text)
            n_sites = len(ranked)
        else:
            p = self.oracle.result(text)
            runs = _word_runs(text)
            order = [i for i, (ws, we) in enumerate(runs) if is_content_word(text[ws:we])]
            self.rng.shuffle(order)
            n_sites = len(order)
        budget = max(1, int(self.edit_fraction * max(len(text.split()), n_sites)))
        if self.oracle.access == "blackbox":
            order = order[:budget]   # the black box's budget caps the sites tried, not the edits
        history = [{"iter": 0, "result": p, "replaced": 0}]
        edits, crossed_at, it = 0, None, 0
        while True:
            if self.stops and crossed_at is None and self._on_target(p):
                crossed_at = edits
            if crossed_at is not None and (crossed_at == 0 or edits >= self.overshoot * crossed_at):
                cause = "target_reached"; break
            if edits >= budget:
                cause = "edit_budget"; break
            it += 1
            if self.ranked:
                if it > 1:
                    p, ranked = self._survey(text)
                spans = ranked[:self.tokens_per_pass]
            else:
                runs = _word_runs(text)
                spans = [runs[i] for i in order[:self.tokens_per_pass]]
                order = order[self.tokens_per_pass:]
            if not spans:
                cause = "edit_budget" if self.oracle.access == "blackbox" else "no_targets"; break
            mlm = self._mlm.candidates(text, spans)
            replaced = 0
            for k in sorted(range(len(spans)), key=lambda k: -spans[k][0]):   # right-to-left
                if edits >= budget:
                    break
                new_text = self._pick(text, *spans[k], mlm[k])
                if new_text is not None:
                    text, edits, replaced = new_text, edits + 1, replaced + 1
            p = self.oracle.result(text)
            history.append({"iter": it, "result": p, "replaced": replaced})
            if replaced == 0 and self.ranked:
                cause = "no_candidates" if it == 1 else "no_replacement"; break
        return {"text": text, "replacements": edits, "exit_cause": cause,
                "crossed_at": crossed_at, "history": history}
