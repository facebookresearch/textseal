# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Watermarked speculative decoding: a drafter proposes `num_speculative_tokens` tokens per call to a target model.

Configured by WatermarkConfig:
    watermark_type   none | textseal2 | textseal | automark | textgrain | synthid
    verification     "block" (Sun et al., 2025) or "token" (standard speculative sampling)
    spec_decision    TextSeal2 decision rule (textseal.watermarking.verification): block "residual_j" (default) |
                     "fresh_stop" | "tau_stop" | "agreement" | "j_only" | "raw_j"; token "residual" | "agreement"
TextSeal2 drafts with the copula race on the drafter's law; the decision reuses the draft races' leftovers so that the
output follows the target law exactly while the emitted tokens stay keyed. Baselines draft under key A, verify natively
on the raw laws, and draw the last token under key B when it comes from a residual row (textGrain with
textgrain_verify_watermarked verifies the watermarked laws instead, which lowers acceptance).
A window's keyed values are used at most once per answer: block verification marks every keyed row a call read,
token verification the verified rows and the bonus row; other rows get fresh Exp(1) clocks (or fresh sampling).
With num_speculative_tokens = 0 the target decodes alone (one call per token; TextSeal2 or no watermark).

Caches roll back by snapshots (hybrid models such as Qwen3.5 mix linear-attention and attention layers): linear-attention
conv / recurrent states are cloned, attention keys / values are cropped. Both caches hold the committed sequence except
its last token. A call feeds the drafter one token at a time (snapshot after each), feeds the target the last token
plus all drafts at once (logits for every draft row and the bonus row), decides, then restores the drafter to the
snapshot matching the kept drafts and, unless every draft was kept, restores the target and re-feeds the kept tokens.
"""

import time
from dataclasses import replace

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

from textseal.watermarking import verification
from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.core import copula_clocks, keyed_uniform, leftover_race, race, score_listed_tokens, SYNTHID_ROUND_STRIDE, synthid_official_g_values
from textseal.watermarking.generator import SynthidGenerator, topp_law
from textseal.watermarking.textgrain import TextGrainConfig, textgrain_law

TOKEN_FIELDS = ("src", "call", "keyed", "p_tok", "q_tok", "r_tok", "H_p", "H_full", "H_q", "H_r", "tv", "supp_p", "E_tok", "xp_agree", "key")
BLOCK_DECISIONS = {"residual_j": "residj", "fresh_stop": "ts2b", "tau_stop": "taub", "agreement": "agreeb", "j_only": "jonly", "raw_j": "residjraw"}
BASELINES = ("textseal", "automark", "textgrain", "synthid")
KEY_CODE = {"A": 0, "B": 1}
NAN = float("nan")


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def entropy(dist: torch.Tensor) -> torch.Tensor:
    return -(dist * torch.log(dist.clamp(min=1e-300))).sum(-1)


def eos_token_ids(model: PreTrainedModel) -> set[int]:
    eos = set()
    for src in (model.config.get_text_config().eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if src is not None:
            eos |= set(src if isinstance(src, (list, tuple)) else [src])
    return eos


class CachedLM:
    """A causal LM with a rollback-able cache (snapshots of linear-attention states, crops of attention keys / values)."""

    def __init__(self, model: PreTrainedModel):
        self.model, self.cache = model, None
        self.device = model.device

    @torch.no_grad()
    def prefill(self, ids: list[int]) -> None:
        self.cache = DynamicCache(config=self.model.config)
        if ids:
            self.model(torch.tensor([ids], device=self.device), past_key_values=self.cache, use_cache=True, logits_to_keep=1)

    @torch.no_grad()
    def feed(self, ids: list[int], n_logits: int) -> torch.Tensor:
        out = self.model(torch.tensor([ids], device=self.device), past_key_values=self.cache, use_cache=True, logits_to_keep=n_logits)
        return out.logits[0, -n_logits:, :]  # n_logits V

    def snapshot(self) -> list:
        snap = []
        for layer in self.cache.layers:
            entry = {}
            if hasattr(layer, "conv_states"):
                entry["conv"] = {k: (None if v is None else v.clone()) for k, v in layer.conv_states.items()}
                entry["rec"] = {k: (None if v is None else v.clone()) for k, v in layer.recurrent_states.items()}
                entry["prev"] = dict(layer.has_previous_state)
            if getattr(layer, "keys", None) is not None and getattr(layer, "is_initialized", False):
                entry["kv_len"] = layer.keys.shape[-2]
            snap.append(entry)
        return snap

    def restore(self, snap: list) -> None:
        for layer, entry in zip(self.cache.layers, snap):
            if "conv" in entry:
                for k, v in entry["conv"].items():
                    layer.conv_states[k] = None if v is None else v.clone()
                for k, v in entry["rec"].items():
                    layer.recurrent_states[k] = None if v is None else v.clone()
                layer.has_previous_state.update(entry["prev"])
            if "kv_len" in entry and layer.keys.shape[-2] > entry["kv_len"]:
                layer.keys = layer.keys[..., : entry["kv_len"], :]
                layer.values = layer.values[..., : entry["kv_len"], :]


def _fresh(dist: torch.Tensor) -> int:
    return int(torch.multinomial(dist.float(), 1))


def _support_sorted(dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ps, idx = torch.sort(dist[None].float(), dim=-1, descending=True)
    n = int((ps > 0).sum())
    return ps[:, :n], idx[:, :n]


class BaselineRound:
    """One answer's state for a baseline under block verification. `used` holds (window, key name) pairs."""

    def __init__(self, wm_args: WatermarkConfig):
        self.wm_args = wm_args
        self.kind = wm_args.watermark_type
        self.keys = [wm_args.key_a, wm_args.key_b]
        self.used: set = set()
        if self.kind == "textgrain":
            tg = lambda key: TextGrainConfig(key=key, h=wm_args.ngram, B=wm_args.textgrain_blocks, m=wm_args.textgrain_columns, beta=wm_args.textgrain_beta)
            self.tg = {"A": tg(wm_args.secret_key), "B": tg(wm_args.secret_key + wm_args.textgrain_key_b_offset)}
        self.own_recipe = self.kind == "textgrain" and wm_args.textgrain_verify_watermarked

    def _free(self, window: tuple, name: str) -> bool:
        return (window, name) not in self.used and (window, name) not in self.round_names

    def textgrain_law(self, dist: torch.Tensor, ctx: torch.Tensor, name: str) -> tuple[torch.Tensor, int]:
        """textGrain law for a full-vocab law dist (V,) -> (Q (V,) float64, one fresh sample from Q)."""
        ps, idx = torch.sort(dist[None].float(), dim=-1, descending=True)
        law, ids, _ = textgrain_law(ps, idx, ctx, self.tg[name], self.wm_args.textgrain_solver)
        full = torch.zeros(dist.shape[0], dtype=torch.float64, device=dist.device)
        full[ids[0]] = law[0]
        return full, int(ids.gather(1, torch.multinomial(law.float(), 1))[0, 0])

    def _sample(self, dist: torch.Tensor, ctx: torch.Tensor, name: str) -> int:
        key = self.keys[KEY_CODE[name]]
        if self.kind == "automark":
            vocab = torch.arange(dist.shape[0], device=dist.device)[None]
            return int(race(copula_clocks(ctx, vocab, [key], self.wm_args.rho, self.wm_args.prf), dist[None])[0])
        if self.kind == "textgrain":
            return self.textgrain_law(dist, ctx, name)[1]
        ps, idx = _support_sorted(dist)
        if self.kind == "textseal":
            u = keyed_uniform(ctx, idx, key, self.wm_args.prf)
            return int(idx[0, torch.argmax(torch.log(u + 1e-30) / ps.double())])
        depth = self.wm_args.depth
        if self.wm_args.synthid_hash == "official":
            g = synthid_official_g_values(ctx, idx, key, depth)
        else:
            g = torch.stack([score_listed_tokens(ctx, replace(self.wm_args, secret_key=key, method="binary"), idx + SYNTHID_ROUND_STRIDE * dd)
                             for dd in range(depth)], -1)
        law = SynthidGenerator.update_probs(ps.double(), g.double())
        law = law / law.sum(1, keepdim=True)
        return int(idx[0, torch.multinomial(law[0].float(), 1)])

    def _normal_names(self) -> list[str]:
        """Key names tried, in order, for a draft or a bonus token (the scheme's normal step)."""
        if self.kind == "textseal":
            first = 0 if float(torch.rand(())) < self.wm_args.mixing_alpha else 1
            return ["AB"[first], "AB"[1 - first]]
        if self.kind == "synthid":
            return ["A", "B"]
        return ["A"]

    def start_round(self) -> None:
        self.round_names: set = set()
        self.row_names: list = []
        self.q_wm: list = []

    def draft(self, q: torch.Tensor, window: tuple) -> int:
        """Draft one token from q (V,) at window; records the key name (or None for fresh) of the row."""
        ctx = torch.tensor([window], device=q.device)
        for name in self._normal_names():
            if self._free(window, name):
                self.round_names.add((window, name))
                self.row_names.append(name)
                if self.own_recipe:
                    q_wm, x = self.textgrain_law(q, ctx, name)
                    self.q_wm.append(q_wm)
                    return x
                return self._sample(q, ctx, name)
        self.row_names.append(None)
        self.q_wm.append(None)
        return _fresh(q)

    def decide(self, P: torch.Tensor, Q: torch.Tensor, D: torch.Tensor, windows: list[tuple], window_b: tuple) -> tuple[int, int, str | None]:
        """Block decision for one call. P (N+1, V), Q (N, V), D (N,) -> (L, X, key name of X)."""
        n = Q.shape[0]
        device = P.device
        if self.own_recipe:
            Pw, Qw = P.clone(), Q.clone()
            for ii in range(n):
                if self.row_names[ii] is not None:
                    Pw[ii] = self.textgrain_law(P[ii], torch.tensor([windows[ii]], device=device), "A")[0]
                    Qw[ii] = self.q_wm[ii]
            bonus_keyed = self._free(window_b, "A")
            if bonus_keyed:
                Pw[n] = self.textgrain_law(P[n], torch.tensor([window_b], device=device), "A")[0]
            W, R, _ = verification.block_law(Pw, Qw, D)
            L = int(torch.multinomial(W.float() / float(W.sum()), 1))
            X = _fresh(R[L])
            name = "A" if (L < n and self.row_names[L] is not None) or (L == n and bonus_keyed) else None
            if L == n and bonus_keyed:
                self.round_names.add((window_b, "A"))
            self.used |= self.round_names
            return L, X, name
        W, R, _ = verification.block_law(P, Q, D)
        L = int(torch.multinomial(W.float() / float(W.sum()), 1))
        if L == n:
            window, names = window_b, self._normal_names()
        else:
            window, draft_name = windows[L], self.row_names[L]
            if self.kind == "textseal":
                names = ["AB"[1 - KEY_CODE[draft_name]]] if draft_name else ["A", "B"]
            elif self.kind in ("textgrain", "synthid"):
                names = ["B"] if draft_name else ["A", "B"]
            else:
                names = ["B"]
        ctx = torch.tensor([window], device=device)
        X, name = None, None
        for nm in names:
            if self._free(window, nm):
                X, name = self._sample(R[L], ctx, nm), nm
                self.round_names.add((window, nm))
                break
        if X is None:
            X = _fresh(R[L])
        self.used |= self.round_names
        return L, X, name


class SpeculativeGenerator:
    """Watermarked speculative decoding (see the module docstring). Batch size 1; generate() loops over prompts."""

    def __init__(self, draft_model: PreTrainedModel | None, target_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, wm_args: WatermarkConfig):
        self.draft = CachedLM(draft_model) if draft_model is not None else None
        self.target = CachedLM(target_model)
        self.tokenizer = tokenizer
        self.wm_args = wm_args
        self.device = target_model.device
        self.eos = eos_token_ids(target_model)
        kind, n_drafts = wm_args.watermark_type, wm_args.num_speculative_tokens
        if kind not in ("none", "textseal2") + BASELINES:
            raise ValueError(f"Speculative decoding supports none, textseal2 and {BASELINES}, not {kind!r}")
        if n_drafts > 0 and kind in BASELINES and wm_args.verification != "block":
            raise ValueError("Baselines are implemented with block verification")
        if n_drafts == 0 and kind not in ("none", "textseal2"):
            raise ValueError("Target-only decoding supports none and textseal2")

    def _scheme(self) -> str:
        kind, block = self.wm_args.watermark_type, self.wm_args.verification == "block"
        if self.wm_args.num_speculative_tokens == 0:
            return "arts2" if kind == "textseal2" else "arnone"
        if kind == "none":
            return "noneb" if block else "none"
        if kind == "textseal2":
            if block:
                return BLOCK_DECISIONS[self.wm_args.spec_decision]
            return {"residual": "cop2r", "agreement": "cop2a"}[self.wm_args.spec_decision]
        return "baseline"

    def clocks(self, window: tuple, vocab: int, keys: list[int]) -> torch.Tensor:
        """TextSeal2 copula clocks over the whole vocabulary -> (V,) float64."""
        ctx = torch.tensor([window], device=self.device)
        return copula_clocks(ctx, torch.arange(vocab, device=self.device)[None], keys, self.wm_args.rho, self.wm_args.prf)[0]

    def fresh_clocks(self, vocab: int) -> torch.Tensor:
        return torch.empty(vocab, dtype=torch.float64, device=self.device).exponential_()

    @torch.no_grad()
    def generate(self, prompts: list[str] | list[list[int]], max_gen_len: int, temperature: float = 1.0, top_p: float = 0.95,
                 return_ids: bool = False) -> list[str] | list[list[int]]:
        out = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
            gen = self.generate_one(ids, max_gen_len, temperature, top_p)["ids"]
            out.append(gen if return_ids else self.tokenizer.decode(gen))
        return out

    @torch.no_grad()
    def generate_one(self, prompt: list[int], max_new: int, temperature: float = 1.0, top_p: float = 0.95, secret_key: int | None = None) -> dict:
        """One answer: token ids (EOS excluded) with per-token, per-call and per-answer logs (sources, laws, timings).

        secret_key overrides wm_args.secret_key for this answer (one key per answer in the distortion tests).
        """
        wm_args = self.wm_args if secret_key is None else replace(self.wm_args, secret_key=secret_key)
        law = lambda logits: topp_law(logits, temperature, top_p)[0]
        if self.wm_args.num_speculative_tokens == 0:
            return self._generate_target_only(prompt, wm_args, max_new, law)
        return self._generate_spec(prompt, wm_args, max_new, law)

    def _generate_spec(self, prompt: list[int], wm_args: WatermarkConfig, max_new: int, law) -> dict:
        scheme = self._scheme()
        is_block = wm_args.verification == "block"
        watermarked = wm_args.watermark_type != "none"
        native_drafts = scheme in ("none", "noneb", "jonly")
        keys = [wm_args.key_a, wm_args.key_b]
        n_window = wm_args.ngram
        bl = BaselineRound(wm_args) if scheme == "baseline" else None
        used: set = set()
        committed = list(prompt)
        n_prompt = len(prompt)
        tok_log = {k: [] for k in TOKEN_FIELDS}
        calls = []
        t_log = 0.0
        t0 = sync_time()
        self.draft.prefill(committed[:-1])
        self.target.prefill(committed[:-1])
        t_prefill = sync_time() - t0
        ttft = None
        while len(committed) - n_prompt < max_new:
            ci = len(calls)
            timing = {"draft_fwd": 0.0, "draft_post": 0.0}
            last = committed[-1]
            drafts, qs, Es, keyed_rows, windows, snaps = [], [], [], [], [], []
            if bl is not None:
                bl.start_round()
            feed_tok = last
            for _ in range(wm_args.num_speculative_tokens):
                ta = sync_time()
                logits = self.draft.feed([feed_tok], 1)
                snaps.append(self.draft.snapshot())
                tb = sync_time()
                q = law(logits)[0].double()  # V
                window = tuple((committed + drafts)[-n_window:])
                if bl is not None:
                    x = bl.draft(q, window)
                    keyed, E = bl.row_names[-1] is not None, None
                else:
                    keyed = watermarked and window not in used and window not in windows
                    E = self.clocks(window, q.shape[0], keys) if keyed else self.fresh_clocks(q.shape[0])
                    # Block decisions treat every draft as the winner of its row's race on E (keyed or fresh), so block
                    # schemes with keyed drafts race even on rows without keyed values.
                    x = int(torch.multinomial(q.float(), 1)) if native_drafts or (not keyed and not is_block) else int(race(E, q))
                tc = sync_time()
                timing["draft_fwd"] += tb - ta
                timing["draft_post"] += tc - tb
                drafts.append(x)
                qs.append(q)
                Es.append(E)
                keyed_rows.append(keyed)
                windows.append(window)
                feed_tok = x
                if x in self.eos:
                    break
            n_d = len(drafts)
            ta = sync_time()
            snap_t = self.target.snapshot()
            target_logits = self.target.feed([last] + drafts, n_d + 1).float()  # (n_d + 1) V
            tb = sync_time()
            P = law(target_logits).double()  # (n_d + 1) V
            window_b = tuple((committed + drafts)[-n_window:])
            if bl is not None:
                keyed_b, E_b = False, None
            else:
                keyed_b = watermarked and window_b not in used and window_b not in windows
                E_b = self.clocks(window_b, P.shape[1], keys) if keyed_b else self.fresh_clocks(P.shape[1])
            tc = sync_time()
            timing["target_fwd"], timing["target_post"] = tb - ta, tc - tb
            Q = torch.stack(qs)  # n_d V
            D = torch.tensor(drafts, device=self.device)
            info = {}
            ta = sync_time()
            key_last = None
            if bl is not None:
                L, X, key_last = bl.decide(P, Q, D, windows, window_b)
                keyed_b = L == n_d and key_last is not None
                emitted_src = ["draft"] * L + ["blast"]
            elif is_block:
                if scheme == "noneb":
                    W, R, _ = verification.block_law(P, Q, D)
                    L = int(torch.multinomial(W.float() / float(W.sum()), 1))
                    X = int(torch.multinomial(R[L].float(), 1))
                else:
                    out = verification.decide(scheme, P, Q, D, torch.stack(Es + [E_b]))
                    L, X = out["L"], out["X"]
                    info = {"fallback": out["fallback"], "plain_match": out["plain_match"]}
                emitted_src = ["draft"] * L + ["blast"]
            else:
                L, X, emitted_src, info = self._verify_tokens(scheme, P, Q, Es, E_b, drafts, keyed_rows, keyed_b)
            timing["decide"] = sync_time() - ta
            if is_block and bl is None:
                for w_, k_ in zip(windows + [window_b], keyed_rows + [keyed_b]):
                    if k_:
                        used.add(w_)
            elif not is_block:
                for ii in range(min(L + 1, n_d)):
                    if keyed_rows[ii]:
                        used.add(windows[ii])
                if L == n_d and keyed_b:
                    used.add(window_b)
            emitted = drafts[:L] + [X]
            tl = sync_time()
            self._log_call(tok_log, ci, P, Q, D, target_logits, emitted, emitted_src, L, n_d, is_block, info, Es + [E_b], bl, keyed_rows + [keyed_b], key_last)
            t_log += sync_time() - tl
            ta = sync_time()
            if L < n_d:
                self.draft.restore(snaps[L])
                self.target.restore(snap_t)
                self.target.feed([last] + drafts[:L], 1)
            else:
                self.draft.restore(snaps[n_d - 1])
                self.draft.feed([drafts[n_d - 1]], 1)
            timing["rollback"] = sync_time() - ta
            committed += emitted
            pd = P[torch.arange(n_d), D].tolist() if n_d else []
            qd = Q[torch.arange(n_d), D].tolist() if n_d else []
            tv_rows = (0.5 * (P[:n_d] - Q).abs().sum(-1)).tolist() if n_d else []
            calls.append({"n_drafts": n_d, "kept": L, "emitted": len(emitted), "drafts": drafts, "p_drafts": [round(v, 6) for v in pd],
                          "q_drafts": [round(v, 6) for v in qd], "tv_rows": [round(v, 5) for v in tv_rows],
                          "keyed_rows": [bool(k) for k in keyed_rows] + [bool(keyed_b)],
                          **{k: v for k, v in info.items() if k != "xp_agree"}, **({"xp_agree": info["xp_agree"]} if "xp_agree" in info else {}),
                          "t": {k: round(v, 6) for k, v in timing.items()}})
            if ttft is None:
                ttft = sync_time() - t0 - t_log
            if any(t in self.eos for t in emitted):
                break
        return self._finish(committed, n_prompt, tok_log, calls, t0, t_prefill, ttft, t_log)

    def _verify_tokens(self, scheme, P, Q, Es, E_b, drafts, keyed_rows, keyed_b) -> tuple[int, int, list[str], dict]:
        """Token-by-token verification: fresh accept coin and residual race (cop2r), or race coupling (cop2a)."""
        n_d = len(drafts)
        L, X, emitted_src, xp_flags = n_d, None, [], []
        for ii in range(n_d):
            p, q, E, x = P[ii], Q[ii], Es[ii], drafts[ii]
            if scheme == "cop2a" and keyed_rows[ii]:
                xp = int(race(E, p))
                xp_flags.append(int(xp == x))
                ok, y, fallback = verification.verify(p, q, x, xp)
                if not ok:
                    L, X = ii, y
                    emitted_src.append("repl_fresh" if fallback else "repl")
                    break
                emitted_src.append("draft")
                continue
            xp_flags.append(None)
            ratio = min(1.0, float(p[x]) / max(float(q[x]), 1e-30))
            if float(torch.rand(())) < ratio:
                emitted_src.append("draft")
                continue
            r = (p - q).clamp(min=0)
            r = r / r.sum() if float(r.sum()) > 0 else p
            y = leftover_race(E, q, r, x) if (keyed_rows[ii] and scheme == "cop2r") else int(torch.multinomial(r.float(), 1))
            L, X = ii, y
            emitted_src.append("repl")
            break
        if X is None:
            X = int(race(E_b, P[n_d])) if keyed_b else int(torch.multinomial(P[n_d].float(), 1))
            emitted_src.append("bonus")
        return L, X, emitted_src, ({"xp_agree": xp_flags} if scheme == "cop2a" else {})

    def _log_call(self, tok_log, ci, P, Q, D, target_logits, emitted, emitted_src, L, n_d, is_block, info, all_E, bl, keyed_all, key_last) -> None:
        rows = list(range(L + 1))
        P_rows = P[rows]
        H_p = entropy(P_rows).tolist()
        H_full = entropy(torch.softmax(target_logits[rows].double(), -1)).tolist()
        supp = (P_rows > 0).sum(-1).tolist()
        tv_rows = (0.5 * (P[:n_d] - Q).abs().sum(-1)).tolist() if n_d else []
        H_q_rows = entropy(Q).tolist() if n_d else []
        if is_block:
            W, R, _ = verification.block_law(P, Q, D)
            r_last, H_r_last = float(R[L, emitted[-1]]), float(entropy(R[L]))
            info["W"] = [round(v, 6) for v in W.tolist()]
        elif emitted_src[-1].startswith("repl"):
            r = (P[L] - Q[L]).clamp(min=0)
            r = r / r.sum() if float(r.sum()) > 0 else P[L]
            r_last, H_r_last = float(r[emitted[-1]]), float(entropy(r))
        else:
            r_last, H_r_last = NAN, NAN
        xp_rows = info.get("xp_agree", [])
        if bl is not None:
            row_keys = [KEY_CODE.get(nm, -1) for nm in bl.row_names[:L]] + [KEY_CODE.get(key_last, -1)]
            keyed_flags = [k >= 0 for k in row_keys]
        else:
            keyed_flags = [bool(k) for k in keyed_all[: L + 1]]
            row_keys = [2 if k else -1 for k in keyed_flags]
        for jj, (t, s) in enumerate(zip(emitted, emitted_src)):
            last_tok, drafted_row = jj == L, jj < n_d
            E_row = all_E[jj]
            values = (s, ci, keyed_flags[jj], float(P[jj, t]), float(Q[jj, t]) if drafted_row else NAN, r_last if last_tok else NAN, H_p[jj], H_full[jj],
                      H_q_rows[jj] if drafted_row else NAN, H_r_last if last_tok else NAN, tv_rows[jj] if drafted_row else NAN, int(supp[jj]),
                      NAN if E_row is None else float(E_row[t]), xp_rows[jj] if jj < len(xp_rows) else None, row_keys[jj])
            for k, v in zip(TOKEN_FIELDS, values):
                tok_log[k].append(v)

    def _finish(self, committed, n_prompt, tok_log, calls, t0, t_prefill, ttft, t_log) -> dict:
        wall = sync_time() - t0
        gen = committed[n_prompt:]
        cut = next((ii for ii, t in enumerate(gen) if t in self.eos), len(gen))
        return {"ids": gen[:cut], "ended_eos": cut < len(gen), **{k: v[:cut] for k, v in tok_log.items()}, "calls": calls,
                "t_prefill": t_prefill, "ttft": ttft, "wall": wall, "t_log": t_log,
                "proposed": sum(c["n_drafts"] for c in calls), "accepted": sum(c["kept"] for c in calls), "n_calls": len(calls)}

    def _generate_target_only(self, prompt: list[int], wm_args: WatermarkConfig, max_new: int, law) -> dict:
        """Target alone, one call per token (latency reference), same logging fields as speculative decoding."""
        keys = [wm_args.key_a, wm_args.key_b]
        used, committed, n_prompt = set(), list(prompt), len(prompt)
        tok_log = {k: [] for k in TOKEN_FIELDS}
        calls, t_log, ttft = [], 0.0, None
        t0 = sync_time()
        self.target.prefill(committed[:-1])
        t_prefill = sync_time() - t0
        while len(committed) - n_prompt < max_new:
            ta = sync_time()
            logits = self.target.feed([committed[-1]], 1).float()
            tb = sync_time()
            p = law(logits)[0].double()
            window = tuple(committed[-wm_args.ngram:])
            keyed = wm_args.watermark_type == "textseal2" and window not in used
            if keyed:
                E = self.clocks(window, p.shape[0], keys)
                used.add(window)
            else:
                E = self.fresh_clocks(p.shape[0])
            x = int(race(E, p))
            tc = sync_time()
            tl = sync_time()
            values = ("ar", len(calls), keyed, float(p[x]), NAN, NAN, float(entropy(p)), float(entropy(torch.softmax(logits[0].double(), -1))),
                      NAN, NAN, NAN, int((p > 0).sum()), float(E[x]), None, 2 if keyed else -1)
            for k, v in zip(TOKEN_FIELDS, values):
                tok_log[k].append(v)
            t_log += sync_time() - tl
            calls.append({"n_drafts": 0, "kept": 0, "emitted": 1, "t": {"target_fwd": round(tb - ta, 6), "target_post": round(tc - tb, 6)}})
            committed.append(x)
            if ttft is None:
                ttft = sync_time() - t0 - t_log
            if x in self.eos:
                break
        return self._finish(committed, n_prompt, tok_log, calls, t0, t_prefill, ttft, t_log)


if __name__ == "__main__":
    # Baselines under block verification on toy laws, fresh windows each trial (keys average out): the first emitted
    # token follows p_0 and the number of kept drafts follows native block verification (not textGrain's own recipe).
    import numpy as np
    from scipy import stats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    V, N, trials = 12, 3, 6000
    gen = torch.Generator(device=device).manual_seed(1)
    P = torch.softmax(torch.randn(N + 1, V, device=device, dtype=torch.float64, generator=gen) * 1.5, -1)
    Qd = torch.softmax(torch.randn(N, V, device=device, dtype=torch.float64, generator=gen) * 1.5, -1)
    cases = [("textseal", {"mixing_alpha": 0.7}), ("automark", {"rho": 0.9, "prf": "hires"}), ("synthid", {"depth": 5}),
             ("textgrain", {"textgrain_beta": 0.5}), ("textgrain", {"textgrain_beta": 0.5, "textgrain_verify_watermarked": True})]
    for kind, extra in cases:
        first, kept, kept_ref = np.zeros(V), np.zeros(N + 1), np.zeros(N + 1)
        for t in range(trials):
            bl = BaselineRound(WatermarkConfig(watermark_type=kind, secret_key=11 + 1000 * t, secret_key_b=22 + 1000 * t, ngram=3, **extra))
            bl.start_round()
            windows = [(t, ii, 7) for ii in range(N)]
            D = torch.tensor([bl.draft(Qd[ii], windows[ii]) for ii in range(N)], device=device)
            L, X, _ = bl.decide(P, Qd, D, windows, (t, N, 7))
            first[int(D[0]) if L > 0 else X] += 1
            kept[L] += 1
            D_ref = torch.stack([torch.multinomial(Qd[ii].float(), 1)[0] for ii in range(N)])
            kept_ref += verification.block_law(P, Qd, D_ref)[0].cpu().numpy()
        p_first = stats.chisquare(first, P[0].cpu().numpy() * trials).pvalue
        p_kept = stats.chisquare(kept, kept_ref / kept_ref.sum() * trials).pvalue
        print(f"{kind} {extra}: first token ~ p_0: chi2 p = {p_first:.3f}   kept drafts ~ native: chi2 p = {p_kept:.3f}")
