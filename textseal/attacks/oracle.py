"""What an attacker reads from the public detector, at each access level:

    NoBox     nothing: no public detector exists (the uninformed attacker)
    BlackBox  one bit per text: the public decision at a published threshold
    WhiteBox  token-level scores

The public detector is the deployed scheme's single-key detector on key A: same
positions, dedup, scores and p-value as the public channel of
textseal.watermarking.detector.dual_key_channels. The detector deals in scores
only; r, the steering currency the informed attacks bias by, is minted
attacker-side (`steering_r`).

Every oracle has `result(text)`, what the public detector returns for a text: the
p-value (white-box), the decision (black-box) or None (no-box). WhiteBox also reads:

    candidates_after_multi(prefixes, cands, decode)  [{token_id: r}] for appending one token  [rephrasing]
    vocab_after(prefix, token_ids)                   r tensor over a vocabulary                [rephrasing]
    per_token(text)                                  [{pos, token_id, r, score}]               [edits]
    span_stats(text, start, length)                  (worst_r, score sum) over a span          [edits]
"""
import math
from dataclasses import replace

import torch

from textseal.attacks.utils import char_offsets
from textseal.watermarking.core import SYNTHID_ROUND_STRIDE, _M, score_listed_tokens
from textseal.watermarking.detector import build_detector


def _family(wm_config) -> str:
    """Per-token statistic: uniform -log(1 - r), greenlist green bit, synthid tournament wins."""
    if wm_config.watermark_type == "greenlist":
        return "greenlist"
    if wm_config.watermark_type.startswith("synthid"):
        return "synthid"
    return "uniform"


def _kept_positions(token_ids, ngram, scoring_method):
    """Positions the detector scores: from ngram + 1, deduplicated as in get_scores_by_t."""
    seen, kept = set(), []
    for p in range(ngram + 1, len(token_ids)):
        if scoring_method == "v1":
            key = tuple(token_ids[p - ngram:p])
        elif scoring_method == "v2":
            key = tuple(token_ids[p - ngram:p + 1])
        else:
            kept.append(p)
            continue
        if key not in seen:
            seen.add(key)
            kept.append(p)
    return kept


def steering_r(score, family, delta, chance):
    """Score -> steering currency r. Uniform inverts score = -log(1 - r); greenlist maps
    the bit to {1, exp(-delta)}; synthid maps wins above chance to 1, else exp(-delta)."""
    if family == "greenlist":
        return math.exp(-delta * (1.0 - score))
    if family == "synthid":
        return 1.0 if score > chance else math.exp(-delta)
    return 1.0 - math.exp(-score)


def _steering_r_tensor(scores, family, delta, chance):
    if family == "greenlist":
        return torch.exp(-delta * (1.0 - scores))
    if family == "synthid":
        out = torch.full_like(scores, math.exp(-delta))
        out[scores > chance] = 1.0
        return out
    return 1.0 - torch.exp(-scores)


class _PublicDetector:
    """The public detector: key A alone (key_routing off, mixing_alpha 1)."""

    def __init__(self, tokenizer, wm_config):
        self.tokenizer = tokenizer
        self.detector = build_detector(tokenizer, replace(wm_config, key_routing=False, mixing_alpha=1.0))
        self.cfg = getattr(self.detector, "wm_args", None) or self.detector.wm_config  # carries `method`
        self.family = _family(wm_config)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def pvalue_of(self, score_sum, n):
        return self.detector.get_pvalue(score_sum, n, 1e-200) if n > 0 else 1.0

    def score_tokens(self, windows: torch.Tensor, toks: torch.Tensor) -> torch.Tensor:
        """score_tok of the detector for m (window, token) pairs, vectorized."""
        if self.family == "synthid":
            listed = toks.unsqueeze(1) + SYNTHID_ROUND_STRIDE * torch.arange(self.cfg.depth)
            return (score_listed_tokens(windows, self.cfg, listed) * self.detector.weights).sum(dim=-1)
        scores = score_listed_tokens(windows, self.cfg, toks.unsqueeze(1)).squeeze(1)
        return -(1 - scores).log() if self.family == "uniform" else scores

    def per_token(self, token_ids):
        ngram = self.cfg.ngram
        positions = _kept_positions(token_ids, ngram, self.cfg.scoring_method)
        if not positions:
            return []
        windows = torch.tensor([token_ids[p - ngram:p] for p in positions], dtype=torch.long)
        toks = torch.tensor([token_ids[p] for p in positions], dtype=torch.long)
        scores = self.score_tokens(windows, toks).tolist()
        return list(zip(positions, toks.tolist(), scores))

    def pvalue(self, text):
        rows = self.per_token(self.encode(text))
        return self.pvalue_of(sum(s for _, _, s in rows), len(rows))


class _Oracle:
    """Shared plumbing: the public detector, the score -> r map, a one-entry cache."""

    access = None

    def __init__(self, tokenizer, wm_config):
        self.det = _PublicDetector(tokenizer, wm_config)
        self.tok, self.ngram = tokenizer, wm_config.ngram
        self.family, self.delta = self.det.family, float(wm_config.delta)
        self.chance = wm_config.gamma * wm_config.depth
        self.r_levels = _M if self.family == "uniform" else None  # distinct values of a uniform r
        self._last = (None, None)

    def _steering_r(self, score):
        return steering_r(score, self.family, self.delta, self.chance)

    def _cached(self, text, read):
        if text != self._last[0]:
            self._last = (text, read(text))
        return self._last[1]


class WhiteBox(_Oracle):
    """Token-level scores."""

    access = "whitebox"

    def per_token(self, text):
        """[{pos, token_id, r, score}] for every scored position."""
        return self._cached(text, lambda t: [
            {"pos": p, "token_id": tok, "score": s, "r": self._steering_r(s)}
            for p, tok, s in self.det.per_token(self.det.encode(t))])

    def result(self, text):
        """The public p-value of `text`."""
        per = self.per_token(text)
        return self.det.pvalue_of(sum(t["score"] for t in per), len(per))

    def candidates_after_multi(self, prefixes, cands, decode):
        """[{id: r}] for appending each candidate to its prefix.

        prefix + candidate is retokenized in the watermark tokenizer, because that is
        what the detector scores. A candidate that adds no scored position (too short,
        or a deduplicated repeat) reads 0.5.
        """
        texts, owner = [], []
        for i, (pre, ids) in enumerate(zip(prefixes, cands)):
            for tok in ids:
                texts.append(pre + decode(tok))
                owner.append((i, tok))
        out = [{} for _ in prefixes]
        if not texts:
            return out
        windows, toks, at = [], [], []
        sm = self.det.cfg.scoring_method
        for (i, tok), ids in zip(owner, self.tok(texts, add_special_tokens=False)["input_ids"]):
            if _kept_positions(ids, self.ngram, sm)[-1:] != [len(ids) - 1]:
                out[i][tok] = 0.5
                continue
            windows.append(ids[-(self.ngram + 1):-1])
            toks.append(ids[-1])
            at.append((i, tok))
        if at:
            scores = self.det.score_tokens(torch.tensor(windows, dtype=torch.long),
                                           torch.tensor(toks, dtype=torch.long)).tolist()
            for (i, tok), s in zip(at, scores):
                out[i][tok] = self._steering_r(s)
        return out

    def vocab_after(self, prefix_text, wm_token_ids):
        """r for every candidate id after prefix_text, as a tensor. A position that
        would not be scored (prefix too short, or a deduplicated repeat) reads 0.5."""
        ids = self.det.encode(prefix_text)
        toks = torch.as_tensor(wm_token_ids, dtype=torch.long)
        n = self.ngram
        if len(ids) < n + 1:
            return torch.full((len(toks),), 0.5)
        window = torch.tensor(ids[-n:], dtype=torch.long).unsqueeze(0).expand(len(toks), n)
        r = _steering_r_tensor(self.det.score_tokens(window, toks).float(), self.family, self.delta, self.chance)
        sm, ctx = self.det.cfg.scoring_method, ids[-n:]
        earlier = [q for q in range(n + 1, len(ids)) if ids[q - n:q] == ctx]
        if sm == "v1" and earlier:
            r[:] = 0.5
        elif sm == "v2" and earlier:
            r[torch.isin(toks, torch.tensor([ids[q] for q in earlier]))] = 0.5
        return r

    def span_stats(self, text, char_start, span_len):
        """(worst_r, score sum) over scored tokens overlapping the span. worst_r is 1.0
        when the span covers no scored position, so an unscorable edit is never
        mistaken for a clean one."""
        per = {t["pos"]: t for t in self.per_token(text)}
        end = char_start + span_len
        worst, total, hit = 0.0, 0.0, False
        for pos, (cs, ce) in enumerate(char_offsets(text, self.tok)):
            if ce <= char_start or cs >= end or pos not in per:
                continue
            worst = max(worst, per[pos]["r"])
            total += per[pos]["score"]
            hit = True
        return (worst if hit else 1.0), total


class BlackBox(_Oracle):
    """One bit per text: the public decision at a published threshold.

    A bit is constant over the candidates of one decode step, so there is nothing to
    rank on and every ranking read is absent.
    """

    access = "blackbox"

    def __init__(self, tokenizer, wm_config, threshold):
        super().__init__(tokenizer, wm_config)
        self.threshold = float(threshold)

    def result(self, text):
        """True when the public detector flags `text`."""
        return self._cached(text, lambda t: bool(self.det.pvalue(t) < self.threshold))


class NoBox:
    """No public detector: `result` is None, so an attack that tries to steer on it
    fails instead of steering wrong."""

    access = "nobox"

    def result(self, text):
        return None


def build(access, tokenizer, wm_config, threshold=None):
    """The oracle for an access level. `threshold` is the provider's published decision
    threshold, required by the black box."""
    if access == "nobox":
        return NoBox()
    if access == "blackbox":
        if threshold is None:
            raise ValueError("the black box needs the published decision threshold")
        return BlackBox(tokenizer, wm_config, threshold)
    if access == "whitebox":
        return WhiteBox(tokenizer, wm_config)
    raise ValueError(f"access must be nobox, blackbox or whitebox, got {access!r}")
