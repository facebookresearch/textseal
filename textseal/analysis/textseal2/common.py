# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared pieces of the TextSeal2 paper scripts: run names -> WatermarkConfig, prompts, records, diversity and intervals.

Run names (config folders) follow the paper code: "<scheme>[_rho<param>][_n<drafts>]", e.g. residj_rho0.9_n8.
Speculative decoding (Qwen3.5-0.8B drafting for Qwen3.5-9B by default):
    none / noneb                 unwatermarked, token / block verification
    residj                       TextSeal2, block verification, residual J race (the paper's method; param = rho)
    ts2b, taub, agreeb, jonly, residjraw    TextSeal2 block-verification ablations and controls
    cop2r, cop2a                 TextSeal2, token verification: residual race / race coupling
    tsb (alpha), amb (rho), tgb (beta), tgpb (beta), sidb (depth)   baselines under block verification
    arnone, arts2                target only (no speculation)
Without speculative decoding (batched WmGenerator): ns_none, ns_ts (alpha), ns_ts2 (rho), ns_am (rho), ns_tg (beta),
ns_sid (depth), with the paper's repeated-window rules.
"""

import json
import math
import os
import re
from dataclasses import replace

import numpy as np

from textseal.watermarking.config import WatermarkConfig

ELI5_FILE = os.environ.get("TEXTSEAL2_ELI5", "/checkpoint/avseal/pfz/data/eli5_full_nomath.jsonl")
AIME_FILE = os.environ.get("TEXTSEAL2_AIME", "/checkpoint/avseal/datasets/llm_evals/aime/aime_all.jsonl")
AIME_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
DRAFT, TARGET = "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B"
WINDOW = 3
UNWATERMARKED = ("none", "noneb", "arnone", "ns_none")
BLOCK_DECISIONS = {"residj": "residual_j", "ts2b": "fresh_stop", "taub": "tau_stop", "agreeb": "agreement", "jonly": "j_only", "residjraw": "raw_j"}
TS2_SCHEMES = tuple(BLOCK_DECISIONS) + ("cop2r", "cop2a", "arts2", "ns_ts2")
# Paper method of each scheme (rows of Table 1 and curves of Fig. 1).
METHOD = {"tsb": "ts", "ns_ts": "ts", "tgb": "tg", "ns_tg": "tg", "tgpb": "tgp", "sidb": "sid", "ns_sid": "sid", "amb": "cop1", "ns_am": "cop1",
          **{s: "ts2" for s in TS2_SCHEMES}}


def parse_config(cfg: str) -> tuple[str, float, int]:
    """'residj_rho0.9_n8' -> ('residj', 0.9, 8); 'none_n4' -> ('none', 0.0, 4); 'ns_ts2_rho0.9' -> ('ns_ts2', 0.9, 0)."""
    m = re.match(r"^([a-z0-9_]+?)(?:_rho([0-9.]+))?(?:_n(\d+))?$", cfg)
    return m.group(1), float(m.group(2) or 0.0), int(m.group(3) or 0)


def config_name(scheme: str, param: float, ndraft: int) -> str:
    base = scheme if scheme in UNWATERMARKED else f"{scheme}_rho{param}"
    return base if scheme.startswith(("ar", "ns_")) else f"{base}_n{ndraft}"


def is_speculative(scheme: str) -> bool:
    return not scheme.startswith("ns_")


def watermark_config(scheme: str, param: float, ndraft: int, secret: int = 42, prf: str = "hires") -> WatermarkConfig:
    """WatermarkConfig of a run. prf: PRF of the TextSeal2 runs ("hires" 53-bit; "textseal" 13-bit for the Llama pair)."""
    base = WatermarkConfig(secret_key=secret, ngram=WINDOW, num_speculative_tokens=ndraft)
    if scheme in ("arnone", "arts2"):
        return replace(base, watermark_type="textseal2" if scheme == "arts2" else "none", prf=prf, rho=param, num_speculative_tokens=0)
    if scheme in ("none", "noneb"):
        return replace(base, watermark_type="none", verification="token" if scheme == "none" else "block")
    if scheme in ("cop2r", "cop2a"):
        return replace(base, watermark_type="textseal2", prf=prf, rho=param, verification="token", spec_decision="residual" if scheme == "cop2r" else "agreement")
    if scheme in BLOCK_DECISIONS:
        return replace(base, watermark_type="textseal2", prf=prf, rho=param, verification="block", spec_decision=BLOCK_DECISIONS[scheme])
    if scheme == "tsb":
        return replace(base, watermark_type="textseal", mixing_alpha=param)
    if scheme == "amb":
        return replace(base, watermark_type="automark", prf="hires", rho=param)
    if scheme in ("tgb", "tgpb"):
        return replace(base, watermark_type="textgrain", textgrain_beta=param, textgrain_solver="chunked32", textgrain_verify_watermarked=scheme == "tgpb")
    if scheme == "sidb":
        return replace(base, watermark_type="synthid", depth=int(param), synthid_hash="official")
    if scheme == "ns_none":
        return replace(base, watermark_type="none")
    if scheme == "ns_ts":
        return replace(base, watermark_type="textseal", mixing_alpha=param, repeated_windows="per_key")
    if scheme == "ns_ts2":
        return replace(base, watermark_type="textseal2", prf="hires", rho=param, repeated_windows="skip")
    if scheme == "ns_am":
        return replace(base, watermark_type="automark", prf="hires", rho=param, repeated_windows="skip")
    if scheme == "ns_tg":
        return replace(base, watermark_type="textgrain", textgrain_beta=param, repeated_windows="skip")
    if scheme == "ns_sid":
        return replace(base, watermark_type="synthid", depth=int(param), synthid_hash="official", repeated_windows="skip")
    raise ValueError(scheme)


def detector_config(scheme: str, param: float, key_share_a: float | None, secret: int = 42, prf: str = "hires") -> WatermarkConfig | None:
    """Detector configuration of a run, as in the paper. key_share_a: share of emitted tokens under key A (speculative
    baselines draw residual tokens under key B, so their two-key tests weight the keys by the measured shares)."""
    base = WatermarkConfig(secret_key=secret, ngram=WINDOW)
    method = METHOD.get(scheme)
    spec = is_speculative(scheme)
    if method == "ts2":
        return replace(base, watermark_type="textseal2", prf=prf)
    if method == "ts":
        alpha = param if (not spec or param == 0.5) else key_share_a
        return replace(base, watermark_type="textseal", mixing_alpha=alpha)
    if method == "cop1":
        # Under speculative decoding, residual tokens use key B, so the two-key test weights each key by its share.
        if spec:
            return replace(base, watermark_type="textseal", prf="hires", mixing_alpha=key_share_a)
        return replace(base, watermark_type="automark", prf="hires")
    if method == "tg":
        return replace(base, watermark_type="textgrain", key_routing=spec, mixing_alpha=key_share_a if spec else 0.5)
    if method == "tgp":
        return replace(base, watermark_type="textgrain")
    if method == "sid":
        return replace(base, watermark_type="synthid", depth=int(param), synthid_hash="official", key_routing=spec,
                       mixing_alpha=key_share_a if spec else 0.5)
    return None


def load_questions(n: int, offset: int = 0, path: str = ELI5_FILE, suffix: str = "") -> list[str]:
    out = []
    with open(path) as f:
        for ii, line in enumerate(f):
            if ii < offset:
                continue
            if len(out) >= n:
                break
            out.append(json.loads(line)["question"].strip() + suffix)
    return out


def aime_problems(n: int = 100, seed: int = 0, path: str = AIME_FILE) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    pick = sorted(np.random.default_rng(seed).permutation(len(rows))[:n].tolist())
    return [{"id": rows[ii]["id"], "question": rows[ii]["question"], "answer": rows[ii]["answer"], "year": rows[ii]["year"]} for ii in pick]


def chat_prompt(tok, question: str, thinking: bool = False) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return tok.encode(text, add_special_tokens=False)


def boxed_answer(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces), normalised for AIME integer answers."""
    idx = text.rfind("\\boxed")
    start = text.find("{", idx) if idx >= 0 else -1
    if start < 0:
        return None
    depth = 0
    for ii in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[ii], 0)
        if depth == 0:
            ans = re.sub(r"[\s$,]|\\!|\\text\{[^}]*\}", "", text[start + 1 : ii]).replace("^\\circ", "").replace("^{\\circ}", "")
            return str(int(ans)) if re.fullmatch(r"-?\d+", ans) else ans
    return None


def is_correct(pred: str | None, gold) -> bool:
    gold = str(gold).strip()
    return pred is not None and (pred == str(int(gold)) if gold.lstrip("-").isdigit() else pred == gold)


def read_records(path: str) -> list[dict]:
    """Records of a jsonl shard, skipping a last line that a running generator may still be writing."""
    out = []
    for line in open(path):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def self_bleu(texts: list[str]) -> float:
    """Mean sacrebleu sentence BLEU over all ordered pairs of answers (TextSeal's diversity measure)."""
    import sacrebleu

    return float(np.mean([sacrebleu.sentence_bleu(texts[ii], [texts[jj]]).score for ii in range(len(texts)) for jj in range(len(texts)) if ii != jj]))


def distinct_n(seqs: list[list[int]], n_tok: int, n: int = 4) -> float:
    grams = [tuple(s[ii : ii + n]) for s in seqs for ii in range(min(len(s), n_tok) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def wilson(k: int, n: int) -> list[float]:
    z = 1.96
    c = (k + z * z / 2) / (n + z * z)
    h = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [c - h, c + h]


def paired_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> list[float]:
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    return [float(diff.mean()), *np.quantile(diff[idx].mean(1), [0.025, 0.975]).tolist()]


def _interp(xs, ys, x0) -> float:
    o = np.argsort(xs)
    return float(np.interp(x0, np.asarray(xs)[o], np.asarray(ys)[o]))


def matched(points: dict[str, list[tuple]], sbq: dict[str, list[float]], ref_key: str, lengths: list, pairs: list[tuple] = (),
            n_boot: int = 20000, nq_full: int = 100, seed: int = 0) -> dict | None:
    """Detection at the reference Self-BLEU (TextSeal's coin flip, alpha = 0.5), as the paper's tables_paired.py.

    points: {method: [(label, sbq key, {"n": answers, L: detected count})]}; sbq: per-question Self-BLEU per key. Every
    method is interpolated linearly in Self-BLEU; the bootstrap resamples questions jointly for all configurations and
    detection binomially per point, from a per-method stream (adding a method leaves the others' intervals unchanged).
    """
    rng = np.random.default_rng(seed)
    points = {m: [(lab, k, c) for lab, k, c in rows if k in sbq and len(sbq[k]) >= nq_full] for m, rows in points.items()}
    points = {m: rows for m, rows in points.items() if rows}
    if ref_key not in sbq:
        return None
    nq = min(len(sbq[k]) for k in [k for rows in points.values() for _, k, _ in rows] + [ref_key])
    V = {k: np.asarray(sbq[k][:nq]) for k in [k for rows in points.values() for _, k, _ in rows] + [ref_key]}
    x0 = V[ref_key].mean()
    est = {}
    for m, rows in points.items():
        xs = [V[k].mean() for _, k, _ in rows]
        if len(xs) >= 2 and min(xs) <= x0 <= max(xs):
            est[m] = {L: _interp(xs, [c[L] / c["n"] for _, _, c in rows], x0) for L in lengths}
    boot = {m: {L: np.zeros(n_boot) for L in lengths} for m in est}
    rng_m = {m: np.random.default_rng(int.from_bytes(m.encode(), "little")) for m in est}
    for bb in range(n_boot):
        idx = rng.integers(0, nq, nq)
        xb0 = V[ref_key][idx].mean()
        for m in est:
            xs = [V[k][idx].mean() for _, k, _ in points[m]]
            for L in lengths:
                boot[m][L][bb] = _interp(xs, [rng_m[m].binomial(c["n"], c[L] / c["n"]) / c["n"] for _, _, c in points[m]], xb0)
    rows_of = lambda rows: [{"label": lab, "key": k, "selfbleu": float(V[k].mean()), "sb_se": float(V[k].std(ddof=1) / np.sqrt(nq)), "n": c["n"],
                             **{f"tpr{L}": c[L] / c["n"] for L in lengths}, **{f"ci{L}": wilson(c[L], c["n"]) for L in lengths}} for lab, k, c in rows]
    out = {"ref_selfbleu": float(x0), "nq": nq, "all_points": {m: rows_of(rows) for m, rows in points.items()}, "diff": {},
           "methods": {m: {**{f"tpr{L}": est[m][L] for L in lengths}, **{f"ci{L}": np.quantile(boot[m][L], [0.025, 0.975]).tolist() for L in lengths}}
                       for m in est}}
    for a, b in pairs:
        if a in est and b in est and a != b:
            for L in lengths:
                out["diff"][f"{a}-{b}_{L}"] = [est[a][L] - est[b][L], *np.quantile(boot[a][L] - boot[b][L], [0.025, 0.975]).tolist()]
    return out
