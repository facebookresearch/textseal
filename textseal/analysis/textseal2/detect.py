# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Detection for the TextSeal2 paper: p-values of every run with its own detector, empirical thresholds, entropy weighting.

For each <gen_dir>/<dataset>/det/<config>, the detector of the configuration (common.detector_config, textseal.watermarking
detectors on the generated token ids, each (window, token) pair scored once) gives one p-value per answer and length.
Speculative baselines fuse their two keys weighted by the shares of emitted tokens under each key. Thresholds are the
empirical 1e-3 quantile of the same detector's p-values over 10 keys x the unwatermarked answers of --null_dir (the
paper uses 10,000 unwatermarked answers of the target model); never the nominal p-value.
With --entropy (GPU stage `entropy` first), also TextSeal's entropy-weighted test on the same per-position scores, with
next-token entropies of the target model on the answer alone (thresholds calibrated the same way).
Writes <out>/pv_<dataset>_<config>.json and caches thresholds in <out>/null_<dataset>_<detector>.json.

    python -m textseal.analysis.textseal2.detect pv --gen_dirs runs/spec --null_dir runs/null/eli5/det/ns_none --out runs/detect
"""

import argparse
import glob
import json
import os
import pickle
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import special

from textseal.analysis.textseal2.common import METHOD, TARGET, detector_config, is_speculative, parse_config, read_records
from textseal.watermarking.detector import build_detector

NULL_KEYS = [42 + 100_000 * jj for jj in range(10)]
_DETECTORS: dict = {}


def _detector(cfg_json: str):
    if cfg_json not in _DETECTORS:
        from textseal.watermarking.config import WatermarkConfig

        _DETECTORS[cfg_json] = build_detector(None, WatermarkConfig(**json.loads(cfg_json)))
    return _DETECTORS[cfg_json]


def scores_of(cfg_json: str, ids: list[int], key: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-position scores and kept positions of token ids under base secret `key`."""
    det = _detector(json.dumps({**json.loads(cfg_json), "secret_key": key}))
    scores, mask = det.get_scores_by_t([ids], scoring_method="v2", seen_windows=set(), return_aux=True)
    return np.asarray(scores[0], dtype=np.float64), np.flatnonzero(mask[0]) + det.ngram


def pvalue_job(job: tuple) -> tuple[float, float]:
    """(key-only p-value, entropy-weighted p-value or nan) of one answer."""
    cfg_json, ids, key, entropies = job
    if len(ids) <= 4:
        return 1.0, 1.0
    det = _detector(json.dumps({**json.loads(cfg_json), "secret_key": key}))
    scores, kept = scores_of(cfg_json, ids, key)
    if len(scores) == 0:
        return 1.0, 1.0
    p = float(det.get_pvalue(float(scores.sum()), len(scores), 1e-300))
    if entropies is None:
        return p, float("nan")
    unit, theta = unit_scores(det, scores)
    return p, weighted_pvalue(unit, np.asarray([entropies[t] for t in kept]), theta)


def unit_scores(det, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Scores rescaled to mean 1 without watermark, and their null variance per position (key weights a, 1 - a)."""
    alpha = getattr(det, "alpha", 1.0)
    mix = alpha**2 + (1 - alpha) ** 2
    if type(det).__name__ == "SynthidDetector":  # Sum of `depth` fair coins: mean depth / 2, variance depth / 4.
        depth = det.wm_args.depth
        return 2 * scores / depth, mix / depth
    return scores, mix


def weighted_pvalue(scores: np.ndarray, entropies: np.ndarray, theta: float) -> float:
    """TextSeal's entropy-weighted test on mean-1 scores: weights 0.1 + 0.9 (H - H_min) / (H_max - H_min), moment-matched
    Gamma null with variance theta per position."""
    lo, hi = entropies.min(), entropies.max()
    if hi - lo < 1e-6:
        lo, hi = 0.0, 5.0
    w = 0.1 + 0.9 * np.clip((entropies - lo) / (hi - lo), 0.0, 1.0)
    mu, var = w.sum(), (w**2).sum() * theta
    return float(special.gammaincc(mu**2 / var, (w * scores).sum() / (var / mu))) if var > 1e-10 else 1.0


def key_shares(recs: list[dict]) -> float | None:
    """Share of emitted tokens under key A among keyed tokens (per-token key codes 0 = A, 1 = B)."""
    codes = np.array([k for r in recs if isinstance(r.get("key"), list) for k in r["key"] if k in (0, 1)])
    return round(float(np.mean(codes == 0)), 2) if len(codes) else None


def load_entropies(cache: Path, name: str) -> list | None:
    f = cache / f"{name}.pkl"
    return pickle.load(open(f, "rb"))["ent"] if f.exists() else None


def stage_entropy(args) -> None:
    """Next-token entropies of the target on each answer alone (first max(lengths) tokens), full softmax."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16, device_map="cuda").eval()
    cache = Path(args.out) / "entropies"
    cache.mkdir(parents=True, exist_ok=True)
    L = max(args.lengths)
    jobs = [("null", sorted(glob.glob(os.path.join(args.null_dir, "*.jsonl"))))]
    jobs += [(f"{Path(g).name}_{Path(d).name}", sorted(glob.glob(os.path.join(d, "*.jsonl")))) for g in args.gen_dirs for d in sorted(glob.glob(os.path.join(g, args.dataset, "det", "*")))]
    for name, files in jobs:
        f = cache / f"{name}.pkl"
        recs = [r for fl in files for r in read_records(fl)]
        recs.sort(key=lambda r: (r["row"], r.get("ans", 0)))
        if f.exists() and len(pickle.load(open(f, "rb"))["ent"]) == len(recs):
            continue
        out = []
        with torch.no_grad():
            for start in range(0, len(recs), 32):
                batch = [r["ids"][:L] for r in recs[start : start + 32]]
                width = max(len(x) for x in batch)
                ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
                att = torch.zeros_like(ids)
                for ii, x in enumerate(batch):
                    ids[ii, : len(x)] = torch.tensor(x)
                    att[ii, : len(x)] = 1
                lsm = torch.log_softmax(model(ids, attention_mask=att).logits.float()[:, :-1], -1)  # b L-1 V
                ent = -(lsm.exp() * lsm).sum(-1)
                for ii, x in enumerate(batch):
                    e = np.zeros(len(x), dtype=np.float32)
                    e[1:] = ent[ii, : len(x) - 1].cpu().numpy()
                    out.append(e)
        pickle.dump({"ent": out}, open(f, "wb"))
        print(name, len(out), flush=True)


def stage_pv(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    null_recs = sorted((r for f in sorted(glob.glob(os.path.join(args.null_dir, "*.jsonl"))) for r in read_records(f)), key=lambda r: (r["row"], r.get("ans", 0)))
    null_ids = [r["ids"] for r in null_recs]
    null_ent = load_entropies(out / "entropies", "null") if args.entropy else None
    print(f"null: {len(null_ids)} answers x {len(NULL_KEYS)} keys", flush=True)
    pool = Pool(args.workers)

    def thresholds(cfg_json: str) -> dict:
        tag = hashlib_tag(cfg_json)
        f = out / f"null_{args.dataset}_{tag}.json"
        if f.exists():
            saved = json.load(open(f))
            if all(str(L) in saved["tau"] for L in args.lengths) and (not args.entropy or "tau_weighted" in saved):
                return saved
        res = {"detector": json.loads(cfg_json), "n_null": len(null_ids) * len(NULL_KEYS), "tau": {}, "fpr_at_nominal": {}, "tau_weighted": {}}
        for L in args.lengths:
            jobs = [(cfg_json, x[:L], k, None if null_ent is None else e[:L]) for k in NULL_KEYS for x, e in zip(null_ids, null_ent or [None] * len(null_ids))]
            p0 = np.array(pool.map(pvalue_job, jobs, chunksize=256))
            q = int(1e-3 * len(p0)) - 1
            res["tau"][str(L)] = float(np.sort(p0[:, 0])[q])
            res["fpr_at_nominal"][str(L)] = float(np.mean(p0[:, 0] < 1e-3))
            res.setdefault("fpr_curve", {})[str(L)] = [float(np.mean(p0[:, 0] < a)) for a in np.logspace(-4, -1, 60)]
            if args.entropy:
                res["tau_weighted"][str(L)] = float(np.sort(p0[:, 1])[q])
        json.dump(res, open(f, "w"), indent=1)
        print("  null", tag, {L: f"{res['tau'][L]:.2e} (FPR at 1e-3: {res['fpr_at_nominal'][L]:.4f})" for L in res["tau"]}, flush=True)
        return res

    for g in args.gen_dirs:
        for folder in sorted(glob.glob(os.path.join(g, args.dataset, "det", "*"))):
            cfg = Path(folder).name
            scheme, param, _ = parse_config(cfg)
            if scheme not in METHOD:
                continue
            recs = sorted((r for f in sorted(glob.glob(os.path.join(folder, "*.jsonl"))) for r in read_records(f)), key=lambda r: (r["row"], r.get("ans", 0)))
            f_out = out / f"pv_{args.dataset}_{cfg}.json"
            if not recs or (f_out.exists() and json.load(open(f_out))["n"] == len(recs) and (not args.entropy or "p_weighted" in json.load(open(f_out)))):
                continue
            share = key_shares(recs) if is_speculative(scheme) else None
            prf = recs[0].get("meta", {}).get("prf", "hires")
            det_cfg = detector_config(scheme, param, share, prf=prf)
            cfg_json = json.dumps({k: v for k, v in asdict(det_cfg).items()}, sort_keys=True)
            th = thresholds(cfg_json)
            ent = load_entropies(out / "entropies", f"{Path(g).name}_{cfg}") if args.entropy else None
            res = {"config": cfg, "dataset": args.dataset, "n": len(recs), "share": share, "detector": json.loads(cfg_json), "tau": th["tau"],
                   "rows": [r["row"] for r in recs], "ans": [r.get("ans", 0) for r in recs], "p": {}, "det": {}}
            for L in args.lengths:
                jobs = [(cfg_json, r["ids"][:L], 42, None if ent is None else ent[ii][:L]) for ii, r in enumerate(recs)]
                p = np.array(pool.map(pvalue_job, jobs, chunksize=16))
                res["p"][str(L)] = p[:, 0].tolist()
                res["det"][str(L)] = float(np.mean(p[:, 0] < th["tau"][str(L)]))
                if args.entropy:
                    res.setdefault("p_weighted", {})[str(L)] = p[:, 1].tolist()
                    res.setdefault("det_weighted", {})[str(L)] = float(np.mean(p[:, 1] < th["tau_weighted"][str(L)]))
            json.dump(res, open(f_out, "w"))
            print(f"{args.dataset}/{cfg}: n {len(recs)}  share {share}  detected " + " / ".join(f"{100 * res['det'][str(L)]:.1f}" for L in args.lengths), flush=True)


def hashlib_tag(cfg_json: str) -> str:
    import hashlib

    d = json.loads(cfg_json)
    return f"{d['watermark_type']}_{hashlib.md5(cfg_json.encode()).hexdigest()[:10]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pv", "entropy"])
    ap.add_argument("--gen_dirs", nargs="+", required=True)
    ap.add_argument("--null_dir", required=True, help="folder of unwatermarked answers (jsonl records with ids)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="eli5")
    ap.add_argument("--lengths", nargs="+", type=int, default=[100, 200])
    ap.add_argument("--entropy", action="store_true", help="also the entropy-weighted test (run the entropy stage first)")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    args.gen_dirs = " ".join(args.gen_dirs).split()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    stage_entropy(args) if args.stage == "entropy" else stage_pv(args)


if __name__ == "__main__":
    main()
