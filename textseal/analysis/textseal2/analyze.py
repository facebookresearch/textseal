# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Analysis of TextSeal2 runs (generate.py records + detect.py p-values) into the paper's tables.

Stages (cached in --out):
    selfbleu  per-question Self-BLEU and distinct-4: eli5/div (first 200 tokens), aime/det (first 1024 tokens)
    summary   per configuration: tokens, tokens per target call, acceptance (kept / proposed drafts), share of answers
              ending before the budget, AIME accuracy (last \\boxed{}) paired by problem with the unwatermarked run, mean
              detection score by token source, timings (TTFT, inter-token latency, ms per token per component), and the
              distortion tests (eli5/dist, one key per answer): mean -ln p - H(p) over all tokens and -ln r - H(r) over
              replaced / block-last tokens, paired by question with the unwatermarked run of the same verification
    tables    detection at TextSeal's Self-BLEU (alpha = 0.5), per setting, with paired bootstrap intervals: speculative
              decoding per draft length (Table 1 / number of drafts), AIME, and without speculative decoding
Writes selfbleu.json, summary.json, tables.json and tables.md.

    python -m textseal.analysis.textseal2.analyze all --gen_dirs runs/spec runs/nonspec --pv_dir runs/detect --out runs/analysis
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from textseal.analysis.textseal2.common import (
    METHOD, TARGET, UNWATERMARKED, boxed_answer, distinct_n, is_correct, matched, paired_ci, parse_config, read_records, self_bleu, wilson,
)

TIME_KEYS = ("draft_fwd", "draft_post", "target_fwd", "target_post", "decide", "rollback")
SPEC_PAPER = ("residj", "tsb", "tgb", "tgpb", "sidb", "amb")  # Table 1 / Fig. 1 schemes under speculative decoding.
NONSPEC_PAPER = ("ns_ts2", "ns_ts", "ns_tg", "ns_sid", "ns_am")
METHOD_PAIRS = [("ts2", b) for b in ("ts", "tg", "tgp", "sid", "cop1")]
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(os.environ.get("TEXTSEAL2_TOKENIZER", TARGET))
    return _TOK


def folder_records(folder: str) -> list[dict]:
    return sorted((r for f in sorted(glob.glob(os.path.join(folder, "*.jsonl"))) for r in read_records(f)), key=lambda r: (r["row"], r.get("ans", 0)))


def verification_of(scheme: str) -> str:
    if scheme.startswith("ns_"):
        return "nonspec"
    if scheme.startswith("ar"):
        return "ar"
    return "token" if scheme in ("none", "cop2r", "cop2a") else "block"


def base_config(cfg: str) -> str:
    """Unwatermarked run paired with a configuration (same verification and number of drafts)."""
    scheme, _, nd = parse_config(cfg)
    return {"token": f"none_n{nd}", "block": f"noneb_n{nd}", "ar": "arnone", "nonspec": "ns_none"}[verification_of(scheme)]


def _selfbleu_folder(job: tuple) -> tuple[str, dict]:
    folder, key, n_tok, n_answers = job
    by_q = defaultdict(list)
    for r in folder_records(folder):
        by_q[r["row"]].append((r.get("ans", 0), r["ids"]))
    tok = tokenizer()
    out = {}
    for q, answers in sorted(by_q.items()):
        if len(answers) >= n_answers:
            seqs = [ids for _, ids in sorted(answers)[:n_answers]]
            out[q] = (self_bleu([tok.decode(s[:n_tok], skip_special_tokens=True) for s in seqs]), distinct_n(seqs, n_tok))
    return key, out


def stage_selfbleu(gen_dirs: list[str], out: Path, workers: int, n_answers: dict) -> None:
    f_out = out / "selfbleu.json"
    res = json.load(open(f_out)) if f_out.exists() else {}
    jobs = []
    for g in gen_dirs:
        for folder in sorted(glob.glob(os.path.join(g, "eli5", "div", "*"))):
            cfg = Path(folder).name
            jobs.append((folder, f"eli5/{cfg}", 200, n_answers["nonspec" if cfg.startswith("ns_") else "spec"]))
        for folder in sorted(glob.glob(os.path.join(g, "aime", "det", "*"))):
            jobs.append((folder, f"aime/{Path(folder).name}", 1024, n_answers["aime"]))
    with Pool(workers) as pool:
        for key, by_q in pool.imap_unordered(_selfbleu_folder, jobs):
            qs = sorted(by_q)
            res[key] = {"questions": qs, "selfbleu": [by_q[q][0] for q in qs], "distinct4": [by_q[q][1] for q in qs]}
            print(f"{key}: {len(qs)} complete question sets, Self-BLEU {np.mean(res[key]['selfbleu']) if qs else float('nan'):.2f}", flush=True)
    json.dump(res, open(f_out, "w"))


def _summary_row(r: dict) -> dict:
    """Compact per-answer row from a record."""
    scheme, _, _ = parse_config(r["config"])
    n = len(r["ids"])
    row = {k: r.get(k) for k in ("dataset", "kind", "config", "row", "ans")}
    row.update(n=n, ended=bool(r.get("ended_eos", False)))
    if "calls" in r:
        row.update(n_calls=r["n_calls"], proposed=r["proposed"], accepted=r["accepted"], ttft=r["ttft"], wall=r["wall"] - r["t_log"],
                   **{"t_" + k: sum(c["t"].get(k, 0.0) for c in r["calls"]) for k in TIME_KEYS})
        p, H = np.array(r["p_tok"], float), np.array(r["H_p"], float)
        ok = p > 0
        row.update(sum_d=float((-np.log(p[ok]) - H[ok]).sum()), n_d=int(ok.sum()))
        rr = np.array([np.nan if v is None else v for v in r["r_tok"]], float)
        Hr = np.array([np.nan if v is None else v for v in r["H_r"]], float)
        okr = np.isfinite(rr) & (rr > 0)
        row.update(sum_dr=float((-np.log(rr[okr]) - Hr[okr]).sum()), n_dr=int(okr.sum()), src={s: r["src"].count(s) for s in set(r["src"])})
    if r.get("dataset") == "aime":
        row["correct"] = is_correct(boxed_answer(tokenizer().decode(r["ids"], skip_special_tokens=True)), r["gold"])
    return row


def _summary_folder(folder: str) -> list[dict]:
    return [_summary_row(r) for r in folder_records(folder)]


def stage_summary(gen_dirs: list[str], out: Path, workers: int) -> None:
    folders = [d for g in gen_dirs for d in sorted(glob.glob(os.path.join(g, "*", "*", "*")))]
    with Pool(workers) as pool:
        rows = [row for rs in pool.imap_unordered(_summary_folder, folders) for row in rs]
    rng = np.random.default_rng(0)
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[(r["dataset"], r["kind"], r["config"])].append(r)
    summary = {"configs": {}, "distortion": {}, "aime_paired_accuracy": {}, "timing": {}}
    for (ds, kind, cfg), R in sorted(by_cfg.items()):
        e = {"n": len(R), "tokens": float(np.mean([r["n"] for r in R])), "ended": float(np.mean([r["ended"] for r in R]))}
        if "n_calls" in R[0]:
            e.update(tok_per_call=sum(r["n"] for r in R) / sum(r["n_calls"] for r in R), acceptance=sum(r["accepted"] for r in R) / max(sum(r["proposed"] for r in R), 1))
            tok = sum(r["n"] for r in R)
            summary["timing"][f"{ds}/{kind}/{cfg}"] = {"ttft": float(np.mean([r["ttft"] for r in R])), "ms_per_token": 1000 * sum(r["wall"] for r in R) / tok,
                                                       "itl_ms": 1000 * float(np.mean([(r["wall"] - r["ttft"]) / max(r["n"] - 1, 1) for r in R])),
                                                       **{f"ms_per_token_{k}": 1000 * sum(r["t_" + k] for r in R) / tok for k in TIME_KEYS}}
        if ds == "aime":
            c = sum(r["correct"] for r in R)
            e["accuracy"] = [c / len(R), *wilson(c, len(R))]
        summary["configs"][f"{ds}/{kind}/{cfg}"] = e
    # Distortion over keys, paired with the unwatermarked run of the same verification.
    dist = {cfg: {r["row"]: r for r in R} for (ds, kind, cfg), R in by_cfg.items() if kind == "dist"}
    for cfg, R in sorted(dist.items()):
        base = base_config(cfg)
        e = {"n": len(R), "mean_d": sum(r["sum_d"] for r in R.values()) / max(sum(r["n_d"] for r in R.values()), 1),
             "mean_dr": sum(r["sum_dr"] for r in R.values()) / max(sum(r["n_dr"] for r in R.values()), 1)}
        if cfg != base and base in dist:
            common = sorted(set(R) & set(dist[base]))
            for name, num, den in (("d", "sum_d", "n_d"), ("dr", "sum_dr", "n_dr")):
                m = np.array([[R[q][num], R[q][den], dist[base][q][num], dist[base][q][den]] for q in common], float)
                f = lambda a: a[:, 0].sum() / max(a[:, 1].sum(), 1) - a[:, 2].sum() / max(a[:, 3].sum(), 1)
                e[f"paired_{name}"] = [f(m), *np.quantile([f(m[rng.integers(0, len(m), len(m))]) for _ in range(2000)], [0.025, 0.975]).tolist()]
            if verification_of(parse_config(cfg)[0]) in ("token", "block"):
                acc = lambda rr, q: rr[q]["accepted"] / max(rr[q]["proposed"], 1)
                e["paired_acceptance"] = paired_ci(np.array([acc(R, q) - acc(dist[base], q) for q in common]), rng)
                e["paired_tok_per_call"] = paired_ci(np.array([R[q]["n"] / R[q]["n_calls"] - dist[base][q]["n"] / dist[base][q]["n_calls"] for q in common]), rng)
        summary["distortion"][cfg] = e
    # AIME accuracy and tokens per call, paired by problem with the unwatermarked run.
    aime = {cfg: R for (ds, kind, cfg), R in by_cfg.items() if ds == "aime"}
    for cfg, R in aime.items():
        base = base_config(cfg)
        if cfg == base or base not in aime:
            continue
        per_q = lambda rows, f: {q: float(np.mean([f(r) for r in rows if r["row"] == q])) for q in {r["row"] for r in rows}}
        out_e = {}
        for name, f in (("accuracy", lambda r: float(r["correct"])), ("tok_per_call", lambda r: r["n"] / max(r.get("n_calls", r["n"]), 1))):
            a, b = per_q(R, f), per_q(aime[base], f)
            qs = sorted(set(a) & set(b))
            out_e[name] = paired_ci(np.array([a[q] - b[q] for q in qs]), rng)
        summary["aime_paired_accuracy"][cfg] = out_e
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"summary: {len(rows)} answers, {len(by_cfg)} configurations", flush=True)


def stage_tables(out: Path, pv_dir: Path, ndrafts: list[int]) -> None:
    sb = json.load(open(out / "selfbleu.json"))
    sbq = {k: v["selfbleu"] for k, v in sb.items()}
    pv = {f.stem[3:]: json.load(open(f)) for f in sorted(pv_dir.glob("pv_*.json"))}
    tables = {}

    def setting(ds: str, select, ref_cfg: str, nq_full: int):
        points = defaultdict(list)
        for key, res in pv.items():
            d, cfg = key.split("_", 1)
            scheme, param, nd = parse_config(cfg)
            if d != ds or not select(scheme, nd) or scheme in UNWATERMARKED:
                continue
            sbk = f"{ds}/{cfg}" if ds == "aime" else f"eli5/{cfg}"
            if sbk not in sbq:
                continue
            counts = {"n": res["n"], **{L: int(np.sum(np.array(res["p"][L]) < res["tau"][L])) for L in res["p"]}}
            points[METHOD[scheme]].append((f"{param}", sbk, counts))
        lengths = sorted({L for rows in points.values() for _, _, c in rows for L in c if L != "n"}, key=int)
        ref = f"{ds}/{ref_cfg}" if ds == "aime" else f"eli5/{ref_cfg}"
        return matched(dict(points), sbq, ref, lengths, METHOD_PAIRS, nq_full=nq_full)

    for nd in ndrafts:
        tables[f"spec/eli5/n{nd}"] = setting("eli5", lambda s, n, nd=nd: s in SPEC_PAPER and n == nd, f"tsb_rho0.5_n{nd}", 100)
        tables[f"spec/aime/n{nd}"] = setting("aime", lambda s, n, nd=nd: s in SPEC_PAPER and n == nd, f"tsb_rho0.5_n{nd}", 100)
    tables["nonspec/eli5"] = setting("eli5", lambda s, n: s in NONSPEC_PAPER, "ns_ts_rho0.5", 100)
    tables = {k: v for k, v in tables.items() if v}
    json.dump(tables, open(out / "tables.json", "w"), indent=1)
    lines = ["# TextSeal2 tables", ""]
    for name, t in tables.items():
        Ls = sorted({k[3:] for v in t["methods"].values() for k in v if k.startswith("tpr")}, key=int)
        lines += [f"## {name}: detection (%) at TextSeal's Self-BLEU {t['ref_selfbleu']:.2f} ({t['nq']} questions)", "",
                  "| method | " + " | ".join(f"{L} tokens" for L in Ls) + " |", "|---|" + "---|" * len(Ls)]
        for m, v in t["methods"].items():
            lines.append(f"| {m} | " + " | ".join(f"{100 * v[f'tpr{L}']:.1f} [{100 * v[f'ci{L}'][0]:.1f}, {100 * v[f'ci{L}'][1]:.1f}]" for L in Ls) + " |")
        for k, v in sorted(t["diff"].items()):
            lines.append(f"- {k}: {100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]")
        lines.append("")
    if (out / "summary.json").exists():
        s = json.load(open(out / "summary.json"))
        lines += ["## Per configuration", "", "| run | n | tokens/call | acceptance | ended | AIME accuracy |", "|---|---|---|---|---|---|"]
        for key, e in s["configs"].items():
            acc = f"{100 * e['accuracy'][0]:.1f}" if "accuracy" in e else ""
            lines.append(f"| {key} | {e['n']} | {e.get('tok_per_call', float('nan')):.2f} | {100 * e.get('acceptance', float('nan')):.1f} | {100 * e['ended']:.0f} | {acc} |")
        lines += ["", "## Distortion over keys, paired with the unwatermarked run", "", "| run | all tokens | replaced / last tokens | acceptance (pts) | tokens/call |", "|---|---|---|---|---|"]
        fmt = lambda e, k, sc=1.0: "" if k not in e else f"{sc * e[k][0]:+.4f} [{sc * e[k][1]:+.4f}, {sc * e[k][2]:+.4f}]"
        for cfg, e in s["distortion"].items():
            lines.append(f"| {cfg} | {fmt(e, 'paired_d')} | {fmt(e, 'paired_dr')} | {fmt(e, 'paired_acceptance', 100)} | {fmt(e, 'paired_tok_per_call')} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selfbleu", "summary", "tables", "all"])
    ap.add_argument("--gen_dirs", nargs="+", default=[])
    ap.add_argument("--pv_dir", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ndrafts", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--n_answers", nargs=3, type=int, default=[16, 32, 16], metavar=("SPEC", "NONSPEC", "AIME"), help="answers per question for Self-BLEU")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    gen_dirs = " ".join(args.gen_dirs).split()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("selfbleu", "all"):
        stage_selfbleu(gen_dirs, out, args.workers, dict(zip(("spec", "nonspec", "aime"), args.n_answers)))
    if args.stage in ("summary", "all"):
        stage_summary(gen_dirs, out, args.workers)
    if args.stage in ("tables", "all"):
        stage_tables(out, Path(args.pv_dir), args.ndrafts)


if __name__ == "__main__":
    main()
