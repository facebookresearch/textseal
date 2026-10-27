# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Figures of the TextSeal2 paper from analyze.py's tables.json (and detect.py's null calibration). Needs matplotlib.

    fig_pareto[_200]          Fig. 1: diversity vs detection without / with speculative decoding (100 or 200 tokens)
    fig_spec_vs_nonspec[_200] speculative decoding, with TextSeal2 without speculative decoding as a dashed curve
    fig_distinct4             Fig. 1 with diversity measured by distinct 4-grams
    fig_aime                  AIME under speculative decoding, detection at 1024 / 2048 tokens
    fig_calibration           false-positive rate vs nominal threshold, per detector (detect.py null files)

    python -m textseal.analysis.textseal2.figures --analysis runs/analysis --detect runs/detect --ndraft 8
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COL = {"sid": "#b8860b", "ts": "#1f6fd1", "tg": "#17908a", "cop1": "#8e44ad", "ts2": "#d1242f"}
MK = {"sid": "v", "ts": "o", "tg": "D", "cop1": "s", "ts2": "^"}
LAB = {"sid": "SynthID-Text (layers)", "ts": r"TextSeal, coin flip ($\alpha$)", "tg": r"textGrain ($\beta$)", "cop1": r"AutoMark copula ($\rho$)",
       "ts2": r"TextSeal2 ($\rho$)"}
ORDER = ["sid", "ts", "tg", "cop1", "ts2"]
FS = 8
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": FS, "axes.titlesize": FS + 0.5, "axes.labelsize": FS, "legend.fontsize": FS - 0.5,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e8e8e8", "grid.linewidth": 0.6, "axes.edgecolor": "#555555", "axes.linewidth": 0.7, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "pdf.fonttype": 42})
XLAB, YLAB = "Diversity (Self-BLEU, %)", "Detection rate (%)"


def curve(ax, points: list[dict], method: str, length: int, ls: str = "-", mfc: str | None = None, xfun=None, zorder: int = 3) -> None:
    xfun = xfun or (lambda p: p["selfbleu"])
    pts = sorted(points, key=xfun)
    x = [xfun(p) for p in pts]
    y = [100 * p[f"tpr{length}"] for p in pts]
    err = [[a - 100 * p[f"ci{length}"][0] for a, p in zip(y, pts)], [100 * p[f"ci{length}"][1] - a for a, p in zip(y, pts)]]
    ours = method == "ts2"
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=COL[method], elinewidth=0.6, alpha=0.4, capsize=0, zorder=zorder - 1)
    ax.plot(x, y, ls, color=COL[method], marker=MK[method], mfc=mfc or COL[method], mec=COL[method], mew=0.9, lw=2.0 if ours else 1.4,
            ms=5 if ours else 4, zorder=zorder + (2 if ours else 0))


def legend(fig, methods: list[str], y: float = -0.01) -> None:
    handles = [Line2D([], [], color=COL[m], marker=MK[m], lw=2.0 if m == "ts2" else 1.4, ms=5 if m == "ts2" else 4) for m in methods]
    fig.legend(handles, [LAB[m] for m in methods], loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, y), columnspacing=1.0, handlelength=1.6,
               handletextpad=0.4)


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), pad_inches=0.01)
    fig.savefig(out.with_suffix(".png"), dpi=200, pad_inches=0.01)
    plt.close(fig)


def fig_pareto(tables: dict, spec_key: str, length: int, out: Path, xfun=None, xlabel: str = XLAB) -> None:
    settings = [("nonspec/eli5", "(a) Without speculative decoding"), (spec_key, "(b) With speculative decoding")]
    xs = [(xfun or (lambda p: p["selfbleu"]))(p) for st, _ in settings if st in tables for P in tables[st]["all_points"].values() for p in P]
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.15), sharex=True, sharey=True, gridspec_kw={"wspace": 0.13})
    for ax, (st, title) in zip(axes, settings):
        for m in ORDER:
            if st in tables and m in tables[st]["all_points"]:
                curve(ax, tables[st]["all_points"][m], m, length, ls="--" if st == spec_key and m in ("tg", "cop1") else "-", xfun=xfun)
        ax.set_title(title, pad=4)
        ax.set_xlabel(xlabel)
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel(YLAB)
    legend(fig, ORDER)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.3)
    save(fig, out)


def fig_spec_vs_nonspec(tables: dict, spec_key: str, length: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.9, 2.6))
    for m in ORDER:
        if m in tables[spec_key]["all_points"]:
            curve(ax, tables[spec_key]["all_points"][m], m, length)
    if "nonspec/eli5" in tables and "ts2" in tables["nonspec/eli5"]["all_points"]:
        curve(ax, tables["nonspec/eli5"]["all_points"]["ts2"], "ts2", length, ls="--", mfc="white", zorder=2)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(YLAB)
    ax.set_ylim(0, 100)
    ax.set_title(f"With speculative decoding ({spec_key.rsplit('n', 1)[-1]} drafts per round, block verification)", pad=4)
    legend(fig, ORDER, 0.075)
    dashed = Line2D([], [], color=COL["ts2"], marker=MK["ts2"], mfc="white", ls="--", lw=2.0, ms=5)
    fig.legend([dashed], ["TextSeal2 without speculative decoding"], loc="lower center", bbox_to_anchor=(0.5, -0.01), handlelength=2.2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.36)
    save(fig, out)


def fig_aime(tables: dict, key: str, out: Path) -> None:
    t = tables[key]["all_points"]
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.15), sharex=True, gridspec_kw={"wspace": 0.2})
    for ax, length in zip(axes, (1024, 2048)):
        for m in ORDER:
            if m in t and f"tpr{length}" in t[m][0]:
                curve(ax, t[m], m, length, ls="--" if m in ("tg", "cop1") else "-")
        ax.set_title(f"({'a' if length == 1024 else 'b'}) First {length} tokens", pad=4)
        ax.set_xlabel(XLAB)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(YLAB)
    legend(fig, ORDER)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.3)
    save(fig, out)


def fig_calibration(detect_dir: Path, out: Path) -> None:
    names = {"textseal2": ("TextSeal2", COL["ts2"]), "textseal": ("TextSeal", COL["ts"]), "automark": ("AutoMark copula", COL["cop1"]),
             "textgrain": ("textGrain", COL["tg"]), "synthid": ("SynthID-Text", COL["sid"])}
    a = np.logspace(-4, -1, 60)
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.2), sharey=True, gridspec_kw={"wspace": 0.1})
    seen = set()
    for f in sorted(glob.glob(str(detect_dir / "null_eli5_*.json"))):
        res = json.load(open(f))
        kind = res["detector"]["watermark_type"]
        if kind in seen or "fpr_curve" not in res or res["detector"].get("key_routing"):
            continue
        seen.add(kind)
        for ax, length in zip(axes, sorted(res["fpr_curve"], key=int)[:2]):
            ax.plot(a, np.maximum(res["fpr_curve"][length], 1e-6), color=names[kind][1], label=names[kind][0], lw=2.0 if kind == "textseal2" else 1.3)
            ax.set_title(f"{length} tokens", pad=4)
    for ax in axes:
        ax.plot(a, a, color="#8a8a8a", lw=0.8, ls="--")
        ax.axvline(1e-3, color="#8a8a8a", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Nominal p-value threshold")
    axes[0].set_ylabel("False-positive rate")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.3)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, help="analyze.py output folder (tables.json, selfbleu.json)")
    ap.add_argument("--detect", default="", help="detect.py output folder (null calibration files)")
    ap.add_argument("--ndraft", type=int, default=8)
    args = ap.parse_args()
    run = Path(args.analysis)
    tables = json.load(open(run / "tables.json"))
    figs = run / "figures"
    spec_key = f"spec/eli5/n{args.ndraft}"
    if spec_key in tables:
        for length, suffix in ((100, ""), (200, "_200")):
            fig_pareto(tables, spec_key, length, figs / f"fig_pareto{suffix}")
            fig_spec_vs_nonspec(tables, spec_key, length, figs / f"fig_spec_vs_nonspec{suffix}")
        d4 = {k: v["distinct4"] for k, v in json.load(open(run / "selfbleu.json")).items()}
        fig_pareto(tables, spec_key, 100, figs / "fig_distinct4", xfun=lambda p: 100 * float(np.mean(d4[p["key"]])), xlabel="Diversity (distinct 4-grams, %)")
    if f"spec/aime/n{args.ndraft}" in tables:
        fig_aime(tables, f"spec/aime/n{args.ndraft}", figs / "fig_aime")
    if args.detect:
        fig_calibration(Path(args.detect), figs / "fig_calibration")
    print("written to", figs)


if __name__ == "__main__":
    main()
