# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Generation for the TextSeal2 paper, with and without speculative decoding.

Work items: (dataset, kind, config, row, ans).
    eli5/det   one answer to each of the first --n_det questions (detection)
    eli5/div   --n_div_answers answers to each of the first --n_div questions (Self-BLEU, distinct-4)
    eli5/dist  one answer per question, a different key per answer (distortion over keys; speculative only)
    aime/det   --n_aime_answers answers to each of --n_aime AIME problems (detection, diversity, accuracy)
    code/det   one answer to each problem of --code_file (jsonl with a question field, e.g. MBPP)
Configs (common.py): speculative schemes through textseal.watermarking.speculative.SpeculativeGenerator (one answer at
a time, full per-token / per-call logs), non-speculative ones (ns_*) through build_generator (batched).
The process runs the items whose index is congruent to --shard modulo --n_shards, skips items already written (a relaunch
resumes) and appends one jsonl record per answer to <out_dir>/<dataset>/<kind>/<config>/shard<k>.jsonl.

Examples (paper settings: temperature 1, top-p 0.95, window 3):
    python -m textseal.analysis.textseal2.generate --out_dir runs/spec --configs residj:0.9 tsb:0.5 noneb --ndrafts 8
    python -m textseal.analysis.textseal2.generate --out_dir runs/nonspec --configs ns_ts2:0.9 ns_ts:0.5 --n_div_answers 32
"""

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from textseal.analysis.textseal2.common import (
    AIME_SUFFIX, DRAFT, TARGET, aime_problems, chat_prompt, config_name, is_speculative, load_questions, watermark_config,
)
from textseal.watermarking.generator import build_generator
from textseal.watermarking.speculative import SpeculativeGenerator


def answer_secret(row: int) -> int:
    """Secret of the one-key-per-answer runs (distortion tests over keys)."""
    return 10_000_000 + 7919 * row


def sig6(obj):
    """Round floats to 6 significant digits, recursively (small records, tiny probabilities kept)."""
    if isinstance(obj, float):
        return obj if math.isnan(obj) or math.isinf(obj) else float(f"{obj:.6g}")
    if isinstance(obj, list):
        return [sig6(v) for v in obj]
    if isinstance(obj, dict):
        return {k: sig6(v) for k, v in obj.items()}
    return obj


def parse_configs(text: list[str]) -> list[tuple[str, float]]:
    out = []
    for c in " ".join(text).split():
        scheme, _, param = c.partition(":")
        out.append((scheme, float(param or 0.0)))
    return out


def build_items(args) -> list[dict]:
    items = []
    ndrafts = [int(n) for n in " ".join(args.ndrafts).split()]
    datasets, kinds = " ".join(args.datasets).split(), " ".join(args.kinds).split()
    aime_cfgs = parse_configs(args.aime_configs) if args.aime_configs else None
    for nd in ndrafts:
        for scheme, param in parse_configs(args.configs):
            if scheme.startswith(("ar", "ns_")) and nd != ndrafts[0]:
                continue  # No drafts: generated once, not per draft length.
            base = dict(scheme=scheme, rho=param, ndraft=nd)
            if "eli5" in datasets and "det" in kinds:
                items += [dict(dataset="eli5", kind="det", row=row, ans=0, seed=0, **base) for row in range(args.n_det)]
            if "eli5" in datasets and "div" in kinds:
                items += [dict(dataset="eli5", kind="div", row=row, ans=a, seed=1 + a, **base) for row in range(args.n_div) for a in range(args.n_div_answers)]
            if "aime" in datasets and "det" in kinds and (aime_cfgs is None or (scheme, param) in aime_cfgs):
                items += [dict(dataset="aime", kind="det", row=row, ans=a, seed=1 + a, **base) for row in range(args.n_aime) for a in range(args.n_aime_answers)]
            if "code" in datasets:
                items += [dict(dataset="code", kind="det", row=row, ans=0, seed=0, **base) for row in range(args.n_code)]
        for scheme, param in parse_configs(args.dist_configs) if "eli5" in datasets and "dist" in kinds else []:
            if not scheme.startswith(("ar", "ns_")) or nd == ndrafts[0]:
                items += [dict(dataset="eli5", kind="dist", scheme=scheme, rho=param, ndraft=nd, row=row, ans=0, seed=0) for row in range(args.n_dist)]
    return items


def item_id(it: dict) -> tuple:
    return (it["dataset"], it["kind"], config_name(it["scheme"], it["rho"], it["ndraft"]), it["row"], it["ans"])


def prompt_of(it: dict, tok, prompts: dict, args) -> tuple[list[int], int, dict]:
    if it["dataset"] == "aime":
        prob = prompts["aime"][it["row"]]
        return chat_prompt(tok, prob["question"] + AIME_SUFFIX), args.max_new_aime, {"problem_id": prob["id"], "gold": prob["answer"], "year": prob["year"]}
    if it["dataset"] == "code":
        return chat_prompt(tok, prompts["code"][it["row"]]), args.max_new_code, {}
    return chat_prompt(tok, prompts["eli5"][it["row"]]), args.max_new_eli5, {}


def write(out_dir: Path, it: dict, shard: int, rec: dict) -> None:
    path = out_dir / it["dataset"] / it["kind"] / rec["config"] / f"shard{shard:04d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(sig6(rec)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1, help="processes on this GPU (e.g. with MPS); shards are split further")
    ap.add_argument("--datasets", nargs="+", default=["eli5"])
    ap.add_argument("--kinds", nargs="+", default=["det", "div"])
    ap.add_argument("--configs", nargs="+", required=True, help="scheme:param, e.g. residj:0.9 tsb:0.5 noneb ns_ts2:0.9")
    ap.add_argument("--aime_configs", nargs="*", default=None, help="subset of --configs for AIME (default: all)")
    ap.add_argument("--dist_configs", nargs="+", default=["none"])
    ap.add_argument("--ndrafts", nargs="+", default=["8"])
    ap.add_argument("--n_det", type=int, default=3000)
    ap.add_argument("--n_div", type=int, default=100)
    ap.add_argument("--n_div_answers", type=int, default=16)
    ap.add_argument("--n_dist", type=int, default=480)
    ap.add_argument("--n_aime", type=int, default=100)
    ap.add_argument("--n_aime_answers", type=int, default=16)
    ap.add_argument("--n_code", type=int, default=500)
    ap.add_argument("--code_file", default="")
    ap.add_argument("--max_new_eli5", type=int, default=200)
    ap.add_argument("--max_new_aime", type=int, default=8192)
    ap.add_argument("--max_new_code", type=int, default=512)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--prf", default="hires", choices=["hires", "textseal"], help="PRF of the TextSeal2 clocks (textseal: 13-bit)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=64, help="non-speculative runs")
    args = ap.parse_args()
    if args.procs > 1:
        script_args = [a for a in sys.argv[1:]]
        children = [subprocess.Popen([sys.executable, "-m", "textseal.analysis.textseal2.generate", *script_args, "--procs", "1",
                                      "--shard", str(args.shard * args.procs + jj), "--n_shards", str(args.n_shards * args.procs)],
                                     env={**os.environ, "PROCS_PER_GPU": str(args.procs)}) for jj in range(args.procs)]
        sys.exit(max(c.wait() for c in children))
    torch.backends.cuda.enable_cudnn_sdp(False)
    items = build_items(args)[args.shard :: args.n_shards]
    out_dir = Path(args.out_dir)
    done = set()
    for f in out_dir.glob(f"*/*/*/shard{args.shard:04d}.jsonl"):
        for line in open(f):
            try:
                done.add(item_id(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [it for it in items if item_id(it) not in done]
    print(f"shard {args.shard}/{args.n_shards}: {len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.target)
    load = lambda name: AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    target = load(args.target)
    draft = load(args.draft) if any(is_speculative(it["scheme"]) and not it["scheme"].startswith("ar") for it in todo) else None
    prompts = {"eli5": load_questions(max(args.n_det, args.n_div, args.n_dist)), "aime": aime_problems(args.n_aime) if "aime" in " ".join(args.datasets) else [],
               "code": load_questions(args.n_code, path=args.code_file) if args.code_file else []}
    code_md5 = hashlib.md5(b"".join(Path(m.__file__).read_bytes() for m in (sys.modules[SpeculativeGenerator.__module__], sys.modules[build_generator.__module__]))).hexdigest()[:12]
    meta = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "procs_per_gpu": int(os.environ.get("PROCS_PER_GPU", "1")), "draft": args.draft,
            "target": args.target, "prf": args.prf, "torch": torch.__version__, "transformers": transformers.__version__, "code_md5": code_md5}
    t_start = time.time()
    spec_items = [it for it in todo if is_speculative(it["scheme"])]
    for jj, it in enumerate(spec_items):
        secret = answer_secret(it["row"]) if it["kind"] == "dist" else 42
        wm_args = watermark_config(it["scheme"], it["rho"], it["ndraft"], secret, args.prf)
        gen = SpeculativeGenerator(draft, target, tok, wm_args)
        torch.manual_seed(1_000_003 * it["seed"] + 1009 * it["row"] + it["ans"] + 7 * it["ndraft"])
        prompt, max_new, extra = prompt_of(it, tok, prompts, args)
        res = gen.generate_one(prompt, max_new, args.temperature, args.top_p)
        rec = {**it, **extra, "config": config_name(it["scheme"], it["rho"], it["ndraft"]), "key": res.pop("key"), "secret": secret,
               "n_prompt": len(prompt), "max_new": max_new, "meta": meta, **res}
        write(out_dir, it, args.shard, rec)
        if jj % 20 == 0 or jj == len(spec_items) - 1:
            print(f"{jj + 1}/{len(spec_items)} {it['dataset']} {rec['config']}  tokens {len(res['ids'])}  tok/call {len(res['ids']) / max(res['n_calls'], 1):.2f}  "
                  f"{(time.time() - t_start) / (jj + 1):.1f}s/item", flush=True)
    # Non-speculative runs, batched per configuration.
    groups: dict = {}
    for it in todo:
        if not is_speculative(it["scheme"]):
            groups.setdefault((it["scheme"], it["rho"], it["dataset"], it["kind"]), []).append(it)
    for (scheme, param, dataset, kind), group in groups.items():
        gen = build_generator(target, tok, watermark_config(scheme, param, 0, 42, args.prf))
        name = config_name(scheme, param, 0)
        for start in range(0, len(group), args.batch_size):
            batch = group[start : start + args.batch_size]
            torch.manual_seed(int(hashlib.md5(f"{name}/{dataset}/{kind}/{batch[0]['row']}/{batch[0]['ans']}".encode()).hexdigest()[:8], 16))
            built = [prompt_of(it, tok, prompts, args) for it in batch]
            t0 = time.time()
            ids = gen.generate([p for p, _, _ in built], max(m for _, m, _ in built), args.temperature, args.top_p, return_ids=True)
            for it, (prompt, max_new, extra), out in zip(batch, built, ids):
                out = out[:max_new]
                write(out_dir, it, args.shard, {**it, **extra, "config": name, "secret": 42, "n_prompt": len(prompt), "max_new": max_new,
                                                "meta": meta, "ids": out, "wall_batch": time.time() - t0})
            print(f"{name} {dataset}/{kind}: {start + len(batch)}/{len(group)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
