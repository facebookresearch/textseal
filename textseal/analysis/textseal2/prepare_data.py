# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Prompt files of the TextSeal2 experiments, downloaded from the Hugging Face hub.

    eli5.jsonl      ELI5 questions and human answers: the train split of sentence-transformers/eli5, in its original
                    order, without the 57,574 math-heavy rows listed in eli5_dropped_rows.txt.gz (267,901 rows). The paper
                    uses the questions in file order: the first 3000 for detection, the first 100 for diversity, rows
                    100000-109999 for the unwatermarked calibration set.
    aime_all.jsonl  AIME problems 1983-2024 (gneubig/aime-1983-2024): id, question, answer, year (933 problems).
    mbpp.jsonl      MBPP test split (google-research-datasets/mbpp, tasks 11-510) as code prompts with their assert tests.

    python -m textseal.analysis.textseal2.prepare_data --out_dir data
The scripts read data/eli5.jsonl and data/aime_all.jsonl, or the files in TEXTSEAL2_ELI5 and TEXTSEAL2_AIME; pass
data/mbpp.jsonl to generate.py with --code_file.
"""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

DROPPED_ROWS = Path(__file__).resolve().parent / "eli5_dropped_rows.txt.gz"
MBPP_INSTRUCTION = "Write a Python solution to the following problem. Explain briefly, then give the code.\n\n"


def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{path}: {len(rows)} rows")


def eli5_rows() -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(hf_hub_download("sentence-transformers/eli5", "pair/train-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    dropped = {int(line) for line in gzip.open(DROPPED_ROWS, "rt") if line.strip()}
    assert len(rows) == 325_475, f"unexpected size of sentence-transformers/eli5: {len(rows)}"
    return [{"question": r["question"], "answer": r["answer"]} for ii, r in enumerate(rows) if ii not in dropped]


def aime_rows() -> list[dict]:
    path = hf_hub_download("gneubig/aime-1983-2024", "AIME_Dataset_1983_2024.csv", repo_type="dataset")
    return [{"id": r["ID"], "question": r["Question"], "answer": str(r["Answer"]), "year": r["Year"], "problem_number": r["Problem Number"], "part": r["Part"]}
            for r in csv.DictReader(open(path))]


def mbpp_rows() -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(hf_hub_download("google-research-datasets/mbpp", "full/test-00000-of-00001.parquet", repo_type="dataset")).to_pylist()
    return [{"question": MBPP_INSTRUCTION + r["text"] + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"]), "task_id": r["task_id"]}
            for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(eli5_rows(), os.path.join(args.out_dir, "eli5.jsonl"))
    write_jsonl(aime_rows(), os.path.join(args.out_dir, "aime_all.jsonl"))
    write_jsonl(mbpp_rows(), os.path.join(args.out_dir, "mbpp.jsonl"))


if __name__ == "__main__":
    main()
