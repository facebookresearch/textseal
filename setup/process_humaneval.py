# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Run with:
    python setup/process_humaneval.py
"""

# https://github.com/openai/human-eval

import urllib.request
import gzip
import shutil
import os
import json

# Download and unzip .gz to .jsonl
url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
out_gz = "HumanEval.jsonl.gz"
out_file = "HumanEval.jsonl"
urllib.request.urlretrieve(url, out_gz)
with gzip.open(out_gz, 'rb') as f_in, open(out_file, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
os.remove(out_gz)
print(f"Downloaded {out_gz} and extracted to {out_file}")

# Load data
with open(out_file, "r") as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

# Create text and test for each problem
out_file = out_file.replace(".jsonl", "_processed.jsonl")
with open(out_file, "w") as f:
    for problem in data:
        prompt = problem["prompt"]
        canonical_solution = problem["canonical_solution"]
        text = prompt + canonical_solution
        test = problem["test"] + "\n" + f"check({problem['entry_point']})"
        processed_problem = {
            # save the new fields
            "text": text,
            "test": test,
            # keep some original fields
            "task_id": problem["task_id"],
            "prompt": prompt,
            "canonical_solution": canonical_solution,
        }
        f.write(json.dumps(processed_problem) + "\n")
print(f"Processed data saved to {out_file}")

# Example of processed data
print("Example processed problem:")
with open(out_file, "r") as f:
    lines = f.readlines()
example_problem = json.loads(lines[0])
print(f"available fields: {list(example_problem.keys())}")
print(f"task_id: {example_problem['task_id']}")
print("--- text ---")
print(example_problem["text"])
print("--- test ---")
print(example_problem["test"])