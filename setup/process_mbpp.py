# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Run with:
    python setup/process_mbpp.py
"""

# https://github.com/google-research/google-research/blob/master/mbpp/

import requests
import re

import shutil
import os
import json

# Download

url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
out_file = "mbpp.jsonl"
response = requests.get(url)
response.raise_for_status()  # will crash if download failed
with open(out_file, "wb") as f:
    f.write(response.content)
print(f"Downloaded file saved to {out_file}")

# Load data
with open(out_file, "r") as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

# Create text and test for each problem
out_file = out_file.replace(".jsonl", "_processed.jsonl")
with open(out_file, "w") as f:
    for ii, problem in enumerate(data):
        text = problem["text"]
        code = problem["code"]
        # replace the instruction by a comment
        text = text.replace("Write a ", "# A ") + "\n"
        text = text + code + "\n"
        # create test function under same format as human eval
        func_defs = re.findall(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', code, re.MULTILINE)
        prefix = "def check(candidate):\n   "
        # take the longest function name that appears in asserts
        func_defs_in_assert = [func_def for func_def in func_defs if func_def in problem["test_list"][0]]
        func_name = max(func_defs_in_assert, key=len)
        # replace function name in asserts with 'candidate' and add check(candidate)
        asserts = "\n   ".join(problem["test_list"]).replace(func_name, "candidate")
        suffix = f"\n\ncheck({func_name})\n"
        test = prefix + asserts + suffix
        # make sure that the last upper level defined function is the one being tested
        top_level_pattern = r'^(?:def|async def)\s+([A-Za-z_]\w*)\s*\('
        top_level_defs = re.findall(top_level_pattern, code, re.MULTILINE)
        if top_level_defs[-1] != func_name:
            print(f"Warning: in problem {ii}, last defined function '{top_level_defs[-1]}' != tested function '{func_name}'")
            while top_level_defs[-1] != func_name:
                pointer1 = code.find(f'def {top_level_defs[0]}(')
                pointer2 = code.find(f'def {top_level_defs[1]}(')
                code = code[:pointer1] + code[pointer2:] + "\n" + code[pointer1:pointer2-1]
                top_level_defs = re.findall(top_level_pattern, code, re.MULTILINE)
        # create the json
        processed_problem = {
            # save the new fields
            "text": text,
            "test": test,
            # keep some original fields
            "task_id": problem["task_id"],
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