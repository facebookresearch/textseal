# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared helpers for the attacks: JSONL rows, word spans, and text quality."""
import glob
import json
import re
from difflib import SequenceMatcher

import numpy as np
import torch


def load_jsonl(path):
    """Rows of one JSONL file, or of every file matching a glob, in path order."""
    paths = sorted(glob.glob(path)) if "*" in path else [path]
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    if not rows:
        raise SystemExit(f"no records in {path}")
    return rows


def shard(rows, start, count):
    """Rows [start, start + count), or to the end when count <= 0."""
    return rows[start:] if count <= 0 else rows[start:start + count]


def char_offsets(text, tokenizer):
    """(start, end) char span per token."""
    return tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]


def is_content_word(word):
    """A Unicode letter anywhere: drops punctuation, digits and the underscore."""
    return bool(re.search(r"[^\W\d_]", word))


def word_overlap(original, modified):
    """Fraction of words kept, SequenceMatcher ratio over whitespace-split words."""
    return SequenceMatcher(None, original.split(), modified.split()).ratio()


class QualityScorer:
    """Text quality of an attacked text against its source, the paper's four measures:

      - bertscore_f1:        BERTScore F1, roberta-large layer 17 (distortion = 1 - F1)
      - semantic_similarity: cosine of MiniLM sentence embeddings
      - perplexity_ratio:    GPT-2 ppl(attacked) / ppl(source)
      - word_overlap:        fraction of words kept
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import bert_score
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.ppl_model = AutoModelForCausalLM.from_pretrained(
            "gpt2", dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager").to(self.device).eval()
        # Loaded once: bert_score.score() reloads roberta on every call.
        self.bertscorer = bert_score.BERTScorer(model_type="roberta-large", num_layers=17, lang="en",
                                                rescale_with_baseline=False, device=self.device)

    @torch.no_grad()
    def _perplexity(self, text, max_length=512):
        enc = self.ppl_tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=max_length).to(self.device)
        if enc["input_ids"].shape[1] < 2:
            return float("nan")
        return float(torch.exp(self.ppl_model(**enc, labels=enc["input_ids"]).loss).item())

    def score(self, original, attacked) -> dict:
        emb = self.sim_model.encode([original, attacked], normalize_embeddings=True,
                                    show_progress_bar=False)
        ppl_orig = self._perplexity(original)
        _, _, f1 = self.bertscorer.score([attacked], [original], verbose=False)
        return {
            "bertscore_f1": float(f1[0]),
            "semantic_similarity": float((emb[0] * emb[1]).sum()),
            "perplexity_ratio": (self._perplexity(attacked) / ppl_orig
                                 if ppl_orig > 0 and not np.isnan(ppl_orig) else float("nan")),
            "word_overlap": word_overlap(original, attacked),
        }
