# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation utilities for post-hoc watermarking.

This module provides tools for evaluating both watermark quality (detectability)
and text quality (semantic similarity, ROUGE, BLEU) for watermarked documents.
"""

import re
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
import torch

# Ensure NLTK data is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from .config import EvaluationConfig


class WatermarkEvaluator:
    """Handles evaluation of watermark detection and text quality metrics."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None, quality_model=None, quality_tokenizer=None):
        """
        Initialize the watermark evaluator.
        
        Args:
            config: Evaluation configuration (uses defaults if None)
            quality_model: Pre-loaded model for perplexity evaluation (optional)
            quality_tokenizer: Pre-loaded tokenizer for perplexity evaluation (optional)
        """
        self.config = config or EvaluationConfig()
        self.quality_model = quality_model
        self.quality_tokenizer = quality_tokenizer
        
        # Initialize quality evaluation tools
        if self.config.enable_semantic_similarity:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not initialize SentenceTransformer: {e}")
                self.sentence_transformer = None
        if self.config.enable_rouge:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            except Exception as e:
                print(f"Warning: Could not initialize ROUGE scorer: {e}")
                self.rouge_scorer = None
    
    def _evaluate_single_watermark_test_with_entropies(
        self, 
        text: str, 
        detector, 
        watermark_type: str,
        scoring_method: str = "v1", 
        entropy_threshold: float = None,
        precomputed_entropies: list = None
    ) -> dict:
        """
        Evaluate watermark for a single test configuration with optional precomputed entropies.
        
        Args:
            text: Text to evaluate
            detector: Watermark detector instance
            watermark_type: Type of watermark ("greenlist", "gumbelmax", "synthid", etc.)
            scoring_method: Scoring method for evaluation
            entropy_threshold: If set, only score tokens with entropy > threshold
            precomputed_entropies: Precomputed entropy values to avoid recomputation
            
        Returns:
            dictionary with watermark evaluation metrics for this test
        """
        # Initialize seen_windows for deduplication if using v1 or v2 scoring
        seen_windows = set() if scoring_method in ['v1', 'v2'] else None
        
        # Compute filtered scores (with entropy threshold)
        scores, masks_lists = detector.get_scores_by_t(
            [text], 
            scoring_method=scoring_method, 
            return_aux=True,
            entropy_threshold=entropy_threshold,
            seen_windows=seen_windows,
            precomputed_entropies=[precomputed_entropies] if precomputed_entropies else None
        )
        masks_array = np.array(masks_lists[0])  # Convert to numpy array
        scores_array = np.array(scores[0])  # Convert to numpy array
        p_value = detector.get_pvalue(float(np.sum(scores_array)), int(len(scores_array)), eps=1e-200)
        stats = {
            "score": float(np.mean(scores_array)),
            "p_value": float(p_value),
            "det": bool(p_value < self.config.detection_threshold),
            "toks_gen": masks_array.shape[0],
            "toks_scored": int(masks_array.sum()), 
            "entropy_filtered": entropy_threshold is not None,
            "entropy_threshold": entropy_threshold
        }

        if watermark_type in ["greenlist", "dipmark", "synthid"] or watermark_type.startswith("synthid"):
            # Calculate proportion of green tokens
            green_tokens = scores_array.sum()
            total_tokens = len(scores_array)
            if watermark_type.startswith("synthid"):
                # adjust by the depth
                total_tokens *= detector.wm_args.depth
            green_proportion = green_tokens / total_tokens if total_tokens > 0 else 0
            additional_stats = {
                "toks_green": int(green_tokens),
                "toks_tot": int(total_tokens),
                "green_proportion": float(green_proportion),
            }
        else:
            additional_stats = {}

        stats = {**stats, **additional_stats}
        return stats
    
    def _evaluate_single_watermark_test(
        self, 
        text: str, 
        detector, 
        watermark_type: str,
        scoring_method: str = "v1", 
        entropy_threshold: float = None
    ) -> dict:
        """
        Backward compatible wrapper that calls the optimized version.
        """
        return self._evaluate_single_watermark_test_with_entropies(
            text, detector, watermark_type, scoring_method, entropy_threshold, None
        )
    
    def evaluate_watermark(
        self, 
        text: str, 
        detector, 
        watermark_type: str,
        scoring_method: str = "v1", 
        entropy_threshold: float = None,
        tokenizer=None,
        wm_config=None,
        model=None
    ) -> dict:
        """
        Evaluate watermark quality in the text. Supports multiple test configurations.
        
        Args:
            text: Text to evaluate
            detector: Watermark detector instance
            watermark_type: Type of watermark ("greenlist", "gumbelmax", "synthid", etc.)
            scoring_method: Scoring method for evaluation
            entropy_threshold: If set, only score tokens with entropy > threshold (backward compat)
            tokenizer: Tokenizer for building alternate detectors (optional)
            wm_config: WatermarkConfig for building alternate detectors (optional)
            model: Model for building alternate detectors (optional)
            
        Returns:
            If multi-test enabled: {"tests": [{test_name: results}, ...], "primary": primary_results}
            Otherwise: single results dict (backward compatible)
        """
        from apps.posthoc.detector import build_detector
        from dataclasses import replace
        
        # Check if multi-test mode is enabled
        multi_test_enabled = (
            self.config.test_entropy_thresholds is not None or 
            self.config.test_watermark_types is not None
        )
        
        if not multi_test_enabled:
            # Backward compatible: single test
            return self._evaluate_single_watermark_test(
                text, detector, watermark_type, scoring_method, entropy_threshold
            )
        
        # Multi-test mode: run all combinations
        test_results = []
        
        # Determine test configurations
        entropy_thresholds = self.config.test_entropy_thresholds or [entropy_threshold]
        watermark_types = self.config.test_watermark_types or [watermark_type]
        
        # Optimization: Compute entropy once if multiple thresholds are tested
        needs_entropy = any(t is not None for t in entropy_thresholds)
        precomputed_entropies = None
        if needs_entropy and len(entropy_thresholds) > 1:
            # Tokenize once
            if hasattr(detector.tokenizer, 'encode') and hasattr(detector.tokenizer, 'decode') and not hasattr(detector.tokenizer, 'add_special_tokens'):
                tokens = detector.tokenizer.encode(text, add_bos=False, add_eos=False)
            else:
                tokens = detector.tokenizer.encode(text, add_special_tokens=False)
            
            # Compute entropies once
            if detector.model is not None:
                precomputed_entropies = detector.compute_token_entropies(tokens)
        
        for wm_type in watermark_types:
            # Build detector for this watermark type if needed
            if wm_type != watermark_type and tokenizer is not None and wm_config is not None:
                # Create a new config with the alternate watermark type
                alt_config = replace(wm_config, watermark_type=wm_type)
                test_detector = build_detector(tokenizer, alt_config, model)
            else:
                test_detector = detector
            
            for ent_thresh in entropy_thresholds:
                # Run test with precomputed entropies
                result = self._evaluate_single_watermark_test_with_entropies(
                    text, test_detector, wm_type, scoring_method, ent_thresh, precomputed_entropies
                )
                
                # Create test name
                test_name = f"wm={wm_type}"
                if ent_thresh is not None:
                    test_name += f",entropy={ent_thresh}"
                else:
                    test_name += ",entropy=none"
                
                test_results.append({
                    "test_name": test_name,
                    "watermark_type": wm_type,
                    "entropy_threshold": ent_thresh,
                    **result
                })
        
        # Return structured results
        return {
            "tests": test_results,
            "primary": test_results[0] if test_results else {}  # First test as primary for backward compat
        }
        
    def evaluate_quality(self, original_text: str, rephrased_text: str) -> dict:
        """
        Evaluate the quality of rephrased text compared to original.
        
        Args:
            original_text: Original input text
            rephrased_text: Rephrased watermarked text
            
        Returns:
            dictionary with quality metrics
        """
        results = {}

        # BLEU score
        if self.config.enable_bleu:
            try:
                smoothing = SmoothingFunction()
                original_tokens = original_text.split()
                rephrased_tokens = rephrased_text.split()
                
                bleu_score = sentence_bleu(
                    [original_tokens],
                    rephrased_tokens,
                    smoothing_function=smoothing.method1
                )
                results["bleu_score"] = float(bleu_score)
            except Exception as e:
                print(f"Error calculating BLEU: {e}")
                results["bleu_score"] = None
        
        # ROUGE scores
        if self.config.enable_rouge and self.rouge_scorer:
            try:
                rouge_scores = self.rouge_scorer.score(original_text, rephrased_text)
                results["rouge_scores"] = {
                    metric: {
                        "precision": float(score.precision),
                        "recall": float(score.recall),
                        "fmeasure": float(score.fmeasure)
                    }
                    for metric, score in rouge_scores.items()
                }
            except Exception as e:
                print(f"Error calculating ROUGE: {e}")
                results["rouge_scores"] = None
        
        # Semantic similarity
        if self.config.enable_semantic_similarity and self.sentence_transformer:
            try:
                original_embedding = self.sentence_transformer.encode([original_text], convert_to_tensor=True).squeeze(0)
                rephrased_embedding = self.sentence_transformer.encode([rephrased_text], convert_to_tensor=True).squeeze(0)
                similarity = float(torch.nn.functional.cosine_similarity(original_embedding, rephrased_embedding, dim=0).item())
                results["semantic_similarity"] = round(similarity, 4)
            except Exception as e:
                print(f"Error calculating semantic similarity: {e}")
                results["semantic_similarity"] = None
        
        # Length statistics
        results["length_stats"] = {
            "original_length": int(len(original_text)),
            "rephrased_length": int(len(rephrased_text)),
            "length_ratio": float(len(rephrased_text) / len(original_text)) if len(original_text) > 0 else 0.0
        }
        
        # Perplexity with independent quality model
        if self.config.enable_perplexity and self.quality_model is not None and self.quality_tokenizer is not None:
            try:
                perplexity = self._compute_conditional_perplexity(original_text, rephrased_text)
                results["perplexity"] = float(perplexity)
            except Exception as e:
                print(f"Error calculating perplexity: {e}")
                results["perplexity"] = None
        
        return results
    
    def _compute_conditional_perplexity(self, original_text: str, rephrased_text: str) -> float:
        """
        Compute perplexity of rephrased text conditioned on the original text.
        
        Uses the format: "Here is a rephrased version of this text:\n\n[Original Text]\n\nRephrased version:\n\n[Rephrased Text]"
        Only computes loss on the rephrased portion while having the original in context.
        
        Args:
            original_text: Original input text
            rephrased_text: Rephrased watermarked text
            
        Returns:
            Perplexity value
        """
        # Build the prompt with original text as context
        prompt = f"Here is a rephrased version of this text:\n\n{original_text}\n\nRephrased version:\n\n"
        full_text = prompt + rephrased_text
        
        # Tokenize the full text
        if hasattr(self.quality_tokenizer, 'encode') and hasattr(self.quality_tokenizer, 'decode') and not hasattr(self.quality_tokenizer, 'add_special_tokens'):
            # LLaMA-style tokenizer
            tokens = self.quality_tokenizer.encode(full_text, add_bos=True, add_eos=False)
            prompt_tokens = self.quality_tokenizer.encode(prompt, add_bos=True, add_eos=False)
        else:
            # HuggingFace tokenizer
            tokens = self.quality_tokenizer.encode(full_text, add_special_tokens=True)
            prompt_tokens = self.quality_tokenizer.encode(prompt, add_special_tokens=True)
        
        # Determine where the rephrased portion starts
        prompt_len = len(prompt_tokens)
        
        if len(tokens) <= prompt_len:
            # Edge case: rephrased text is empty or too short
            return float('inf')
        
        # Convert to tensor
        device = next(self.quality_model.parameters()).device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.quality_model(input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # Compute loss only on the rephrased portion
        # Shift for next-token prediction: logits[i] predicts token[i+1]
        shift_logits = logits[:, prompt_len-1:-1, :].contiguous()  # Logits for rephrased tokens
        shift_labels = input_ids[:, prompt_len:].contiguous()  # Actual rephrased tokens
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # Get log prob of actual next tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute average negative log likelihood
        avg_nll = -token_log_probs.mean().item()
        
        # Perplexity is exp(avg_nll)
        perplexity = np.exp(avg_nll)
        
        return perplexity

    def evaluate_code(self, code: str, test: str) -> dict:
        """
        Evaluate functional correctness of rephrased code.
        Args:
            code: Code to evaluate
            test: Test to run against the code
        """
        code = uniformize_names(code, test)
        is_correct = check_correctness(code, test, timeout=5.0)
        return is_correct


### Following code mostly from human_eval/evaluation.py

import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile

import multiprocessing
# HF creates warning when using multiprocessing after tokenizers import
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

def unsafe_execute(to_run: str, timeout: float, result):
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(to_run, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(
    code: str, test: str, timeout: float,
) -> bool:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """
    # code = "\n".join([line.rstrip() for line in code.splitlines()])
    # test = "\n".join([line.rstrip() for line in test.splitlines()])
    to_run = f"{code}\n\n{test}"
    # print("---\n", to_run)

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(to_run, timeout, result))
    p.start()
    p.join(timeout=timeout + 1 if timeout is not None else None)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

### End of human_eval code

def uniformize_names(code: str, test: str = None) -> str:
    """
    Rename the last top-level function in `code` to the expected name extracted from `test`
    (looks for the last check(name) call in the test). Does not rename nested (indented)
    function definitions. Returns the possibly-modified code.
    """
    if not test:
        return code

    matches = list(re.finditer(r'check\s*\(\s*([A-Za-z_]\w*)\s*\)', test))
    if not matches:
        return code
    expected_name = matches[-1].group(1)

    # Find top-level function definitions only (no leading indentation)
    top_level_pattern = r'^(?:def|async def)\s+([A-Za-z_]\w*)\s*\('
    top_level_defs = list(re.finditer(top_level_pattern, code, re.MULTILINE))

    if not top_level_defs:
        return code
    old_name = top_level_defs[-1].group(1)

    # If the expected name is already used among top-level defs, do nothing.
    top_level_names = [m.group(1) for m in top_level_defs]
    if expected_name in top_level_names or old_name == expected_name:
        return code

    # Rename all occurences of top-level def by the expected name.
    code = code.replace(f'{old_name}', f'{expected_name}')

    return code
