# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from dataclasses import dataclass, replace
from scipy import special
from scipy import stats

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from textseal.watermarking.config import WatermarkConfig
from textseal.watermarking.core import score_listed_tokens

class WmDetector():
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            wm_args: WatermarkConfig,
            model: PreTrainedModel = None,
        ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        # Get vocab size generically
        self.vocab_size = self._get_vocab_size()
        # watermark config
        self.wm_args = wm_args
        self.ngram = wm_args.ngram
        self.secret_key = wm_args.secret_key
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size in a generic way."""
        # Try different ways to get vocab size
        if self.model is not None:
            if hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'vocab_size'):
                return self.model.config.text_config.vocab_size
            elif hasattr(self.model.config, 'vocab_size'):
                return self.model.config.vocab_size
        elif hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'get_vocab'):
            return len(self.tokenizer.get_vocab())
        else:
            # Fallback: use a reasonable default
            return 128256  # Common vocab size for modern models

    @torch.no_grad()
    def compute_token_entropies(self, tokens: list[int], temperature: float = 1.0) -> list[float]:
        """
        Compute entropy of model's output distribution for each token position.
        Args:
            tokens_tensor: list of token ids
            temperature: temperature for softmax (default 1.0)
        """
        if self.model is None:
            raise ValueError("Model is required for entropy computation")
        if len(tokens) <= 1:
            return []

        device = next(self.model.parameters()).device
        tokens_tensor = torch.tensor([tokens], device=device)  # Already shape (1, seq_len)
        
        # Forward pass to get all logits (use tokens[:-1] to predict tokens[1:])
        outputs = self.model(tokens_tensor[..., :-1])
        logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # Compute entropy for each position
        probs = torch.softmax(logits / temperature, dim=-1)  # (1, seq_len, vocab_size)
        log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum(dim=-1)  # (1, seq_len)
        
        return entropy.squeeze(0).tolist()
    
    def get_scores_by_t(
        self, 
        texts: list[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        return_aux: bool = False,
        entropy_threshold: float = None,
        seen_windows: set = None,
        precomputed_entropies: list = None
    ) -> list[list[float]]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            return_aux: if True, return masks of scored tokens information
            entropy_threshold: if set, only score tokens with entropy < threshold
            seen_windows: if provided, persist deduplication across multiple calls
            precomputed_entropies: if provided, use these instead of recomputing (list of lists)
        Output:
            score_lists: list of [score increments for every token] for each text
            masks_lists (optional): list of [1 if token is scored, 0 otherwise] for each text
        """
        bsz = len(texts)
        if hasattr(self.tokenizer, 'encode') and hasattr(self.tokenizer, 'decode') and not hasattr(self.tokenizer, 'add_special_tokens'):
            # TikTokenTokenizer from textseal.wmtraining.lingua (doesn't have add_special_tokens method)
            tokens_id = [self.tokenizer.encode(x, add_bos=False, add_eos=False) for x in texts]
        else:
            # HuggingFace tokenizer
            tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        
        # Use precomputed entropies or compute them if threshold is set
        entropies_list = []
        if entropy_threshold is not None:
            if precomputed_entropies is not None:
                # Use precomputed entropies (big optimization for multiple thresholds)
                entropies_list = precomputed_entropies
            else:
                # Compute entropies
                for tokens in tokens_id:
                    entropies = self.compute_token_entropies(tokens)
                    entropies_list.append(entropies)
        
        score_lists = []
        masks_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram + 1
            rts = []  # list of score increments for each token
            mask_scored = []  # stores 1 for token if scored, 0 otherwise
            
            # Get entropies for this text if available
            entropies = entropies_list[ii] if entropies_list else None
            
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                mask_scored += [0]  # 0 by default
                
                # Check entropy threshold if enabled
                if entropy_threshold is not None and entropies is not None:
                    # entropies[i] corresponds to token at position i+1
                    entropy_idx = cur_pos - 1
                    if entropy_idx < len(entropies):
                        token_entropy = entropies[entropy_idx]
                        if token_entropy < entropy_threshold:
                            continue  # Skip this token due to low entropy (too predictable)
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if seen_windows is not None:
                        if tup_for_unique in seen_windows:
                            continue
                        seen_windows.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if seen_windows is not None:
                        if tup_for_unique in seen_windows:
                            continue
                        seen_windows.add(tup_for_unique)
                mask_scored[-1] = 1  # 1 since we are scoring this token
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos])
                rts.append(rt)
            score_lists.append(rts)
            masks_lists.append(mask_scored)
        if return_aux:
            return score_lists, masks_lists
        return score_lists

    def get_pvalues(
            self, 
            scores: list[list[float]], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            scores: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text
        """
        pvalues = []
        scores = np.asarray(scores)  # bsz x ntoks
        for ss in scores:
            ntoks = ss.shape[0]
            final_score = ss.sum(axis=0) if ntoks != 0 else -1.0
            pval = self.get_pvalue(final_score, ntoks, eps=eps)
            pvalues.append(pval)
        return np.asarray(pvalues)  # bsz

    def get_pvalues_by_t(
            self, 
            scores: list[float],
            eps: float=1e-200
        ) -> list[float]:
        """Get p-value for each text, at each scored token."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks, eps)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: list[int], token_id: int) -> float:
        """ for each token in the text, compute the score increment """
        return 0
    
    def get_pvalue(self, score: float, ntoks: int, eps: float) -> float:
        """ compute the p-value for a couple of score and number of tokens """
        return 0.5


class GreenlistDetector(WmDetector):

    def __init__(self, 
            tokenizer: PreTrainedTokenizer,
            wm_args: WatermarkConfig,
            model: PreTrainedModel = None,
            **kwargs):
        super().__init__(tokenizer, wm_args, model, **kwargs)
        self.gamma = wm_args.gamma
        self.delta = wm_args.delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        """
        # Use the unified scoring function for the GreenlistDetector
        ngram_tokens_tensor = torch.tensor(ngram_tokens).unsqueeze(0)  # Shape: (1, ngram)
        scores = score_listed_tokens(ngram_tokens_tensor, self.wm_args, [token_id])
        return scores[0, 0].item()
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ 
        Compute p-value from binomial distribution with mid-p correction.
        Mid-p = P(X > score) + 0.5 * P(X = score)
        This improves uniformity of p-values under H0 for discrete distributions.
        """
        # P(X >= score) using upper tail
        pvalue_upper = special.betainc(score, 1 + ntoks - score, self.gamma)
        # P(X = score) using binomial PMF
        pmf = stats.binom.pmf(score, ntoks, self.gamma)
        # Mid-p correction
        pvalue = pvalue_upper - 0.5 * pmf
        return max(pvalue, eps)


class GumbelmaxDetector(WmDetector):

    def __init__(self, 
            tokenizer: PreTrainedTokenizer,
            wm_args: WatermarkConfig,
            model: PreTrainedModel = None,
            **kwargs):
        super().__init__(tokenizer, wm_args, model, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id])
        """
        # Use the unified scoring function for the GumbelmaxDetector
        ngram_tokens_tensor = torch.tensor(ngram_tokens).unsqueeze(0)  # Shape: (1, ngram)
        scores = score_listed_tokens(ngram_tokens_tensor, self.wm_args, [token_id])
        score_log = -(1 - scores[0, 0]).log()
        return score_log.item()
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)


class SynthidDetector(WmDetector):

    def __init__(self, 
            tokenizer: PreTrainedTokenizer,
            wm_args: WatermarkConfig,
            model: PreTrainedModel = None,
            weighted: bool = False,
            **kwargs):
        super().__init__(tokenizer, wm_args, model, **kwargs)
        self.weighted = weighted
        self._compute_weights()
    
    def _compute_weights(self):
        """Compute and normalize the weights alpha_ell for weighted scoring."""
        # Get settings.
        d = self.wm_args.depth
        gamma = self.wm_args.gamma

        if self.weighted:
            # Compute weights: alpha_1 = kappa = 10, ..., alpha_m = mu = 1
            weights = np.linspace(10.0, 1.0, d)
            weights = weights * d / weights.sum() # Normalize
        else:
            weights = np.ones(d)
        self.weights = torch.tensor(weights).float()

        # Precompute mean and variance for weighted Z-test under null (Bernoulli(gamma))
        # null_mean = gamma * sum(alpha_ell) = gamma * m (since weights sum to m)
        # null_variance = gamma*(1-gamma) * sum(alpha_ell^2)
        self.null_mean = gamma * d
        self.null_variance = (gamma * (1.0 - gamma)) * (weights ** 2).sum()

    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_depth = 1 if token_id in greenlist_depth else 0 
        """
        ngram_tokens_tensor = torch.tensor(ngram_tokens).unsqueeze(0)  # Shape: (1, ngram)
        scores = score_listed_tokens(
            ngram_tokens_tensor, 
            self.wm_args, 
            [token_id + dd for dd in range(0, self.wm_args.depth)]
        ) # Shape: (1, depth)
        weighted_score = (scores[0] * self.weights).sum() # sum over depth
        return weighted_score.item()
                
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        if not self.weighted:
            # From cdf of a binomial distribution with mid-p correction.
            # Here score is sum over depths, so ntoks is multiplied by depth.
            total_trials = ntoks * self.wm_args.depth
            pvalue_upper = special.betainc(
                score, 
                1 + total_trials - score, 
                self.wm_args.gamma
            )
            # Mid-p correction
            pmf = stats.binom.pmf(int(score), total_trials, self.wm_args.gamma)
            pvalue = pvalue_upper - 0.5 * pmf
        else:
            """ 
            Z-test based p-value for weighted sum.
            The score is the sum over T tokens of weighted sums.
            Under null: Normal(T * null_mean, T * null_variance)
            We compute: 1 - CDF(score / (ntoks * m))
            """
            if ntoks == 0:
                return 1.0
            avg_score = score / ntoks 
            std_dev = np.sqrt(self.null_variance / ntoks) # std of average
            z_score = (avg_score - self.null_mean) / std_dev if std_dev > 0 else 0
            pvalue = 1 - special.ndtr(z_score) # CDF(z_score)
        return max(pvalue, eps)


def _gumbel_score(r: float) -> float:
    return -(1 - r).log().item()


class TextSealDetector:
    """Dual-key entropy-aware TextSeal detector."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        wm_config: WatermarkConfig,
        model: PreTrainedModel = None,
        scoring_method: str = "v2",
    ):
        self.tokenizer = tokenizer
        self.wm_config = wm_config
        self.ngram = wm_config.ngram
        self.key_a = wm_config.key_a
        self.key_b = wm_config.key_b
        self.alpha = wm_config.mixing_alpha
        self.model = model
        self.scoring_method = scoring_method

    @torch.no_grad()
    def _compute_entropies(self, token_ids: list[int]) -> list[float]:
        if self.model is None or len(token_ids) <= 1:
            return []
        device = next(self.model.parameters()).device
        tokens = torch.tensor([token_ids], device=device)
        logits = self.model(tokens).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.squeeze(0)[:-1].tolist()

    @torch.no_grad()
    def _compute_entropies_batch(self, token_lists: list[list[int]]) -> list[list[float]]:
        if self.model is None or not token_lists:
            return [[] for _ in token_lists]

        device = next(self.model.parameters()).device
        lengths = [len(tokens) for tokens in token_lists]
        max_len = max(lengths)
        padded = []
        attention_masks = []
        for tokens in token_lists:
            pad_len = max_len - len(tokens)
            padded.append([0] * pad_len + tokens)
            attention_masks.append([0] * pad_len + [1] * len(tokens))

        input_ids = torch.tensor(padded, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        results = []
        for ii, seq_len in enumerate(lengths):
            if seq_len <= 1:
                results.append([])
                continue
            start = max_len - seq_len
            results.append(entropy[ii, start:start + seq_len - 1].tolist())
        return results

    @staticmethod
    def _entropy_weight(entropy: float, entropy_min: float, entropy_max: float) -> float:
        if entropy_max <= entropy_min:
            return 1.0
        ratio = max(0.0, min(1.0, (entropy - entropy_min) / (entropy_max - entropy_min)))
        return 0.1 + 0.9 * ratio

    def _score_text(
        self,
        token_ids: list[int],
        entropies: list[float] | None,
        scoring_method: str | None = None,
    ) -> dict:
        scoring_method = scoring_method or self.scoring_method
        base_var = self.alpha ** 2 + (1 - self.alpha) ** 2

        if len(token_ids) <= self.ngram + 1:
            return {"p_value": 1.0, "n_tokens": 0, "detected": False, "entropy_weighted": False}

        fused_scores = []
        scored_entropies = []
        seen = set()

        for pos in range(self.ngram + 1, len(token_ids)):
            ctx = token_ids[pos - self.ngram:pos]
            tok = token_ids[pos]
            if scoring_method == "v1":
                dedup_key = tuple(ctx)
            elif scoring_method == "v2":
                dedup_key = tuple(ctx) + (tok,)
            else:
                dedup_key = None

            if dedup_key is not None:
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

            ctx_tensor = torch.tensor([ctx])
            config_a = replace(self.wm_config, secret_key=self.key_a)
            config_b = replace(self.wm_config, secret_key=self.key_b)
            r_a = score_listed_tokens(ctx_tensor, config_a, [tok])[0, 0]
            r_b = score_listed_tokens(ctx_tensor, config_b, [tok])[0, 0]
            fused_scores.append(self.alpha * _gumbel_score(r_a) + (1 - self.alpha) * _gumbel_score(r_b))

            if entropies is not None and (pos - 1) < len(entropies):
                scored_entropies.append(entropies[pos - 1])

        n_tokens = len(fused_scores)
        if n_tokens == 0:
            return {"p_value": 1.0, "n_tokens": 0, "detected": False, "entropy_weighted": False}

        scores = np.array(fused_scores)
        score_sum = np.sum(scores)
        p_unweighted = float(special.gammaincc(n_tokens / base_var, score_sum / base_var))

        if scored_entropies and len(scored_entropies) == n_tokens:
            entropy_min = min(scored_entropies)
            entropy_max = max(scored_entropies)
            if entropy_max - entropy_min < 1e-6:
                entropy_min, entropy_max = 0.0, 5.0
            weights = np.array([self._entropy_weight(e, entropy_min, entropy_max) for e in scored_entropies])
            weighted_sum = np.sum(weights * scores)
            mu = np.sum(weights)
            var = np.sum(weights ** 2) * base_var
            if var > 1e-10:
                shape = mu ** 2 / var
                scale = var / mu
                p_weighted = float(special.gammaincc(shape, weighted_sum / scale))
            else:
                p_weighted = 1.0
            p_value = min(p_weighted, p_unweighted)
            return {
                "p_value": p_value,
                "p_value_weighted": p_weighted,
                "p_value_unweighted": p_unweighted,
                "n_tokens": n_tokens,
                "detected": p_value < 0.01,
                "entropy_weighted": True,
            }

        return {
            "p_value": p_unweighted,
            "n_tokens": n_tokens,
            "detected": p_unweighted < 0.01,
            "entropy_weighted": False,
        }

    def detect(self, text: str, scoring_method: str | None = None) -> dict:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        entropies = self._compute_entropies(token_ids) if self.model is not None else None
        return self._score_text(token_ids, entropies, scoring_method)

    def get_scores_by_t(
        self,
        texts: list[str],
        scoring_method: str = "none",
        ntoks_max: int = None,
        return_aux: bool = False,
        entropy_threshold: float = None,
        seen_windows: set = None,
        precomputed_entropies: list = None,
    ) -> list[list[float]] | tuple[list[list[float]], list[list[int]]]:
        score_lists = []
        masks_lists = []
        for ii, text in enumerate(texts):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ntoks_max is not None:
                token_ids = token_ids[:ntoks_max]

            entropies = None
            if entropy_threshold is not None:
                if precomputed_entropies is not None:
                    entropies = precomputed_entropies[ii]
                else:
                    entropies = self._compute_entropies(token_ids)

            scores = []
            mask_scored = []
            local_seen = seen_windows if seen_windows is not None else set()
            config_a = replace(self.wm_config, secret_key=self.key_a)
            config_b = replace(self.wm_config, secret_key=self.key_b)

            for pos in range(self.ngram + 1, len(token_ids)):
                ctx = token_ids[pos - self.ngram:pos]
                tok = token_ids[pos]
                mask_scored.append(0)

                if entropy_threshold is not None and entropies is not None:
                    entropy_idx = pos - 1
                    if entropy_idx < len(entropies) and entropies[entropy_idx] < entropy_threshold:
                        continue

                if scoring_method == "v1":
                    dedup_key = tuple(ctx)
                elif scoring_method == "v2":
                    dedup_key = tuple(ctx) + (tok,)
                else:
                    dedup_key = None

                if dedup_key is not None:
                    if dedup_key in local_seen:
                        continue
                    local_seen.add(dedup_key)

                ctx_tensor = torch.tensor([ctx])
                r_a = score_listed_tokens(ctx_tensor, config_a, [tok])[0, 0]
                r_b = score_listed_tokens(ctx_tensor, config_b, [tok])[0, 0]
                scores.append(self.alpha * _gumbel_score(r_a) + (1 - self.alpha) * _gumbel_score(r_b))
                mask_scored[-1] = 1

            score_lists.append(scores)
            masks_lists.append(mask_scored)

        if return_aux:
            return score_lists, masks_lists
        return score_lists

    def get_pvalue(self, score: float, ntoks: int, eps: float = 1e-200) -> float:
        if ntoks == 0:
            return 1.0
        base_var = self.alpha ** 2 + (1 - self.alpha) ** 2
        return max(float(special.gammaincc(ntoks / base_var, score / base_var)), eps)

    def detect_batch(self, texts: list[str], scoring_method: str | None = None) -> list[dict]:
        if not texts:
            return []
        token_lists = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        if self.model is not None:
            entropies_list = self._compute_entropies_batch(token_lists)
        else:
            entropies_list = [None] * len(texts)
        return [self._score_text(tokens, entropies, scoring_method) for tokens, entropies in zip(token_lists, entropies_list)]


@dataclass
class LocalizedResult:
    global_pvalue: float
    localized_pvalue: float
    final_pvalue: float
    detected: bool
    region_start: int
    region_end: int
    n_tokens: int
    token_labels: list[int]


def _geometric_cover_search(
    scores: np.ndarray,
    min_length: int = 50,
    base_variance: float = 1.0,
) -> tuple[int, int, float]:
    n = len(scores)
    prefix = np.zeros(n + 1)
    for ii in range(n):
        prefix[ii + 1] = prefix[ii] + scores[ii]

    best_start, best_end, best_z = 0, n, float("-inf")
    max_power = int(np.floor(np.log2(n))) if n > 0 else 0
    for power in range(max_power + 1):
        length = 2 ** power
        if length < min_length:
            continue
        stride = max(1, length // 2)
        for start in range(0, n - length + 1, stride):
            end = start + length
            raw_sum = prefix[end] - prefix[start]
            z_score = (raw_sum - length) / np.sqrt(length * base_variance)
            if z_score > best_z:
                best_start, best_end, best_z = start, end, z_score
    return best_start, best_end, best_z


def _count_tests(n: int, min_length: int) -> int:
    total = 0
    for power in range(int(np.floor(np.log2(n))) + 1 if n > 0 else 0):
        length = 2 ** power
        if length < min_length or length > n:
            continue
        stride = max(1, length // 2)
        total += (n - length) // stride + 1
    return max(1, total)


def _boundary_smoother(scores: np.ndarray, window: int = 20, threshold: float = 1.2) -> list[int]:
    n = len(scores)
    if n == 0:
        return []
    labels = [0] * n
    half = window // 2
    for ii in range(n):
        start = max(0, ii - half)
        end = min(n, ii + half + 1)
        if np.mean(scores[start:end]) > threshold:
            labels[ii] = 1
    return labels


def localized_detect(
    text: str,
    tokenizer,
    wm_config: WatermarkConfig,
    model=None,
    min_length: int = 50,
    smoother_window: int = 20,
    smoother_threshold: float = 1.2,
) -> LocalizedResult:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    alpha = wm_config.mixing_alpha
    base_var = alpha ** 2 + (1 - alpha) ** 2

    if len(token_ids) <= wm_config.ngram + 1:
        return LocalizedResult(1.0, 1.0, 1.0, False, 0, 0, 0, [])

    entropies = None
    if model is not None:
        tokens_t = torch.tensor([token_ids], device=next(model.parameters()).device)
        with torch.no_grad():
            logits = model(tokens_t).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropies = (-(probs * log_probs).sum(dim=-1)).squeeze(0)[:-1].tolist()

    fused_scores = []
    scored_entropies = []
    seen = set()
    config_a = replace(wm_config, secret_key=wm_config.key_a)
    config_b = replace(wm_config, secret_key=wm_config.key_b)
    for pos in range(wm_config.ngram + 1, len(token_ids)):
        ctx = token_ids[pos - wm_config.ngram:pos]
        tok = token_ids[pos]
        dedup_key = tuple(ctx) + (tok,)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        ctx_tensor = torch.tensor([ctx])
        r_a = score_listed_tokens(ctx_tensor, config_a, [tok])[0, 0]
        r_b = score_listed_tokens(ctx_tensor, config_b, [tok])[0, 0]
        fused_scores.append(alpha * _gumbel_score(r_a) + (1 - alpha) * _gumbel_score(r_b))
        if entropies is not None and (pos - 1) < len(entropies):
            scored_entropies.append(entropies[pos - 1])

    n_tokens = len(fused_scores)
    if n_tokens == 0:
        return LocalizedResult(1.0, 1.0, 1.0, False, 0, 0, 0, [])

    scores = np.array(fused_scores)
    global_sum = np.sum(scores)
    global_p = float(special.gammaincc(n_tokens / base_var, global_sum / base_var))
    start, end, _ = _geometric_cover_search(scores, min_length, base_var)
    region_sum = np.sum(scores[start:end])
    region_len = end - start
    raw_p = float(special.gammaincc(region_len / base_var, region_sum / base_var))
    localized_p = min(1.0, raw_p * _count_tests(n_tokens, min_length))
    final_p = min(1.0, min(global_p, localized_p) * 2)
    labels = _boundary_smoother(scores, smoother_window, smoother_threshold)
    return LocalizedResult(global_p, localized_p, final_p, final_p < 0.01, start, end, n_tokens, labels)
                
def build_detector(
    tokenizer: PreTrainedTokenizer,
    wm_args: WatermarkConfig,
    model: PreTrainedModel = None
) -> WmDetector | TextSealDetector:
    """
    Build watermark detector based on configuration.
    
    Args:
        tokenizer: The tokenizer for the model
        wm_args: Watermark configuration containing all parameters
        model: Optional model (for compatibility)
        
    Returns:
        Appropriate detector instance based on watermark type
    """
    # For WaterMax, return base detector
    if wm_args.watermark_type == "watermax":
        wm_args = replace(wm_args, watermark_type=wm_args.base_watermark)
        return build_detector(tokenizer, wm_args, model)
        
    # replace sampling method
    if wm_args.watermark_type in ["greenlist", "dipmark", "morphmark"] or wm_args.watermark_type.startswith("synthid"):
        sampling_method = "binary" 
    elif wm_args.watermark_type in ["gumbelmax", "textseal"]:
        sampling_method = "uniform"
    wm_args = replace(wm_args, method=sampling_method)
    
    # build detector
    if wm_args.watermark_type in ["greenlist", "dipmark", "morphmark"]:
        return GreenlistDetector(tokenizer, wm_args, model)
    elif wm_args.watermark_type == "gumbelmax":
        return GumbelmaxDetector(tokenizer, wm_args, model)
    elif wm_args.watermark_type == "textseal":
        return TextSealDetector(tokenizer, wm_args, model, wm_args.scoring_method)
    elif wm_args.watermark_type.startswith("synthid"):
        weighted = "weighted" in wm_args.watermark_type
        return SynthidDetector(tokenizer, wm_args, model, weighted)
    elif wm_args.watermark_type == "none":
        return WmDetector(tokenizer, wm_args, model)
    else:
        raise ValueError(f"Unknown watermark type: {wm_args.watermark_type}")
