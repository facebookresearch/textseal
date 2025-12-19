# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from scipy import special
from scipy import stats
from dataclasses import replace

import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizer, PreTrainedModel

from apps.common.watermark.core import WatermarkArgs, score_all_next_tokens, score_listed_tokens

class WmDetector():
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            wm_args: WatermarkArgs,
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
            # TikTokenTokenizer from apps.wmtraining.lingua (doesn't have add_special_tokens method)
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
            wm_args: WatermarkArgs,
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
            wm_args: WatermarkArgs,
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
            wm_args: WatermarkArgs,
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
                
def build_detector(
    tokenizer: PreTrainedTokenizer,
    wm_args: WatermarkArgs,
    model: PreTrainedModel = None
) -> WmDetector:
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
    elif wm_args.watermark_type in ["gumbelmax", ]:
        sampling_method = "uniform"
    wm_args = replace(wm_args, method=sampling_method)
    
    # build detector
    if wm_args.watermark_type in ["greenlist", "dipmark", "opt", "morphmark"]:
        return GreenlistDetector(tokenizer, wm_args, model)
    elif wm_args.watermark_type == "gumbelmax":
        return GumbelmaxDetector(tokenizer, wm_args, model)
    elif wm_args.watermark_type.startswith("synthid"):
        weighted = "weighted" in wm_args.watermark_type
        return SynthidDetector(tokenizer, wm_args, model, weighted)
    elif wm_args.watermark_type == "none":
        return WmDetector(tokenizer, wm_args, model)
    else:
        raise ValueError(f"Unknown watermark type: {wm_args.watermark_type}")


# Backward compatibility aliases
MarylandDetector = GreenlistDetector
OpenaiDetector = GumbelmaxDetector
