# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import replace

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from apps.common.watermark.core import WatermarkArgs, score_all_next_tokens, score_listed_tokens

class WmGenerator():
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        if hasattr(model.config, 'max_sequence_length'):
            self.max_seq_len = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            self.max_seq_len = model.config.max_position_embeddings
        else:
            self.max_seq_len = 4096  # Default fallback
            
        # Handle EOS tokens generically
        self.eos_id = self._get_eos_tokens()
        
        # Handle pad token
        if model.config.pad_token_id is not None:
            self.pad_id = model.config.pad_token_id
        else:
            # Use first EOS token as pad token if no pad token is set
            self.pad_id = self.eos_id[0] if isinstance(self.eos_id, list) else self.eos_id

        # watermark config
        self.wm_args = wm_args
        self.ngram = self.wm_args.ngram
        self.secret_key = self.wm_args.secret_key
    
    def _get_eos_tokens(self) -> int | list[int]:
        """Get EOS tokens from HuggingFace model/tokenizer."""
        # For HuggingFace models, just use the tokenizer's EOS token
        if hasattr(self.model.config, 'eos_token_id') and self.model.config.eos_token_id is not None:
            if "gemma" in self.model.config._name_or_path.lower():
                # for gemma models, we need to also add tokenizer.convert_tokens_to_ids("<end_of_turn>"). First check if its eos_token_id is a list...
                if isinstance(self.model.config.eos_token_id, list):
                    # add <end_of_turn> variants
                    end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    if end_of_turn_id not in self.model.config.eos_token_id:
                        self.model.config.eos_token_id.append(end_of_turn_id)
                else:
                    self.model.config.eos_token_id = [self.model.config.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
            return self.model.config.eos_token_id
        else:
            print("No EOS token found, defaulting to None")
            return None

    def _encode_prompts(self, prompts: list[str]) -> list[list[int]]:
        """Encode prompts using HuggingFace tokenizer."""
        return [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size from HuggingFace model."""
        return self.model.config.vocab_size

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        bsz = len(prompts)
        # Generic tokenizer encoding
        prompt_tokens = self._encode_prompts(prompts)
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        tokens = tokens.to(self.model.device)
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        eos_ids = self.eos_id if isinstance(self.eos_id, list) else [self.eos_id]
        decoded = []

        for i, t in enumerate(tokens.tolist()):
            gen_start = len(prompt_tokens[i])
            # cut from start to max_gen_len
            t = t[gen_start: gen_start + max_gen_len]
            # cut to eos tok if any
            try:
                eos_positions = [t.index(eos) for eos in eos_ids if eos in t]
                if eos_positions:
                    t = t[:min(eos_positions)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # next token (int), in space ordered by probs
            next_token = torch.gather(probs_idx, -1, next_token) # next token (int), in vocab space
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


class GumbelmaxGenerator(WmGenerator):
    """ Generate text using LLaMA and Aaronson's watermarking method. """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and the secret key
        - use the hash to generate V random number r between [0,1]
        - select argmax ( r^(1/p) )
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            
            bsz, vocab_size = ngram_tokens.shape[0], logits.shape[-1]
            # Get uniform random scores for all tokens
            uniform_scores = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size)  # (bsz, vocab_size)
            # Reorder uniform scores according to probability sorting
            batch_indices = torch.arange(bsz, device=probs_idx.device).unsqueeze(1)
            rs_sorted = uniform_scores[batch_indices, probs_idx]  # reorder according to prob sorting
            # Compute r^(1/p) for OpenAI watermarking
            probs_sort = torch.pow(rs_sorted, 1/probs_sort)
            # Select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token


class GreenlistGenerator(WmGenerator):
    """ Generate text using any model and Green-list/Red-list watermarking method. """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)
        self.delta = self.wm_args.delta
        self.after_topp = self.wm_args.after_topp

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        return_probs: bool = False,
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
        - add delta to greenlist words' logits
        """
        if not self.after_topp:
            logits = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            if self.after_topp:
                bsz, vocab_size = logits.shape
                green_scores = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size)
                # Reorder green scores to match probability sorting
                batch_indices = torch.arange(bsz, device=probs_idx.device).unsqueeze(1)
                green_scores_sorted = green_scores[batch_indices, probs_idx]  # (bsz, vocab_size)
                multiplier = torch.exp(self.delta * green_scores_sorted)
                probs_sort = probs_sort * multiplier
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # next token (int), in space ordered by probs
            next_token = torch.gather(probs_idx, -1, next_token) # next token (int), in vocab space
            # return the full probability distribution in vocab space
            if return_probs:
                sampled_probs = torch.zeros_like(probs, dtype=probs_sort.dtype).scatter_(1, probs_idx, probs_sort)
                next_token = next_token.reshape(-1)
                return next_token, sampled_probs
        else:
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")
        next_token = next_token.reshape(-1)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        scores = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size)
        bias = scores * self.delta
        logits += bias
        return logits


# Backward compatibility aliases
MarylandGenerator = GreenlistGenerator
OpenaiGenerator = GumbelmaxGenerator


class MorphMarkGenerator(WmGenerator):
    """ 
    Generate text using MorphMark: Flexible Adaptive Watermarking.
    Paper: https://arxiv.org/abs/2505.11541v2
    
    Implements the adaptive strength mechanism:
    r = exp(k * P_G) - 1  (if P_G > p_0)
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)
        # Default values: k_morphmark=1.30, p_0=0.15, gamma=0.5
        self.k_morphmark = getattr(wm_args, 'k_morphmark', 1.30) 
        self.p_0 = getattr(wm_args, 'p_0', 0.15)
        self.gamma = getattr(wm_args, 'gamma', 0.5)

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_probs: bool = False,
    ) -> torch.LongTensor:
        """
        MorphMark sampling with adaptive watermark strength based on green mass.
        """
        eps = 1e-6
        bsz, vocab_size = logits.shape

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")
            return next_token.reshape(-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)

        # Identify green list and calculate P_G
        green_mask = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size).bool()

        P_G = (probs * green_mask.float()).sum(dim=-1, keepdim=True) # b 1
        # P_G = torch.clamp(P_G, min=1e-6, max=1.0 - 1e-6)

        # Calculate adaptive strength r
        r = self.k_morphmark * P_G
        # r = torch.exp(self.k_morphmark * P_G) - 1.0
        r = torch.where(P_G > self.p_0, r, torch.tensor(eps, device=probs.device))
        # print(self.k_morphmark, r)
        r = torch.clamp(r, max=1-eps)

        # Update probability vector
        green_factor = 1.0 + (r * (1.0 - P_G) / P_G)
        # print(P_G, r, green_factor)
        red_factor = 1.0 - r

        green_factor = green_factor.expand_as(probs)
        red_factor = red_factor.expand_as(probs)
        
        modifiers = torch.where(green_mask, green_factor, red_factor)
        wm_probs = probs * modifiers
        wm_probs = wm_probs / wm_probs.sum(dim=-1, keepdim=True)

        # Sample with top-p
        if top_p < 1.0:
            probs_sort, probs_idx = torch.sort(wm_probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
            
            if return_probs:
                # Return the full probability distribution in vocab space
                sampled_probs = torch.zeros_like(wm_probs, dtype=probs_sort.dtype).scatter_(1, probs_idx, probs_sort)
                next_token = next_token.reshape(-1)
                return next_token, sampled_probs
        else:
            next_token = torch.multinomial(wm_probs, num_samples=1)
            if return_probs:
                next_token = next_token.reshape(-1)
                return next_token, wm_probs

        return next_token.reshape(-1)


class OptGenerator(WmGenerator):
    """
    https://arxiv.org/abs/2312.17295
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)
        self.beta = self.wm_args.delta
        self.after_topp = self.wm_args.after_topp

    def _compute_opt_decision(self, probs, green_mask):
        pass

    def sample_next(
        self,
        logits: torch.FloatTensor,
        ngram_tokens: torch.LongTensor,
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_probs: bool = False,
    ) -> torch.LongTensor:
        
        if temperature > 0:
            pass
        else:
            # Argmax sampling (greedy) usually skips standard watermarking noise logic
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")

        next_token = next_token.reshape(-1)
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """
        Process logits to mask out red words if B <= beta.
        Used when self.after_topp is False.
        """
        pass


class DipmarkGenerator(WmGenerator):
    """ https://arxiv.org/abs/2310.07710 """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)
        self.alpha = self.wm_args.alpha

    def dip_reweight_probs(self, probs: torch.FloatTensor, perm) -> torch.FloatTensor:
        """
        Apply DiP-style reweighting to probs given a permutation.
        Some credits to https://github.com/yihwu/DiPmark/blob/main/dipmark/dipmark.py
        """
        # Shuffle probs according to code
        s_probs = torch.gather(probs, -1, perm)
        # Compute cumulative probability
        s_cumsum = torch.cumsum(s_probs, dim=-1)
        # Find α and (1−α) boundaries
        boundary_1 = torch.argmax((s_cumsum > self.alpha).int(), dim=-1, keepdim=True) # bsz x 1
        boundary_2 = torch.argmax((s_cumsum > (1 - self.alpha)).int(), dim=-1, keepdim=True) # bsz x 1
        # Compute partial membership of boundary to the different sides
        def boundary_fraction(boundary, cutoff):
            p_boundary = s_probs.gather(-1, boundary)
            portion = (s_cumsum.gather(-1, boundary) - cutoff) / (p_boundary + 1e-12)
            return torch.clamp(portion, 0, 1)
        portion_in_right_1 = boundary_fraction(boundary_1, self.alpha)
        portion_in_right_2 = boundary_fraction(boundary_2, 1 - self.alpha)
        # Construct smooth indicator masks
        s_mask_1 = (s_cumsum > self.alpha).type_as(probs)
        s_mask_1.scatter_(-1, boundary_1, portion_in_right_1)
        s_mask_2 = (s_cumsum > (1 - self.alpha)).type_as(probs)
        s_mask_2.scatter_(-1, boundary_2, portion_in_right_2)
        # Combine both sides symmetrically, starts with 0 until α, then 0.5 until 1-α, then 1
        s_mask = 0.5 * (s_mask_1 + s_mask_2) # bsz x vocab_size
        # print("Dipmark reweighting mask stats: "(s_mask==0).sum().item(),(s_mask==0.5).sum().item(),(s_mask==1).sum().item())
        s_shift_probs = s_probs * s_mask
        s_shift_probs = s_shift_probs / (s_shift_probs.sum(dim=-1, keepdim=True) + 1e-12)
        shift_probs = torch.gather(s_shift_probs, -1, torch.argsort(perm, dim=-1))
        return shift_probs 

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        return_probs: bool = False,
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and the secret key
        - use the hash to generate green/red tokens
        - reweight probabilities depending on cumulative distribution
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            
            bsz, vocab_size = ngram_tokens.shape[0], logits.shape[-1]
            # Get uniform random scores for all tokens
            uniform_scores = score_all_next_tokens(ngram_tokens, self.wm_args, vocab_size)  # (bsz, vocab_size)
            # Reorder uniform scores according to probability sorting to align the scores to the good tokens
            rs_sorted = uniform_scores[
                torch.arange(bsz, device=probs_idx.device).unsqueeze(1), probs_idx
            ]  # reorder according to prob sorting
            # Permutation for reweighting, descending order: high rs first (r < γ means green)
            perm = torch.argsort(rs_sorted, dim=-1, descending=True) # (bsz, vocab_size)
            # Reweight logits according to DiP
            probs_sort = self.dip_reweight_probs(probs_sort, perm)
            # Select 
            next_token = torch.multinomial(probs_sort, num_samples=1) # next token (int), in space ordered by probs
            next_token = torch.gather(probs_idx, -1, next_token) # next token (int), in vocab space
            if return_probs:
                sampled_probs = torch.zeros_like(probs, dtype=probs_sort.dtype).scatter_(1, probs_idx, probs_sort)
                next_token = next_token.reshape(-1)
                return next_token, sampled_probs
        else:
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")
        next_token = next_token.reshape(-1)
        return next_token


class SynthidGenerator(WmGenerator):
    """ See https://github/google-deepmind/synthid-text and associated paper. """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)
        self.depth = self.wm_args.depth

    def g_values(
        self,
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        listed_tokens: torch.LongTensor, # (bsz, num_tokens): tokens to score   
        depth: int,
    ) -> list[torch.FloatTensor]:
        """
        Generate g-values for all tokens based on SynthID tournament logic.
        Args:
            ngram_tokens: Tokens to consider when seeding (bsz, ngram).
            wm_args: Watermark arguments.
            listed_tokens: Tokens to score (bsz, num_tokens).
            depth: Depth of the tournament.
        Returns:
            g-values: A tensor of shape (bsz, num_tokens, depth), padded with zeros.
        """
        bsz, num_tokens = listed_tokens.shape
        g_values = torch.zeros(bsz, num_tokens, depth, device=listed_tokens.device)
        valid_mask = listed_tokens >= 0
        if not valid_mask.any():
            return g_values
        for dd in range(depth):
            wm_args_depth = replace(self.wm_args, secret_key=self.wm_args.secret_key)
            for ii in range(bsz):
                valid_tokens = listed_tokens[ii, valid_mask[ii]]
                scores = score_listed_tokens(
                    ngram_tokens[ii].unsqueeze(0),
                    wm_args_depth,
                    valid_tokens + dd,
                ).squeeze(0)
                g_values[ii, valid_mask[ii], dd] = scores
        return g_values

    def update_probs(
        self, 
        probs: torch.FloatTensor, 
        g_values: torch.FloatTensor,
        num_leaves: int = 2,
    ) -> torch.FloatTensor:
        """
        Updates probabilities based on g-values using the SynthID tournament logic.
        Operates in probability space and iterates over the depth.
        Args:
            probs: Probabilities of tokens (bsz, num_tokens).
            g_values: G-values (0 or 1) for the same tokens (bsz, num_tokens, depth).
            num_leaves: Number of leaves in the tournament (default: 2).
        Returns:
            Updated probabilities (bsz, num_tokens).
        """
        depth = g_values.shape[-1]
        # Iteratively apply the update rule for each tournament layer
        for ii in range(depth):
            # Get g-values for the current layer
            g_values_at_depth = g_values[:, :, ii] # b n d -> b n
            # Calculate the total probability mass of "green" tokens (g_value=1)
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True) # b n -> b 1
            # Update probs according to previous proba distribution
            coeff_not_in_g = (1 - g_mass_at_depth) ** (num_leaves - 1)
            coeff_in_g = (1 - (1 - g_mass_at_depth) ** (num_leaves)) / g_mass_at_depth
            coeffs = torch.where(
                torch.logical_and(g_values_at_depth == 1, probs > 0),
                coeff_in_g,
                coeff_not_in_g,
            )
            probs = probs * coeffs
        return probs.clamp(min=0.0)

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        return_probs: bool = False,
    ) -> torch.LongTensor | tuple[torch.LongTensor, torch.FloatTensor]:
        """
        From ngram tokens, select the next token based on the following:
        - score all tokens after topp to get their g-values
        - reweight probabilities by simulating tournament
        - sample from reweighted probabilities
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # get number of tokens remaining after top-p, and max in the batch
            valid_counts = (probs_sort > 0).sum(dim=-1)
            max_tokens = int(valid_counts.max().item()) if valid_counts.numel() > 0 else 0
            # trim to reduce computation, and 
            trimmed_probs_sort = probs_sort[:, :max_tokens]
            trimmed_idx = probs_idx[:, :max_tokens]
            # create a batch of tokens that should be scored, padded with -1
            pad_mask = (
                torch.arange(max_tokens, device=trimmed_probs_sort.device)
                .unsqueeze(0)
                < valid_counts.unsqueeze(1)
            )
            trimmed_probs_sort = trimmed_probs_sort * pad_mask.to(trimmed_probs_sort.dtype)
            listed_tokens = torch.full_like(trimmed_idx, -1)
            listed_tokens[pad_mask] = trimmed_idx[pad_mask]
            # compute g-values and update probs
            g_vals_sort = self.g_values(ngram_tokens, listed_tokens, self.depth)
            g_vals_sort = g_vals_sort * pad_mask.unsqueeze(-1)
            updated_probs_sort = self.update_probs(trimmed_probs_sort, g_vals_sort)
            updated_probs_sort = updated_probs_sort / (updated_probs_sort.sum(dim=-1, keepdim=True) + 1e-9)
            # Select 
            next_token = torch.multinomial(updated_probs_sort, num_samples=1) # next token (int), in space ordered by probs
            next_token = torch.gather(probs_idx, -1, next_token) # next token (int), in vocab space
            if return_probs:
                sampled_probs = torch.zeros_like(probs, dtype=updated_probs_sort.dtype)
                sampled_probs.scatter_(1, trimmed_idx, updated_probs_sort)
                next_token = next_token.reshape(-1)
                return next_token, sampled_probs
        else:
            next_token = torch.argmax(logits, dim=-1)
            if return_probs:
                raise NotImplementedError("return_probs=True not implemented for argmax sampling")
        next_token = next_token.reshape(-1)
        return next_token

class WaterMaxGenerator(WmGenerator):
    """
    WaterMax watermarking method.
    
    Algorithm:
    1. Generate m drafts of L tokens each from the LLM distribution (after temp/top-p, no watermark)
    2. For each draft, compute the watermark score when added to the context
    3. Select the draft with the highest watermark score (or lowest p-value)
    4. Add the selected draft to context and repeat until EOS
    
    The underlying watermark type (Maryland, OpenAI, SynthID, etc.) is specified
    by wm_args.base_watermark and is used for scoring during generation and detection.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
    ):
        super().__init__(model, tokenizer, wm_args)
        self.chunk_size = wm_args.chunk_size  # L
        self.num_drafts = wm_args.num_drafts  # m
        self.scoring_method = wm_args.scoring_method  # v1, v2, or none
        self.base_watermark = wm_args.base_watermark  # Underlying watermark type
        
        # Import here to avoid circular dependency
        from apps.posthoc.detector import build_detector
        
        # Create a detector for the base watermark type
        # This will be used for scoring drafts
        base_wm_args = replace(wm_args, watermark_type=self.base_watermark)
        self.detector = build_detector(tokenizer, base_wm_args, model)
        
    def _compute_draft_score(
        self,
        context_tokens: list[int],
        draft_tokens: list[int],
    ) -> float:
        """
        Compute the watermark p-value for context + draft using the base detector.
        Uses the scoring_method to handle unique watermark windows.
        
        Args:
            context_tokens: List of token IDs in the context
            draft_tokens: List of token IDs in the draft
            
        Returns:
            Watermark p-value (lower = more watermarked)
        """
        combined = context_tokens + draft_tokens
        score = 0.0
        ntoks_scored = 0
        seen_ntuples = set()
        
        # Start scoring after the ngram context
        start_pos = self.ngram + 1
        
        for cur_pos in range(start_pos, len(combined)):
            ngram_tokens = combined[cur_pos - self.ngram:cur_pos]
            
            # Apply scoring method to handle uniqueness
            if self.scoring_method == 'v1':
                # Only score if watermark window is unique
                tup_for_unique = tuple(ngram_tokens)
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            elif self.scoring_method == 'v2':
                # Only score if watermark window + token is unique
                tup_for_unique = tuple(ngram_tokens + [combined[cur_pos]])
                if tup_for_unique in seen_ntuples:
                    continue
                seen_ntuples.add(tup_for_unique)
            # For 'none', score all tokens
            
            # Compute score for this token using the base detector
            token_score = self.detector.score_tok(ngram_tokens, combined[cur_pos])
            score += token_score
            ntoks_scored += 1
        
        # Convert score to p-value (lower is better)
        if ntoks_scored > 0:
            pvalue = self.detector.get_pvalue(score, ntoks_scored, eps=1e-200)
        else:
            pvalue = 1.0  # No tokens scored, worst p-value
            
        return pvalue

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Generate text using WaterMax method.
        For each chunk, generate m drafts and select the one with highest watermark score.
        """
        bsz = len(prompts)
        
        # Encode prompts
        prompt_tokens = self._encode_prompts(prompts)
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)
        
        # Initialize token buffer
        tokens = torch.full((bsz, total_len), self.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        tokens = tokens.to(self.model.device)
        input_text_mask = tokens != self.pad_id
        
        # Track which sequences have finished (seen EOS)
        finished = torch.zeros(bsz, dtype=torch.bool, device=self.model.device)
        
        # EOS token handling
        eos_ids = self.eos_id if isinstance(self.eos_id, list) else [self.eos_id]
        
        cur_pos = min_prompt_size
        prev_pos = 0
        
        while cur_pos < total_len:
            # Determine chunk size (may be smaller at the end)
            chunk_len = min(self.chunk_size, total_len - cur_pos)
            
            # For each sequence in the batch
            for batch_idx in range(bsz):
                if finished[batch_idx]:
                    continue
                
                # Get context tokens for this sequence
                context_tokens = tokens[batch_idx, :cur_pos].tolist()
                
                # Generate m drafts IN PARALLEL
                # Create batch of m copies of the context
                draft_contexts = torch.tensor([context_tokens] * self.num_drafts, device=self.model.device)  # (m, cur_pos)
                draft_tokens_list = [[] for _ in range(self.num_drafts)]
                draft_finished_flags = torch.zeros(self.num_drafts, dtype=torch.bool, device=self.model.device)
                
                # Generate chunk_len tokens for all drafts in parallel
                prev_pos = 0
                for tok_idx in range(chunk_len):
                    # Get logits for all drafts
                    outputs = self.model.forward(
                        draft_contexts[:, prev_pos:],
                        use_cache=True,
                        past_key_values=outputs.past_key_values if prev_pos > 0 else None
                    )
                    logits = outputs.logits[:, -1, :]  # (m, vocab_size)
                    
                    # Sample WITHOUT watermark (vanilla temp/top-p) for all drafts
                    ngram_tokens = draft_contexts[:, -self.ngram:]  # (m, ngram)
                    next_toks = super().sample_next(logits, ngram_tokens, temperature, top_p)  # (m,)
                    
                    # Update each draft's tokens
                    for draft_idx in range(self.num_drafts):
                        if not draft_finished_flags[draft_idx]:
                            next_tok_id = next_toks[draft_idx].item()
                            draft_tokens_list[draft_idx].append(next_tok_id)
                            
                            # Check for EOS
                            if next_tok_id in eos_ids:
                                draft_finished_flags[draft_idx] = True
                    
                    # If all drafts are finished, break early
                    if draft_finished_flags.all():
                        break
                    
                    # Update contexts for next iteration
                    # Only update contexts for non-finished drafts to continue generating
                    draft_contexts = torch.cat([
                        draft_contexts,
                        next_toks.unsqueeze(1)
                    ], dim=1)
                    prev_pos = draft_contexts.shape[1] - 1
                
                # Score all drafts and select the best one (lowest p-value)
                best_draft = None
                best_pvalue = float('inf')
                best_draft_finished = False
                
                for draft_idx in range(self.num_drafts):
                    draft_tokens = draft_tokens_list[draft_idx]
                    draft_pvalue = self._compute_draft_score(context_tokens, draft_tokens)
                    
                    if draft_pvalue < best_pvalue:
                        best_pvalue = draft_pvalue
                        best_draft = draft_tokens
                        best_draft_finished = draft_finished_flags[draft_idx].item()
                
                # Add the best draft to the tokens
                for tok_idx, tok_id in enumerate(best_draft):
                    if cur_pos + tok_idx < total_len:
                        tokens[batch_idx, cur_pos + tok_idx] = tok_id
                
                # Check if this sequence finished
                if best_draft_finished:
                    finished[batch_idx] = True
            
            # Move to next chunk
            cur_pos += chunk_len
            
            # If all sequences are finished, break early
            if finished.all():
                break
        
        # Decode outputs
        eos_ids = self.eos_id if isinstance(self.eos_id, list) else [self.eos_id]
        decoded = []
        
        for i, t in enumerate(tokens.tolist()):
            gen_start = len(prompt_tokens[i])
            # cut from start to max_gen_len
            t = t[gen_start: gen_start + max_gen_len]
            # cut to eos tok if any
            try:
                eos_positions = [t.index(eos) for eos in eos_ids if eos in t]
                if eos_positions:
                    t = t[:min(eos_positions)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded


class BeamSearchGenerator(WmGenerator):
    """
    General beam search decoder for any watermarking method (except OpenAI).
    
    Algorithm:
    1. BIAS: Use any watermarked generation to guide beam search (biased model)
    2. BEAM SEARCH: Maintain B candidate sequences, expanding with biased model
       - Deterministic: Select top-V tokens by probability (default: V=B)
       - Stochastic: Sample V candidates from the distribution
    3. FILTER: Score all BxV sequences using:
       - Original (unbiased) model probabilities (default)
       - Biased (watermarked) probabilities (optional)
    4. SELECT: Return top-B sequences with best quality score (lowest perplexity)
    
    Important: For methods like Maryland after top-p that modify probabilities,
    we track watermarked probabilities for proper scoring when use_biased_for_scoring=True.
    
    Parameters:
        base_generator: The watermarked generator to use for biased generation
        beam_width (B): Number of sequences to maintain
        candidates_per_beam (V): Number of candidates to expand per beam (default: B)
        stochastic: If True, sample V candidates; if False, take top-V deterministic
        use_biased_for_scoring: If True, score with biased model; if False, use original
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        wm_args: WatermarkArgs,
        beam_width: int = 5,
        candidates_per_beam: int = None,
        stochastic: bool = False,
        use_biased_for_scoring: bool = False,
    ):
        super().__init__(model, tokenizer, wm_args)
        
        # Check if watermark type is compatible with beam search
        if wm_args.watermark_type.lower() in ["gumbelmax", "watermax"]:
            raise ValueError(f"{wm_args.watermark_type} watermarking is not compatible with beam search")
        
        self.beam_width = beam_width
        self.candidates_per_beam = candidates_per_beam if candidates_per_beam is not None else beam_width
        self.stochastic = stochastic
        self.use_biased_for_scoring = use_biased_for_scoring
        
        # Create the base watermarked generator
        self.base_generator = build_generator(model, tokenizer, wm_args)
    
    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Generate text using beam search with watermarking.
        
        Special case: beam_width=1 falls back to standard watermarked generation.
        """
        bsz = len(prompts)
        prompt_tokens = self._encode_prompts(prompts)
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)
        
        decoded = []
        eos_ids = self.eos_id if isinstance(self.eos_id, list) else [self.eos_id]
        
        # Process each prompt separately (beam search is per-prompt)
        for prompt_idx in range(bsz):
            prompt_tok = prompt_tokens[prompt_idx]
            prompt_len = len(prompt_tok)
            
            # Initialize beam: B sequences, each starting with the prompt
            B = self.beam_width
            V = self.candidates_per_beam
            beam_sequences = torch.tensor([prompt_tok] * B).long().to(self.model.device)  # (B, prompt_len)
            beam_scores = torch.zeros(B).to(self.model.device)  # Cumulative log probs
            beam_active = torch.ones(B, dtype=torch.bool).to(self.model.device)
            
            # Generate tokens autoregressively
            for pos in range(prompt_len, min(total_len, prompt_len + max_gen_len)):
                # Get logits for all active beams in parallel
                outputs = self.model(beam_sequences)
                logits = outputs.logits[:, -1, :]  # (B, vocab_size)
                
                # Get n-gram context for watermarking
                ngram_tokens = beam_sequences[:, -self.ngram:]  # (B, ngram)
                
                # Get both original and watermarked probabilities
                if temperature > 0:
                    original_probs = torch.softmax(logits / temperature, dim=-1)
                else:
                    raise ValueError("BeamSearchGenerator requires temperature > 0")
                _, watermarked_probs = self.base_generator.sample_next(
                    logits, ngram_tokens, temperature, top_p, return_probs=True
                )  # (B, vocab_size)
                
                # Decide which probabilities to use for scoring
                scoring_probs = watermarked_probs if self.use_biased_for_scoring else original_probs
                
                # For each beam, get V candidate tokens
                if self.stochastic:
                    # Stochastic: sample V candidates from watermarked distribution
                    candidate_indices = torch.multinomial(
                        watermarked_probs, num_samples=V, replacement=True
                    )  # (B, V)
                else:
                    # Deterministic: take top-V tokens by watermarked probability
                    candidate_probs, candidate_indices = torch.topk(
                        watermarked_probs, k=V, dim=-1
                    )  # (B, V)
                
                # Get scoring probabilities for these candidates
                batch_indices = torch.arange(B, device=self.model.device).unsqueeze(1).expand(-1, V)
                candidate_scoring_probs = scoring_probs[batch_indices, candidate_indices]  # (B, V)
                
                # Compute scores: current_score + log(scoring_prob)
                candidate_log_probs = torch.log(candidate_scoring_probs + 1e-10)
                candidate_scores = beam_scores.unsqueeze(1) + candidate_log_probs  # (B, V)
                
                # Flatten to get all BxV candidates
                candidate_scores_flat = candidate_scores.view(-1)  # (BxV,)
                candidate_tokens_flat = candidate_indices.view(-1)  # (BxV,)
                candidate_beam_indices = (
                    torch.arange(B, device=self.model.device)
                    .unsqueeze(1)
                    .expand(-1, V)
                    .reshape(-1)
                )  # (BxV,)
                
                # Select top-B candidates by score
                top_beam_scores, top_beam_flat_indices = torch.topk(
                    candidate_scores_flat, k=B
                )
                
                # Map back to beam and token
                selected_beam_indices = candidate_beam_indices[top_beam_flat_indices]  # (B,)
                selected_tokens = candidate_tokens_flat[top_beam_flat_indices]  # (B,)
                
                # Update beam sequences
                new_beam_sequences = beam_sequences[selected_beam_indices]  # (B, seq_len)
                new_beam_sequences = torch.cat(
                    [new_beam_sequences, selected_tokens.unsqueeze(1)], dim=1
                )  # (B, seq_len+1)
                
                beam_sequences = new_beam_sequences
                beam_scores = top_beam_scores
                
                # Check for EOS and deactivate those beams
                for eos_id in eos_ids:
                    beam_active &= (selected_tokens != eos_id)
                
                # If all beams finished, stop
                if not beam_active.any():
                    break
            
            # Select best sequence from final beam
            best_idx = torch.argmax(beam_scores).item()
            best_sequence = beam_sequences[best_idx].tolist()
            
            # Trim to prompt + max_gen_len and remove EOS
            decoded.append(best_sequence)
        
        for i, t in enumerate(decoded):
            gen_start = len(prompt_tokens[i])
            # cut from start to max_gen_len
            t = t[gen_start: gen_start + max_gen_len]
            # cut to eos tok if any
            try:
                eos_positions = [t.index(eos) for eos in eos_ids if eos in t]
                if eos_positions:
                    t = t[:min(eos_positions)]
            except ValueError:
                pass
            decoded[i] = self.tokenizer.decode(t)
        
        return decoded


def build_generator(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    wm_args: WatermarkArgs,
    beam_width: int = None,
    candidates_per_beam: int = None,
    stochastic_beam: bool = False,
    use_biased_for_scoring: bool = False,
) -> WmGenerator:
    """
    Factory method to build the appropriate watermarked generator.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        wm_args: Watermark arguments
        beam_width: If specified, use beam search with this width (None = standard sampling)
        candidates_per_beam: Number of candidates per beam (default: beam_width)
        stochastic_beam: If True, sample candidates; if False, use top-k
        use_biased_for_scoring: If True, score with watermarked probs; if False, use original
        
    Returns:
        WmGenerator instance (either standard or beam search)
    """
    # replace sampling method based on watermark type
    # For WaterMax, set sampling method based on base_watermark
    if wm_args.watermark_type == "watermax":
        if wm_args.base_watermark in ["greenlist", "synthid", "dipmark"]:
            sampling_method = "binary"
        else:
            sampling_method = "uniform"
    # For other watermark types, set sampling method as usual
    elif wm_args.watermark_type in ["greenlist", "morphmark", "opt"] or wm_args.watermark_type.startswith("synthid"):
        sampling_method = "binary" 
    else:
        sampling_method = "uniform"
    wm_args = replace(wm_args, method=sampling_method)
    
    # If beam search is requested, wrap in BeamSearchGenerator
    if beam_width is not None and beam_width > 1:
        return BeamSearchGenerator(
            model=model,
            tokenizer=tokenizer,
            wm_args=wm_args,
            beam_width=beam_width,
            candidates_per_beam=candidates_per_beam,
            stochastic=stochastic_beam,
            use_biased_for_scoring=use_biased_for_scoring,
        )
    
    # build base generator (standard sampling)
    if wm_args.watermark_type == "greenlist":
        base_generator = GreenlistGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type == "gumbelmax":
        base_generator = GumbelmaxGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type == "dipmark":
        base_generator = DipmarkGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type == "morphmark":
        base_generator = MorphMarkGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type == "opt":
        base_generator = OptGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type.startswith("synthid"):
        base_generator = SynthidGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type == "watermax":
        base_generator = WaterMaxGenerator(model, tokenizer, wm_args)
    elif wm_args.watermark_type == "none":
        base_generator = WmGenerator(model, tokenizer, wm_args)
    else:
        raise ValueError(f"Unknown watermark type: {wm_args.watermark_type}")
    
    return base_generator
