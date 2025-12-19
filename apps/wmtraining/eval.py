# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluation script for LLM checkpoints using lm-eval harness and custom validation.

salloc --partition=learn --nodes=1 --gpus-per-node=4 --cpus-per-gpu=16 --mem-per-cpu=8G --qos=high --time=6:00:00 --job-name=dev_llm_eval
cd /path/to/textseal
conda activate lingua

Run with config file:
    python -m apps.wmtraining.eval --config configs/eval_config.yaml --ckpt_dir /path/to/checkpoint

Run with CLI arguments:
    python -m apps.wmtraining.eval --ckpt_dir /path/to/checkpoint --harness.tasks '["hellaswag", "arc_easy"]' --harness.num_fewshot 5
    
    srun -n 4 python -m apps.wmtraining.eval \
        --ckpt_dir /path/to/checkpoint/consolidated \
        --harness.tasks '["hellaswag", "winogrande", "arc_easy", "arc_challenge"]' \
        --harness.num_fewshot 5 \
        --harness.limit 100 \
        --generator.temperature 0.0 \
        --dump_dir outputs/eval_results

    srun -n 4 python -m apps.wmtraining.eval \
        --ckpt_dir /path/to/data \
        --validation.sources '["arxiv_split/physics", "arxiv_split/mathematics"]' \
        --validation.root_dir /path/to/data \
        --validation.max_steps 500 \
        --dump_dir outputs/validation_results

SLURM example:
    srun --partition=learn --nodes=1 --gpus-per-node=1 --qos=high --time=6:00:00 --job-name=eval_harness --cpus-per-gpu=24 --mem-per-cpu=8G \
    python -m apps.wmtraining.eval \
        --ckpt_dir /path/to/data \
        --harness.tasks '["hellaswag", "winogrande"]' \
        --harness.num_fewshot 5

"""

# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Any, List, Optional, Tuple, Union
from lm_eval import simple_evaluate
from omegaconf import OmegaConf
import torch
from apps.common.llm.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.common.llm.transformer import LMTransformer, LMTransformerArgs
from apps.common.utils.config import dump_config, dataclass_from_dict, cfg_from_cli
from apps.wmtraining.lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from apps.wmtraining.lingua.data import init_choice_state, setup_sources
from apps.wmtraining.lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class LMHarnessArgs:
    tasks: Optional[List[Any]] = None
    num_fewshot: Optional[int] = None
    device: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: Optional[Union[int, float]] = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: Optional[str] = None
    apply_chat_template: Union[bool, str] = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: Optional[str] = None
    # verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234

@dataclass
class ValidationArgs:
    max_steps: Optional[int] = None # If None the whole validation file is used -> /!\ This number of steps is gpu dependent (100 max steps on 8 gpus = 800 steps on 1 gpu)
    use_val_from_train_src: bool = True # Use the validation set from training sources
    root_dir: str = ""
    sources: List[str] = field(default_factory=list) # Other sources to eval on

@dataclass
class CustomBenchmarkArgs:
    benchmark_dir: Optional[str] = None  # Path to directory containing benchmark folders

@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    harness: Optional[LMHarnessArgs] = field(default_factory=LMHarnessArgs)
    validation: Optional[ValidationArgs] = field(default_factory=ValidationArgs)
    custom_benchmark: Optional[CustomBenchmarkArgs] = field(default_factory=CustomBenchmarkArgs)

    wandb: Optional[Any] = None

    global_step: Optional[int] = None  # for in-training evaluation


def all_dicts_same(dict_list):
    if not dict_list:  # Check if the list is empty
        return True

    # Compare each dictionary to the first one
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)


class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()


# Light wrapper around generator for lm-eval harness
class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: List[Instance]) -> List[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        generations, _, _ = self.generator.generate(prompts)
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.replace(e, "")
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(self.generator.tokenizer.encode(p, add_bos=False, add_eos=False))
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))

        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        # We temporarily lower max gen len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = []
        for ll in lls:
            results.append((ll.sum().item(),))
        self.generator.max_gen_len = max_gen_len

        return results
    

def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    srcs = {}
    for src in val_args.sources:
        path = os.path.join(val_args.root_dir, src)
        srcs[path] = 1.0
    for src in train_cfg.data.sources:
        path = os.path.join(train_cfg.data.root_dir, src)
        srcs[path] = 1.0

    multi_state = init_choice_state("", srcs, 0, get_global_rank(), get_world_size(), "*.val.jsonl")
    path_to_iter = setup_sources(multi_state)

    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len
    generator.max_gen_len = 1

    all_val_metrics = {}
    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        logger.info(f"Running validation on {src}...")
        for step, (content, state) in enumerate(jsonl_iterator):
            if state['current_iter'] > 0 or (val_args.max_steps is not None and step >= val_args.max_steps):
                break
            content_key = "text" if ("text" in content) else "content"
            texts.append(content[content_key])
        
        _, loglikelihood, _ = generator.generate(texts)

        metrics = defaultdict(list)
        for i, ll in enumerate(loglikelihood):
            tmp = ll.sum().item()
            metrics['nll'].append(tmp)
            metrics['nll_per_token'].append(tmp / len(ll))
            metrics['nll_per_char'].append(tmp / len(texts[i]))

            metrics['avg_seqlen'].append(len(ll))
        
        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])
        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1")
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics


def eval_on_custom_benchmarks(generator, benchmark_dir: str):
    """
    Custom evaluation on benchmarks in the specified directory.
    
    For each benchmark folder in benchmark_dir:
    - Reads the file ending with chunk.0.jsonl
    - For each question, creates prompts with all possible answers
    - Computes the loss for each answer
    - Checks if the answer with the lowest loss matches the ground truth
    
    Args:
        generator: The generator model
        benchmark_dir: Path to the directory containing benchmark folders
        
    Returns:
        Dictionary with accuracy metrics for each benchmark
    """
    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len to compute loss
    generator.max_gen_len = 1
    
    all_benchmark_metrics = {}
    
    # Find all benchmark folders
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        logger.warning(f"Benchmark directory {benchmark_dir} does not exist")
        return all_benchmark_metrics
    
    benchmark_folders = [f for f in benchmark_path.iterdir() if f.is_dir()]
    
    for folder in benchmark_folders:
        benchmark_name = folder.name
        logger.info(f"Running custom evaluation on {benchmark_name}...")
        
        # Find the chunk.0.jsonl file
        chunk_files = list(folder.glob("*.chunk.0.jsonl"))
        if not chunk_files:
            logger.warning(f"No chunk.0.jsonl file found in {folder}")
            continue
        
        chunk_file = chunk_files[0]
        
        correct = 0
        total = 0
        
        # Read and process each question
        with open(chunk_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data['wm_text']  # Assuming the question is under 'wm_text' key
                # Handle both dictionary and list formats for choices
                if isinstance(data['choices'], dict):
                    choices = data['choices']['text']
                    labels = data['choices']['label']
                else:
                    # If choices is a list, assume it's the text directly
                    choices = data['choices']
                    labels = [i for i in range(len(choices))]  # MMLU
                
                answer_key = data['answerKey'] if 'answerKey' in data else data.get('answer', None)
                
                # Map answer key to index
                answer_idx = labels.index(answer_key)
                
                # Create prompts for each possible answer
                prompts = []
                for choice in choices:
                    prompt = f"During a lecture, the professor posed a question: {question}. After discussion, it was revealed that the answer is: {choice}"
                    prompts.append(prompt)
                
                # Compute loss for each prompt
                _, lls, _ = generator.generate(prompts)
                
                # Use mean log likelihood per token for fair comparison across different length answers
                losses = [ll.mean().item() for ll in lls]
                
                # The correct answer should have the lowest loss (highest log likelihood)
                # Note: log likelihoods are negative, so we want the maximum (least negative)
                predicted_idx = losses.index(max(losses))
                
                if predicted_idx == answer_idx:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        logger.info(f"Custom evaluation on {benchmark_name} done. Accuracy: {accuracy:.4f} ({correct}/{total})")
        all_benchmark_metrics[benchmark_name] = metrics
    
    generator.max_gen_len = max_gen_len
    
    return all_benchmark_metrics

def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())
    if (
        Path(cfg.ckpt_dir).exists()
        and (Path(cfg.ckpt_dir) / "params.json").exists()
        and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(cfg.ckpt_dir)
    else:
        consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMTransformerArgs,
    )
    logger.info("Model loaded")
    model.eval()
    generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)

    wrap = EvalHarnessLM(generator)
    results = None
    if cfg.harness and cfg.harness.tasks:
        results = simple_evaluate(wrap, **asdict(cfg.harness))
    val_results =  None
    if cfg.validation:
        val_results = eval_on_val(generator, cfg.validation, train_cfg)
    custom_benchmark_results = None
    if cfg.custom_benchmark and cfg.custom_benchmark.benchmark_dir:
        custom_benchmark_results = eval_on_custom_benchmarks(generator, cfg.custom_benchmark.benchmark_dir)
    if get_global_rank() == 0:
        if results is not None:
            with open(Path(cfg.dump_dir) / "results.json", "w") as f:
                f.write(json.dumps(results))
            logger.info(f"All evaluation results: {results['results']}")
        if val_results is not None:
            with open(Path(cfg.dump_dir) / "validation.json", "w") as f:
                f.write(json.dumps(val_results))
            logger.info(f"All validation results: {val_results}")
        if custom_benchmark_results is not None:
            with open(Path(cfg.dump_dir) / "custom_benchmark.json", "w") as f:
                f.write(json.dumps(custom_benchmark_results))
            logger.info(f"All custom benchmark results: {custom_benchmark_results}")
    if cfg.metric_log_dir and get_global_rank() == 0:
        metric_log_path = Path(cfg.metric_log_dir) / "metrics.eval.jsonl"

        logger.info(f"Writing metric logs to {metric_log_path}")
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.global_step is not None:
            timestamp["global_step"] = cfg.global_step
        if results is not None:
            print(
                json.dumps(timestamp | results["results"]),
                file=open(metric_log_path, mode="a"),
                flush=True,
            )

        val_log_path = Path(cfg.metric_log_dir) / "metrics.validation.jsonl"
        if val_results is not None:
            print(
                json.dumps(timestamp | val_results),
                file=open(val_log_path, mode="a"),
                flush=True,
            )
        
        custom_benchmark_log_path = Path(cfg.metric_log_dir) / "metrics.custom_benchmark.jsonl"
        if custom_benchmark_results is not None:
            print(
                json.dumps(timestamp | custom_benchmark_results),
                file=open(custom_benchmark_log_path, mode="a"),
                flush=True,
            )
    
    del generator


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
