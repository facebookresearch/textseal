# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Run with 8 GPUs
    salloc --partition=learn --nodes=1 --gpus-per-node=8 --qos=high --time=6:00:00 --job-name=wm_train --cpus-per-gpu=24--mem-per-cpu=8G
    cd /path/to/textseal/
    conda activate text_seal

    srun -n 8 python -m apps.wmtraining.train --config configs/train_with_contamination.yaml \
        --distributed.compile false --data.batch_size 16 \

    srun -n 8 python -m apps.wmtraining.train --config configs/debug.yaml \
        --distributed.compile false --data.batch_size 16 \
    
    srun -n 1 python -m apps.wmtraining.train --config configs/example.yaml \
        --distributed.compile false --data.batch_size 16 \
"""

import os
from copy import deepcopy
import gc
import logging
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from apps.wmtraining.lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from apps.wmtraining.lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from apps.wmtraining.lingua.data_sft import build_sft_dataloader
from apps.wmtraining.lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from apps.wmtraining.lingua.logger import init_logger
from apps.wmtraining.lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from apps.wmtraining.lingua.optim import OptimArgs, build_optimizer
from apps.wmtraining.lingua.profiling import ProfilerArgs, maybe_run_profiler
from apps.wmtraining.lingua.tokenizer import build_tokenizer
from apps.wmtraining.lingua.probe import AutoProbeD
from apps.common.stool import StoolArgs, launch_job
from apps.wmtraining.transformer import (
    LMTransformerArgs,
    LMTransformerWM,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
    tp_parallelize,
    get_no_recompute_ops,
)
from apps.common.utils.config import dataclass_from_dict, dump_config, flatten_dict, cfg_from_cli
from apps.common.watermark.core import score_batch, WatermarkConfig

import wandb

logger = logging.getLogger()

@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: LMTransformerArgs = field(default_factory=LMTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    async_eval_qos: Optional[str] = None  # QoS for async eval jobs (cluster-specific)
    eval: Optional[Any] = None

    # SFT Mode
    sft_mode: bool = False
# sft_data field removed - now using data for both pretraining and SFT

    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)  # Use WatermarkArgs
    
    # Benchmark contamination parameters
    contamination_start_step: Optional[int] = None  # Step to start contamination
    contamination_end_step: Optional[int] = None  # Step to end contamination
    contamination_num_batches: Optional[int] = None  # Number of contamination batches
    contamination_data: Optional[DataArgs] = None  # Data config for contamination source


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: Optional[PackTokensState] = None  # Optional for SFT mode

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "step": self.step,
            "acc_step": self.acc_step,
            "scheduler": self.scheduler.state_dict(),
        }
        if self.data_loader_state is not None:
            state["data_loader_state"] = self.data_loader_state
        return state

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        if "data_loader_state" in state_dict:
            self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    
    logger.info(f"Model vocab_size: {args.model.vocab_size}")
    logger.info(f"Tokenizer output_size: {output_size}")
    
    assert (
        args.model.vocab_size == output_size
    ), f"Vocab size should be the same as output size. Model: {args.model.vocab_size}, Tokenizer: {output_size}"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert os.path.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    args.model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"
    
    # Validate contamination parameters
    if args.contamination_data is not None:
        assert args.contamination_start_step is not None, "contamination_start_step must be set when contamination_data is provided"
        assert args.contamination_end_step is not None, "contamination_end_step must be set when contamination_data is provided"
        assert args.contamination_num_batches is not None, "contamination_num_batches must be set when contamination_data is provided"
        assert args.contamination_start_step < args.contamination_end_step, "contamination_start_step must be less than contamination_end_step"
        assert args.contamination_num_batches > 0, "contamination_num_batches must be positive"
        assert args.contamination_end_step <= args.steps, "contamination_end_step must be within training steps"
        
        for source in args.contamination_data.sources:
            data_path = os.path.join(args.contamination_data.root_dir, source)
            assert os.path.exists(data_path), f"Contamination data path {data_path} doesn't exist"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        # Build tokenizer (now both modes use args.data)
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        
        validate_train_args(
            args,
            tokenizer.n_words,
        )

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)  # For handling preemption signals.
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = LMTransformerWM(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Once we shard the model on different gpus we can actually initialize the model
        # First we create empty tensors of the correct shapes
        model = model.to_empty(device="cuda")
        # Then we init the model. Please make sure this function initializes *ALL* parameters
        # and buffers, otherwise you will have random values in the unitialized tensors
        # which will silently fail (give nan gradients for example)

        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            model.rope_embeddings.reset_parameters() # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        
        # Initialize data loader state (only for pretraining mode)
        if args.sft_mode:
            data_loader_state = None  # SFT doesn't need complex state
            # For SFT, we use the number of authors from the data loader
            temp_sft_loader = build_sft_dataloader(args.data, rank=dp_rank, world_size=dp_degree)
            num_sources = len(temp_sft_loader.authors)
        else:
            data_loader_state = init_dataloader_state_from_args(
                args.data, dp_rank, dp_degree
            )
            num_sources = len(data_loader_state["it_state"]["it_state"]["it_state"]["sources"])

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        saved = False
        # Either load from latest checkpoint or start from scratch
        if args.probe_freq is not None:
            if get_is_master():
                os.makedirs(Path(args.dump_dir) / "probe", exist_ok=True)
            torch.distributed.barrier()
            probe = AutoProbeD(
                model,
                (
                    Path(args.dump_dir) / "probe" / f"probe.{dp_rank}.jsonl"
                    if (dp_rank % 128 == 0)
                    else None
                ),
            )

        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        # Build data loader (SFT or pretraining)
        if args.sft_mode:
            sft_data_loader = build_sft_dataloader(args.data, rank=dp_rank, world_size=dp_degree)
            data_loader = iter(sft_data_loader)
        else:
            data_loader = context_stack.enter_context(
                build_dataloader_from_args(
                    args.data,
                    state=train_state.data_loader_state,
                )
            )
        
        # Build contamination data loader if enabled
        contamination_loader = None
        contamination_steps = set()
        if args.contamination_data is not None:
            logger.info("Setting up contamination dataloader")
            if args.sft_mode:
                contamination_sft_loader = build_sft_dataloader(args.contamination_data, rank=dp_rank, world_size=dp_degree)
                contamination_loader = iter(contamination_sft_loader)
            else:
                contamination_data_state = init_dataloader_state_from_args(
                    args.contamination_data, dp_rank, dp_degree
                )
                contamination_loader = context_stack.enter_context(
                    build_dataloader_from_args(
                        args.contamination_data,
                        state=contamination_data_state,
                    )
                )
            
            # Calculate which steps should use contamination data
            step_range = args.contamination_end_step - args.contamination_start_step
            contamination_interval = step_range / args.contamination_num_batches
            for i in range(args.contamination_num_batches):
                contamination_step = args.contamination_start_step + int(i * contamination_interval)
                contamination_steps.add(contamination_step)
            
            logger.info(f"Contamination will occur at steps: {sorted(contamination_steps)}")
        
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )
        
        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            
            # Determine if this step should use contamination data
            use_contamination = (contamination_loader is not None and 
                                train_state.step in contamination_steps)
            
            if use_contamination:
                logger.info(f"Step {train_state.step}: Using contamination data")
                current_loader = contamination_loader
            else:
                current_loader = data_loader
            
            if args.sft_mode:
                # SFT batch processing
                sft_batch = next(current_loader)
                input_ids = sft_batch.input_ids.cuda()
                labels = sft_batch.labels.cuda()
                batch_sources = sft_batch.authors.cuda()  # Authors as sources
                qa_mask = sft_batch.qa_mask.cuda()
            else:
                # Pretraining batch processing
                if use_contamination:
                    (batch, batch_sources), _ = next(current_loader)
                else:
                    (batch, batch_sources), train_state.data_loader_state = next(current_loader)
                batch_sources = torch.tensor(batch_sources, dtype=torch.long)
                batch = torch.tensor(batch, dtype=torch.long)
                input_ids = batch[:, :, 0].cuda()
                labels = batch[:, :, 1].cuda()
                qa_mask = None  # No Q&A mask for pretraining

            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()

            bsz, seqlen = labels.shape

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # This is an automatic probe that will compute statistics
            # of all linears' inputs, weights and outputs
            # along with attention logits and entropy
            # both in forward and backward pass
            if (args.probe_freq is not None) and every_n_steps(
                train_state, args.probe_freq, acc_step=1 % args.grad_acc_steps
            ):
                # Here we do a fake forward and backward pass on a smaller
                # batch size to avoid OOM
                # This assumes the model has no stateful layers (batch norm..)
                assert (
                    next(model.parameters()).grad is None
                ), "Can't probe model if grads are not reset"

                with probe:
                    probe.metadata = {
                        "it": train_state.step,
                        "global_step": train_state.step,
                        "loop": "lingua",
                    }
                    # Non compiled model uses roughly 2x memory in our exps
                    # So we divide bsz by 2 or seqlen by 2
                    probe_bsz = max(1, bsz // 2)
                    probe_seq = seqlen if (bsz // 2 >= 1) else (seqlen // 2)
                    probe_loss = model(
                        input_ids[:probe_bsz, :probe_seq],
                        labels[:probe_bsz, :probe_seq],
                        batch_sources[:probe_bsz, :probe_seq],
                    )
                    probe_loss.backward()
                    # We zero grads to cancel this fake step
                    optimizer.zero_grad()

                assert (
                    next(model.parameters()).grad is None
                ), "Probe model shouldn't have grads at this point"

            if args.sft_mode:
                loss = model(input_ids, labels, batch_sources, sft_mode=True, qa_mask=qa_mask)
            else:
                loss = model(input_ids, labels, batch_sources)

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                # xformers.profiler.step()
                torch_profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * wps
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )
                
                # Add contamination flag if enabled
                if args.contamination_data is not None:
                    metrics["contamination/used"] = 1 if use_contamination else 0

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            if args.eval is not None and every_n_steps(
                train_state, args.checkpoint.eval.every, acc_step=0
            ):
                from apps.wmtraining.eval import (
                    launch_eval,
                    EVAL_FOLDER_NAME,
                    EvalArgs,
                )

                eval_args = dataclass_from_dict(EvalArgs, args.eval)

                eval_args.global_step = train_state.step
                eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
                eval_args.dump_dir = str(
                    os.path.join(
                        args.dump_dir,
                        "evals",
                        EVAL_FOLDER_NAME.format(train_state.step),
                    )
                )
                eval_args.metric_log_dir = args.dump_dir
                if args.async_eval_gpus is None:
                    launch_eval(eval_args)
                elif get_is_master():
                    if wandb.run is not None and args.logging.wandb is not None:
                        eval_args.wandb = deepcopy(args.logging.wandb)
                    assert args.async_eval_gpus > 0
                    logger.info(f"Launching evals on {args.async_eval_gpus} gpus")
                    with clean_env():
                        stool_args = StoolArgs(
                            asdict(eval_args),
                            script="apps.wmtraining.eval",
                            copy_code=False,
                            nodes=args.async_eval_gpus // 8,
                        )
                        if args.async_eval_qos is not None:
                            stool_args.qos = args.async_eval_qos
                        launch_job(stool_args)

            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    try:
        # Old way of parsing CLI args with stool.py
        cli_args = OmegaConf.from_cli()
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
        default_cfg = OmegaConf.structured(TrainArgs())
        cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
        cfg = OmegaConf.to_object(cfg)
    except Exception as e:
        # Parsing as arguments with argparse
        cli_args = cfg_from_cli()
        default_cfg = OmegaConf.structured(TrainArgs())
        if "config" in cli_args:
            file_cfg = OmegaConf.load(cli_args["config"])
            del cli_args["config"]
            cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
        else:
            cfg = OmegaConf.merge(default_cfg, cli_args)
        cfg = OmegaConf.to_object(cfg)
    print(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
