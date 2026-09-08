"""Tensor-parallel helpers.

Design: SPMD. Every rank runs the *same* engine loop on the same requests, and
all-reduce makes the sharded matmuls add up to the unsharded answer. Because
each rank ends up with identical logits, they sample identical tokens and their
schedulers stay in lockstep by construction — no command channel between ranks,
no driver/worker split. That is a real simplification over vLLM and nano-vllm,
which need one because their ranks do not all run the scheduler.

Launch with torchrun, which sets RANK / WORLD_SIZE / LOCAL_RANK:

    torchrun --nproc_per_node=2 test_stage5.py
"""

import os

import torch
import torch.distributed as dist


def init() -> tuple[int, int]:
    """Join the process group if launched under torchrun. Returns (rank, world)."""
    if "RANK" not in os.environ:
        return 0, 1  # plain python: single GPU, no distribution
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum a partial result across ranks, in place.

    Row-parallel layers each hold a slice of the input dimension, so each
    computes a partial sum of the same output. Adding them gives the true
    output. This is the only communication in the forward pass — two per
    transformer layer, after o_proj and after down_proj.
    """
    if world_size() > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def log(*args) -> None:
    """Print from rank 0 only."""
    if is_main():
        print(*args, flush=True)
