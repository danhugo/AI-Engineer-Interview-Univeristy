"""Per-step attention context.

The attention layers need to know where to write K/V and where to read it back.
Threading that through every module's forward() would mean changing the model's
signature for something the model does not care about, so we stash it here and
the attention layer picks it up. nano-vllm and vLLM both do this.

Set it once per engine step, clear it after.
"""

from dataclasses import dataclass

import torch


@dataclass
class Context:
    # Prefill processes a whole prompt; decode does one token per sequence.
    is_prefill: bool = False
    # Flat KV-cache slot for each token in this step: block_id * block_size + offset.
    slot_mapping: torch.Tensor | None = None
    # (batch, max_blocks_per_seq) physical block ids per sequence.
    block_table: torch.Tensor | None = None
    # (batch,) how many tokens each sequence has in the cache, including this step.
    cache_seqlens: torch.Tensor | None = None


_context = Context()


def get_context() -> Context:
    return _context


def set_context(**kwargs) -> None:
    global _context
    _context = Context(**kwargs)


def reset_context() -> None:
    global _context
    _context = Context()
