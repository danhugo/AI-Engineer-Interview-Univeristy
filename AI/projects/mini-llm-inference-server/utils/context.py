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
    """Everything attention needs for one batched step.

    Tokens are flattened: a prefill step carrying prompts of 5 and 3 tokens has
    8 rows, and cu_seqlens [0, 5, 8] says where each sequence starts and ends.
    That is how one kernel call serves sequences of different lengths without
    padding, which is what makes continuous batching cheap.
    """

    # Prefill processes whole prompts; decode does one token per sequence.
    is_prefill: bool = False

    # Flat KV-cache slot per token this step: block_id * block_size + offset.
    slot_mapping: torch.Tensor | None = None

    # Prefill only: cumulative sequence boundaries, length batch+1.
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Decode only: (batch, max_blocks) block ids, and (batch,) cached lengths.
    block_table: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None

    @property
    def active(self) -> bool:
        """True when the paged path should be used at all."""
        return self.slot_mapping is not None


_context = Context()


def get_context() -> Context:
    return _context


def set_context(**kwargs) -> None:
    global _context
    _context = Context(**kwargs)


def reset_context() -> None:
    global _context
    _context = Context()
