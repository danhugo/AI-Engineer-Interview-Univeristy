"""Greedy generation using the paged KV cache.

Stage 2 handles one sequence. The prefill/decode split here is the same one the
scheduler will drive over many sequences in stage 3.
"""

import torch

from engine.block_manager import BlockManager
from engine.sequence import Sequence, SequenceStatus
from utils.context import reset_context, set_context


def _step(model, manager: BlockManager, seq: Sequence,
          is_prefill: bool) -> int:
    """Run one forward pass and return the next token id."""
    device = next(model.parameters()).device
    n = len(seq)

    if is_prefill:
        start, ids = 0, seq.token_ids            # whole prompt
    else:
        start, ids = n - 1, [seq.last_token]     # just the new token

    # Where this step's K/V goes in the cache.
    slots = manager.slot_mapping(seq, start, n)

    set_context(
        is_prefill=is_prefill,
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        # (batch=1, max_blocks) — how to find this sequence's blocks.
        block_table=torch.tensor([seq.block_table], dtype=torch.int32, device=device),
        # Valid tokens in the cache, including the one written this step.
        cache_seqlens=torch.tensor([n], dtype=torch.int32, device=device),
    )
    try:
        with torch.inference_mode():
            logits = model(
                torch.tensor([ids], dtype=torch.long, device=device),
                torch.arange(start, n, device=device),
            )
    finally:
        reset_context()

    seq.num_cached = n
    return logits[0, -1].argmax().item()


@torch.inference_mode()
def generate(model, manager: BlockManager, prompt_token_ids: list[int],
             max_new_tokens: int = 40, eos_token_id: int | None = None) -> list[int]:
    """Greedy decode one prompt. Returns the full token list."""
    seq = Sequence(prompt_token_ids, max_new_tokens, eos_token_id)
    manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING

    seq.append(_step(model, manager, seq, is_prefill=True))

    while seq.status is SequenceStatus.RUNNING:
        # A decode step may cross a block boundary and need one more block.
        manager.allocate(seq)
        seq.append(_step(model, manager, seq, is_prefill=False))

    manager.deallocate(seq)
    return seq.token_ids
