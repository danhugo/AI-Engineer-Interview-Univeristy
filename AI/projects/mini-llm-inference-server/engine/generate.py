"""Convenience wrappers over LLMEngine.

Stage 2 had its own single-sequence loop. Stage 3 made the engine handle any
number of sequences, so this just drives it — one code path, not two.
"""

from engine.block_manager import BlockManager
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler


def generate_many(model, manager: BlockManager, prompts: list[list[int]],
                  max_new_tokens: int = 40, eos_token_id: int | None = None,
                  max_num_seqs: int = 32,
                  max_num_batched_tokens: int = 8192) -> list[list[int]]:
    """Greedy decode several prompts with continuous batching.

    Returns full token lists in the order the prompts were given (the engine
    finishes them out of order, so we re-sort by seq_id).
    """
    scheduler = Scheduler(manager, max_num_seqs, max_num_batched_tokens)
    engine = LLMEngine(model, manager, scheduler)

    first_id = None
    for prompt in prompts:
        seq = engine.add_request(prompt, max_new_tokens, eos_token_id)
        if first_id is None:
            first_id = seq.seq_id

    finished = engine.run_all()
    finished.sort(key=lambda s: s.seq_id)
    return [s.token_ids for s in finished]


def generate(model, manager: BlockManager, prompt_token_ids: list[int],
             max_new_tokens: int = 40, eos_token_id: int | None = None) -> list[int]:
    """Greedy decode one prompt."""
    return generate_many(model, manager, [prompt_token_ids],
                         max_new_tokens, eos_token_id)[0]
