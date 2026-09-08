"""One request in flight."""

from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()   # queued, no blocks yet
    RUNNING = auto()   # has blocks, generating
    FINISHED = auto()  # hit EOS or the token limit


class Sequence:
    """Token ids plus the KV-cache blocks holding this sequence's K/V.

    block_table maps logical block index -> physical block id. Logical block i
    covers tokens [i*block_size, (i+1)*block_size). The blocks need not be
    contiguous in memory — that is the whole point of paging.
    """

    counter = 0

    def __init__(self, prompt_token_ids: list[int], max_new_tokens: int = 64,
                 eos_token_id: int | None = None,
                 temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0):
        self.seq_id = Sequence.counter
        Sequence.counter += 1

        prompt_token_ids = list(prompt_token_ids)
        assert prompt_token_ids, "empty prompt"
        assert all(isinstance(t, int) for t in prompt_token_ids), (
            "prompt_token_ids must be a flat list of ints, got "
            f"{type(prompt_token_ids[0]).__name__} — a tokenizer returning a "
            "BatchEncoding or nested list is the usual cause"
        )
        self.prompt_len = len(prompt_token_ids)
        self.token_ids = list(prompt_token_ids)
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # "stop" (hit EOS) or "length" (hit max_new_tokens), for the API.
        self.finish_reason: str | None = None

        self.status = SequenceStatus.WAITING
        self.block_table: list[int] = []
        # Tokens whose K/V is already in the cache. A prefix-cache hit makes
        # this non-zero before the sequence has ever run.
        self.num_cached = 0
        # Block hashes waiting to be published once their K/V is written.
        self.pending_hashes: list[tuple[int, int, list[int]]] = []

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_new_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    def append(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        if token_id == self.eos_token_id:
            self.status = SequenceStatus.FINISHED
            self.finish_reason = "stop"
        elif self.num_new_tokens >= self.max_new_tokens:
            self.status = SequenceStatus.FINISHED
            self.finish_reason = "length"

    def num_blocks_needed(self, block_size: int) -> int:
        return -(-len(self) // block_size)  # ceil

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, len={len(self)}, "
                f"cached={self.num_cached}, blocks={self.block_table}, "
                f"status={self.status.name})")

    @property
    def num_uncached(self) -> int:
        """Tokens this sequence still has to run through the model."""
        return len(self) - self.num_cached
