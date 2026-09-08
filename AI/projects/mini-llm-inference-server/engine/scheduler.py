"""Continuous batching: decide what runs in the next step.

The old way was static batching — group N requests, run them to completion,
then take the next N. A short request stuck behind a long one waits for the
whole batch, and finished slots sit idle.

Continuous batching schedules per *token step* instead. Every step we pick who
advances; finished sequences leave immediately and waiting ones take their slot.

One step is either all-prefill or all-decode, never mixed. Prefill wins when
anything is waiting, because a request cannot produce its first token until its
prompt is in the cache. (vLLM mixes them via chunked prefill; that is a later
refinement.)
"""

from collections import deque

from engine.block_manager import BlockManager
from engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, manager: BlockManager,
                 max_num_seqs: int = 32,
                 max_num_batched_tokens: int = 8192):
        self.manager = manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Return (sequences to run, is_prefill)."""
        scheduled: list[Sequence] = []

        # --- prefill: admit waiting sequences while budget allows ---
        num_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            if num_tokens + len(seq) > self.max_num_batched_tokens and scheduled:
                break  # token budget spent; leave it for the next step
            if not self.manager.can_allocate(seq):
                break  # no room in the cache
            self.waiting.popleft()
            self.manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled.append(seq)
            # Charge the budget for work actually done: a prefix-cache hit
            # means those tokens are never pushed through the model.
            num_tokens += seq.num_uncached

        if scheduled:
            return scheduled, True

        # --- decode: every running sequence gets one token ---
        # A decode step may need one more block. If the cache is full, preempt
        # the most recently admitted sequence (it has done the least work) and
        # send it back to the queue to be recomputed later.
        for _ in range(len(self.running)):
            seq = self.running.popleft()
            while not self.manager.can_allocate(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                continue
            self.manager.allocate(seq)
            scheduled.append(seq)

        # Put them back in order for the next round.
        self.running.extend(scheduled)
        return scheduled, False

    def preempt(self, seq: Sequence) -> None:
        """Evict a sequence's blocks and requeue it.

        vLLM calls this recompute preemption: throwing the KV away and
        re-prefilling later is cheaper than swapping it to host memory.
        """
        self.manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.token_ids = seq.token_ids[:seq.prompt_len]  # generated text is lost
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """Append sampled tokens and retire whatever finished."""
        for seq, token_id in zip(seqs, token_ids):
            seq.append(token_id)
            if seq.status is SequenceStatus.FINISHED:
                self.manager.deallocate(seq)
                self.running.remove(seq)
                self.finished.append(seq)
