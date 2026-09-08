"""The engine loop.

    step() = schedule -> run -> postprocess

That is the whole thing. run() turns a list of sequences into flat tensors,
sets the attention context, does one forward pass, and samples. Everything else
in this project exists to make run() cheap.
"""

import torch

from engine.block_manager import BlockManager
from engine.scheduler import Scheduler
from engine.sequence import Sequence
from utils.context import reset_context, set_context


class LLMEngine:
    def __init__(self, model, manager: BlockManager, scheduler: Scheduler,
                 graph_runner=None):
        self.model = model
        self.manager = manager
        self.scheduler = scheduler
        # Optional DecodeGraphRunner. Decode is launch-bound, so replaying a
        # recorded graph is much cheaper than queueing hundreds of kernels.
        self.graph_runner = graph_runner
        self.device = next(model.parameters()).device

    def add_request(self, prompt_token_ids: list[int], max_new_tokens: int = 40,
                    eos_token_id: int | None = None) -> Sequence:
        seq = Sequence(prompt_token_ids, max_new_tokens, eos_token_id)
        self.scheduler.add(seq)
        return seq

    # ---------------- tensor prep ----------------

    def _long(self, xs):
        return torch.tensor(xs, dtype=torch.long, device=self.device)

    def _int32(self, xs):
        return torch.tensor(xs, dtype=torch.int32, device=self.device)

    def _prepare_prefill(self, seqs: list[Sequence]):
        """Concatenate the tokens that still need computing.

        With prefix caching a sequence may arrive with part of its history
        already in the cache. Those tokens are skipped here — they are still
        attended to, via the block table, but never recomputed. So there are
        two sets of lengths: q over new tokens, k over the full history.
        """
        input_ids, positions, slots = [], [], []
        cu_q, cu_k = [0], [0]
        for seq in seqs:
            n, start = len(seq), seq.num_cached
            input_ids += seq.token_ids[start:]
            positions += list(range(start, n))
            slots += self.manager.slot_mapping(seq, start, n)
            cu_q.append(cu_q[-1] + (n - start))
            cu_k.append(cu_k[-1] + n)

        ctx = dict(
            is_prefill=True,
            slot_mapping=self._long(slots),
            cu_seqlens_q=self._int32(cu_q),
            cu_seqlens_k=self._int32(cu_k),
            max_seqlen_q=max(s.num_uncached for s in seqs),
            max_seqlen_k=max(len(s) for s in seqs),
            block_table=self._block_table(seqs),
        )
        # Only the last token of each sequence predicts anything.
        logits_indices = self._long([c - 1 for c in cu_q[1:]])
        return self._long(input_ids), self._long(positions), ctx, logits_indices

    def _block_table(self, seqs: list[Sequence]) -> torch.Tensor:
        """Pad the per-sequence block lists into a rectangle.

        Padding is never read: cu_seqlens_k (prefill) and cache_seqlens
        (decode) bound how far each sequence looks.
        """
        width = max(len(s.block_table) for s in seqs)
        return self._int32([s.block_table + [0] * (width - len(s.block_table))
                            for s in seqs])

    def _prepare_decode(self, seqs: list[Sequence]):
        """One token per sequence; history is reached via the block tables."""
        input_ids, positions, slots, seqlens = [], [], [], []
        for seq in seqs:
            n = len(seq)
            input_ids.append(seq.last_token)
            positions.append(n - 1)
            slots += self.manager.slot_mapping(seq, n - 1, n)
            seqlens.append(n)

        ctx = dict(
            is_prefill=False,
            slot_mapping=self._long(slots),
            block_table=self._block_table(seqs),
            cache_seqlens=self._int32(seqlens),
        )
        return self._long(input_ids), self._long(positions), ctx, None

    # ---------------- the loop ----------------

    @torch.inference_mode()
    def forward_logits(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        """One forward pass. Returns (num_seqs, vocab) next-token logits.

        Split out from run() so tests can compare logits across batch sizes.
        Comparing generated *tokens* instead is unreliable: greedy decoding
        amplifies exact bf16 ties into completely different text.
        """
        prep = self._prepare_prefill if is_prefill else self._prepare_decode
        input_ids, positions, ctx, logits_indices = prep(seqs)

        runner = self.graph_runner
        if (not is_prefill and runner is not None
                and runner.can_run(len(seqs), ctx["block_table"].shape[1])):
            logits = runner.run(input_ids, positions, ctx)
        else:
            set_context(**ctx)
            try:
                logits = self.model(input_ids, positions, logits_indices)
            finally:
                reset_context()

        # The K/V now exists, so any block that filled up is safe to share.
        for seq in seqs:
            seq.num_cached = len(seq)
            self.manager.commit(seq)
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        logits = self.forward_logits(seqs, is_prefill)
        return logits.argmax(-1).tolist()  # greedy; a Sampler goes here later

    def step(self) -> list[Sequence]:
        """Advance every scheduled sequence by one token."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []
        token_ids = self.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        return seqs

    def run_all(self) -> list[Sequence]:
        """Drain the queue. Returns finished sequences in completion order."""
        while self.scheduler.has_work:
            self.step()
        return self.scheduler.finished
