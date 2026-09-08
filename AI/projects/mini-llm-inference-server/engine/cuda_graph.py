"""CUDA graphs for the decode step.

Why decode only. A decode step computes one token per sequence, so every kernel
is tiny — microseconds of work. But the step still launches hundreds of them
(36 layers x matmuls, norms, RoPE, attention), each costing the CPU a few
microseconds to queue. The GPU finishes each kernel and waits for the CPU to
send the next, so the step is bound by launch overhead, not arithmetic.

A CUDA graph records the whole launch sequence once and replays it with a single
call. Hundreds of launches become one; the CPU leaves the hot path.

Prefill gets no benefit: it processes whole prompts, the kernels are large, the
GPU is already saturated, and launch cost is noise.

The cost is rigidity. A graph replays exactly what it recorded, so shapes and
memory addresses must be identical every time. Hence:

  static buffers   inputs are written into pre-allocated tensors, never freshly
                   allocated, so the recorded addresses stay valid.
  bucketed sizes   one graph per batch size in BUCKETS; a real batch is padded
                   up to the next bucket.
  a sink slot      padded rows still execute a K/V write. They are aimed at the
                   block manager's sink block so they cannot corrupt real data.
"""

import torch

from engine.block_manager import BlockManager
from utils.context import reset_context, set_context

BUCKETS = (1, 2, 4, 8, 16, 32)


class DecodeGraphRunner:
    def __init__(self, model, manager: BlockManager, max_batch: int = 32,
                 max_blocks_per_seq: int | None = None):
        self.model = model
        self.manager = manager
        self.device = next(model.parameters()).device

        self.buckets = [b for b in BUCKETS if b <= max_batch] or [max_batch]
        self.max_batch = self.buckets[-1]
        # Block-table width is part of the graph's shape, so it is fixed too.
        self.width = max_blocks_per_seq or manager.num_blocks

        n, w = self.max_batch, self.width
        self.input_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        self.positions = torch.zeros(n, dtype=torch.long, device=self.device)
        self.slot_mapping = torch.zeros(n, dtype=torch.long, device=self.device)
        self.block_table = torch.zeros(n, w, dtype=torch.int32, device=self.device)
        self.cache_seqlens = torch.zeros(n, dtype=torch.int32, device=self.device)

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.captured = False

    def _set_context(self, size: int) -> None:
        set_context(
            is_prefill=False,
            slot_mapping=self.slot_mapping[:size],
            block_table=self.block_table[:size],
            cache_seqlens=self.cache_seqlens[:size],
        )

    @torch.inference_mode()
    def capture(self) -> None:
        """Record one graph per bucket.

        Largest bucket first so the smaller graphs can share its memory pool;
        capturing small-to-large would allocate a fresh pool each time.
        """
        # Point everything at the sink with one valid cached token, so capture
        # runs real kernels without touching any sequence's blocks.
        self.slot_mapping.fill_(self.manager.sink_slot)
        self.block_table.fill_(self.manager.sink_block_id)
        self.cache_seqlens.fill_(1)
        self.input_ids.zero_()
        self.positions.zero_()

        pool = None
        for size in sorted(self.buckets, reverse=True):
            self._set_context(size)
            # Warm up outside the graph: cuBLAS and flash-attn allocate
            # workspaces on first call, and that must not land in the capture.
            self.model(self.input_ids[:size], self.positions[:size])
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out = self.model(self.input_ids[:size], self.positions[:size])
            pool = graph.pool()
            self.graphs[size] = graph
            self.outputs[size] = out

        reset_context()
        torch.cuda.synchronize()
        self.captured = True

    def bucket_for(self, batch: int) -> int | None:
        for b in self.buckets:
            if b >= batch:
                return b
        return None

    def can_run(self, batch: int, width: int) -> bool:
        return self.captured and width <= self.width and self.bucket_for(batch) is not None

    @torch.inference_mode()
    def run(self, input_ids, positions, ctx_kwargs) -> torch.Tensor:
        """Replay the decode graph. Returns (batch, vocab) logits."""
        batch = input_ids.shape[0]
        size = self.bucket_for(batch)
        table = ctx_kwargs["block_table"]
        width = table.shape[1]

        self.input_ids[:batch].copy_(input_ids)
        self.positions[:batch].copy_(positions)
        self.slot_mapping[:batch].copy_(ctx_kwargs["slot_mapping"])
        self.cache_seqlens[:batch].copy_(ctx_kwargs["cache_seqlens"])
        self.block_table[:batch, :width].copy_(table)
        # Columns past this sequence's blocks are never read (cache_seqlens
        # bounds the walk), but keep them pointing somewhere harmless.
        if width < self.width:
            self.block_table[:batch, width:] = self.manager.sink_block_id

        # Padded rows: one token of history, all reads and writes at the sink.
        if size > batch:
            self.input_ids[batch:size].zero_()
            self.positions[batch:size].zero_()
            self.slot_mapping[batch:size] = self.manager.sink_slot
            self.cache_seqlens[batch:size] = 1
            self.block_table[batch:size] = self.manager.sink_block_id

        self.graphs[size].replay()
        return self.outputs[size][:batch]
