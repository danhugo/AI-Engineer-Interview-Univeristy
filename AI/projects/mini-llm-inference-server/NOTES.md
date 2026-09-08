# Numerical notes

## Our Qwen3 is correct — proven in fp32

`diag_fp32.py` runs our model and HuggingFace's side by side in fp32
(one per GPU, SDPA on both sides):

```
max abs logit diff : 0.000000
worst-position TV  : 0.000000
exact top-1 match  : 100.00%
layer-16 max|diff| : 0.0000
```

Bit-identical. Any bf16 gap is precision, not a bug.

## In bf16, two positions blow up — and that is expected

`run.py` at seq_len 2048:

| seq_len | max abs logit diff | relative | worst-position TV | top-1 | top-5 |
|---|---|---|---|---|---|
| 6 | 0.28 | 1.1% | 0.026 | 100% | 100% |
| 128 | 0.66 | 1.5% | 0.039 | 100% | 100% |
| 512 | 6.92 | 12.8% | 0.057 | 99.0% | 100% |
| 2048 | 21.33 | 36.9% | 0.060 | 99.9% | 100% |

The absolute numbers look alarming and are mostly meaningless. What happened:

1. `diag_layers.py` — the per-layer gap grows smoothly through layer 15
   (0.06 → 3.75), then **jumps 2238x at layer 16** and stays ~8400 after.
   Embedding diff is exactly 0, so weight loading is fine.
2. `diag_layer16.py` — attention is fine (diff 4.4); the **MLP** is where it
   jumps. Only **2 of 2048 positions** are affected (603 and 571, both `,`
   tokens). Median over all positions: 0.25.
3. `diag_mlp.py` — feed both MLPs the *same* input and they are
   **bit-identical**, in bf16 and fp32. Weights identical too. So the MLP is a
   faithful amplifier, not the bug. (My cancellation guess was wrong —
   `down_proj` amplification at that element is 1.0x.)
4. `diag_fp32.py` — fp32 end to end: zero difference.

**Root cause.** Layer 16 is where Qwen3-8B grows *massive activations* —
single hidden dims reaching ~10⁴, parked on delimiter tokens like `,`. They act
as attention sinks. The MLP at those positions has enormous local gain, so the
~0.04% bf16 error arriving from the previous 16 layers (diff 2.87 on values of
8704) is amplified ~2930x.

At `(position 603, dim 2276)` the fp32 answer is **10274.6**. In bf16, HF gets
**8448** and we get **52.75**. Neither is right. bf16 has ~8 mantissa bits and
this element is chaotically sensitive — HuggingFace is not ground truth here.

## What to measure instead

Max absolute logit diff is a bad metric: it is dominated by deep-tail tokens
with ~1e-16 probability, where logits are meaningless.

Use these, in order:

1. **top-k mutual agreement** — each model's top-1 must be in the other's top-k.
   This is what vLLM's own `check_logprobs_close` does. Stayed **100%** at all
   lengths.
2. **Total variation distance** between the softmax distributions — bounded by
   how much probability mass actually moves. Stayed **≤ 0.060**.
3. **fp32 exact match** for a hard correctness gate, since bf16 noise vanishes.

Greedy decoding is unaffected: top-1 agreement is 99.9% at 2048, and the
mismatches are near-ties where either token is a reasonable pick.

## Consequence for later stages

Paged KV cache, tensor parallelism, and CUDA graphs are all supposed to be
*bit-neutral* rearrangements. So:

- Gate correctness on **fp32 exact match** — it catches real bugs with no noise
  floor to hide behind.
- Use **top-k agreement** for the bf16 path, not logit diffs.
- If a bf16 gap appears at only a couple of positions, check whether they are
  massive-activation tokens before assuming a bug.

## flash-attn constraint: paged block size must be a multiple of 256

`flash_attn_with_kvcache(..., block_table=...)` raises

```
RuntimeError: Paged KV cache block size must be divisible by 256
```

for anything smaller. This is why nano-vllm uses `block_size=256`.

Consequence for testing: you cannot use tiny blocks to force block-boundary
crossings. Use a long prompt instead — `test_stage2.py` uses 600 tokens so the
sequence spans 3 blocks.

## Do not test batching by comparing generated text

Stage 3's first test compared batched generation against running each prompt
alone, and one prompt out of six diverged. It was not a bug. `diag_batch.py`:

```
=== the decision at token 16 ===
  context: 'anoi is the pho, a noodle soup with broth'
      24.2500  ','      <- uncached, batch=6
      24.2500  ' made'  <- batch=1
  top-1 minus top-2 gap: 0.0000
```

Two tokens with **exactly** the same bf16 logit. `argmax` picks one
arbitrarily, and changing the batch size changes reduction order inside the
attention kernels — enough to flip the tie. From there the two texts are
completely different, though neither is wrong.

Two mistakes to avoid, both of which we made:

1. **Using "each prompt alone" as ground truth.** It is not; it is just another
   run. The only trusted reference is the uncached path, and even that only
   agrees up to ties. Here batch=6 happened to match it and batch=1 did not.
2. **Gating on the autoregressive trajectory.** One flipped tie at token 16
   changes every token after it. The test measures chaos, not correctness.

**Test the machinery instead.** Given the *same* token history, batching must
produce the same logits. `test_stage3.py` prefills the batch, pins the first
sampled token so both runs decode from an identical history, then compares
logits per sequence. Observed spread: 0.14-0.38 against a 0.75 tolerance,
for both prefill and decode.

## Prefix caching: publish a hash only after the K/V is written

The obvious implementation registers a block's hash when the block is
allocated. That is a race. Two sequences sharing a prefix can be admitted in
the same prefill step: the first allocates and registers, the second looks up
the hash, "hits", and skips computing those tokens — but the first has not run
its forward pass yet, so the block still holds zeros. The second sequence then
attends over an empty prefix and silently produces wrong output.

Fix: `allocate()` records pending `(block_id, hash, tokens)` on the sequence,
and `commit()` publishes them after the forward pass. Same-batch siblings miss
and each computes its own copy; the next request hits. `test_stage4.py` asserts
this directly — the cold batch must report **0** cached tokens even though all
three prompts share 512 tokens.

Second guard: confirm a hit by comparing the block's stored `token_ids`, not
just the hash. A 64-bit collision would otherwise serve the wrong K/V.

Measured on three prompts sharing a 512-token prefix:

```
[cold] cached tokens per seq: [0, 0, 0]        hit rate 0.0%
[warm] cached tokens per seq: [512, 512, 512]  hit rate 49.4%
  warm vs prefix-caching-off: max logit diff 0.125-0.234, top-1 identical
```

99% of prompt tokens needed no attention compute on the warm run.

Note that only **full** blocks are cacheable — a partial trailing block can
still grow, so its hash is not final. With `block_size=256`, prompts sharing
fewer than 256 tokens get no benefit at all.

## Tensor parallelism: SPMD removes the need for a driver/worker split

vLLM and nano-vllm run one driver rank that owns the scheduler and pushes
commands to worker ranks over shared memory or a message queue. We do not need
that. Every rank runs the *same* engine loop over the same requests; all-reduce
makes the sharded matmuls sum to the unsharded answer, so every rank derives
identical logits, samples identical tokens, and their schedulers stay in
lockstep by construction. No command channel at all.

That works because our sampling is deterministic. Adding temperature sampling
means seeding identically per step, or sampling on rank 0 and broadcasting.

Sharding scheme, standard Megatron:

| Layer | Split | Communication |
|---|---|---|
| q_proj, k_proj, v_proj | column (output dim) | none |
| o_proj | row (input dim) | all-reduce |
| gate_proj, up_proj | column | none |
| down_proj | row | all-reduce |
| embed_tokens, lm_head, norms | replicated | none |

Column then row is why there are only **two** all-reduces per layer instead of
four: q/k/v hand their sliced output straight into attention and on into
o_proj's sliced input, so the intermediate is never gathered.

The shard dimension is read off the layer class in `utils/loader.py`
(ColumnParallelLinear -> dim 0, RowParallelLinear -> dim 1) rather than from a
name table, so adding a layer cannot desync the loader.

Measured, Qwen3-8B, TP=2:

```
TP=1:  32 heads,  8 kv_heads, 144 KB/token, 8.19B params
TP=2:  16 heads,  4 kv_heads,  72 KB/token, 4.72B params per rank
prefill and decode logits vs TP=1: 0.19-0.28 max diff, identical tokens
```

4.72B x 2 exceeds 8.19B because embed_tokens and lm_head are replicated
(151936 x 4096 each). Vocab-parallelising them would save ~1.2B per rank at the
cost of an all-gather on the logits; not worth it at this scale.

A row-parallel **bias** must live on one rank only, or it gets added once per
rank. Qwen3 has none here, but the rule is in `RowParallelLinear`.

## CUDA graphs: the flat eager timing is the proof

Measured on one A100, Qwen3-8B, decode only:

```
 batch   eager ms   graph ms   speedup   max|diff|
     1      33.29      15.24     2.18x    0.000000
     4      33.63      15.64     2.15x    0.000000
    16      33.50      16.66     2.01x    0.000000
```

The important column is `eager ms`: **33.3 ms whether the batch is 1 or 16**.
Sixteen times the arithmetic for the same wall clock means the GPU is not the
bottleneck at all — the step is spent waiting on the CPU to queue ~hundreds of
tiny kernels (36 layers x matmuls, norms, RoPE, attention), each a few
microseconds of launch cost and microseconds of work.

A graph records that launch sequence once and replays it with one call, so the
CPU leaves the hot path. 2x here, and throughput then scales with batch
(30 -> 478 tok/s eager, 66 -> 960 tok/s graphed).

Prefill is deliberately not graphed: it processes whole prompts, the kernels are
large, the GPU is already saturated, and launch cost is noise.

**This is the one stage whose test demands bit-exactness.** Everywhere else
bf16 noise forces a tolerance, but a replay runs the same kernels in the same
order at the same addresses, so anything other than 0.000000 is a bug.

Three things a graph needs, and what each cost us:

- **Static addresses.** Inputs are written into pre-allocated buffers; a freshly
  allocated tensor each step would invalidate the recorded pointers.
- **Static shapes.** One graph per batch size in `BUCKETS`, real batches padded
  up to the next bucket. Block-table width is part of the shape too, so it is
  fixed at capture time.
- **A sink block.** Padded rows still execute their K/V write. Sending them to
  slot `-1` would land on the last real slot and silently corrupt a sequence,
  so the cache allocates one extra block past the usable pool and padding
  reads and writes there. Nothing ever reads it.

Capture order matters: record the largest bucket first and pass its
`graph.pool()` to the rest, or each capture allocates its own memory pool.
Warm up outside the capture as well — cuBLAS and flash-attn allocate workspaces
on first call, and that allocation must not be recorded.
