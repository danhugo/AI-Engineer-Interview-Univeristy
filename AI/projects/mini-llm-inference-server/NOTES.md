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

## The flash-attention study kernel

`layers/flash_attn_study.py`, Triton, ~120 lines. The engine never imports it;
`test_stage7.py` proves it matches.

The algorithm in one block. Plain attention materialises S = QK^T, which is
seq x seq — 64M numbers per head at 8k context, far too big for SRAM, so every
element goes to HBM and comes back twice. Flash attention never materialises
it: walk K/V in tiles that fit in SRAM and keep a running softmax.

```
m_new = max(m_old, max(tile))
alpha = exp(m_old - m_new)        # correction for everything accumulated so far
l     = l * alpha + sum(exp(tile - m_new))
acc   = acc * alpha + exp(tile - m_new) @ V_tile
```

`acc / l` at the end is exactly softmax attention. Traffic drops from
O(seq^2) to O(seq x head_dim). It is a memory-access rewrite, not an
approximation. The FlashAttention-2 detail is rescaling the accumulator and
dividing by `l` once at the end rather than renormalising each tile.

### Accuracy, measured against fp32 SDPA

```
case                                shape       ours      flash
qwen3 prefill      b2 q512 k512 h32/8 d128    0.00787    0.00846
qwen3 long        b1 q2048 k2048 h32/8 d128   0.00788    0.00786
decode-ish q=1      b4 q1 k777 h32/8 d128     0.00054    0.00065
non-causal         b2 q256 k256 h16/16 d64    0.00201    0.00200
MHA (no GQA)        b2 q320 k320 h8/8 d64     0.00703    0.00888
ragged seq len     b1 q300 k300 h4/2 d128     0.00658    0.00904
head_dim 32         b2 q128 k128 h8/2 d32     0.00735    0.00923
```

Ours is *more* accurate in 5 of 7 cases — and that is exactly why it is
**9.5x slower** (7.30 ms vs 0.77 ms). Loading q/k/v as fp32 means `tl.dot`
runs in fp32 instead of on the bf16 tensor cores. Real flash-attn keeps the
matmuls in bf16 with fp32 accumulators, and also does software pipelining,
warp specialisation, and autotuned tile shapes. None of that is here; the
point was the algorithm.

### The reference must not be `is_causal=True`

The first version of this test compared against
`scaled_dot_product_attention(is_causal=True)` and reported an error of
**3.57 for both kernels**. Two independent kernels cannot be wrong
identically — that is what gave the reference away.

Cause: when `seq_q != seq_k`, SDPA aligns the causal mask **top-left**, so a
single decode query sees only token 0. flash-attn aligns **bottom-right**, so
that query sees the whole history. The convention matters for exactly the
shape decode uses. Build the mask explicitly:

```python
mask = (torch.arange(m)[:, None] + (n - m)) >= torch.arange(n)[None, :]
```

Our kernel carries the same `shift = N - M` so it follows flash-attn.

## The serving API: keep the engine off the event loop

`server/api.py`. The engine step is blocking GPU work (15-30 ms), so it runs on
its own thread and no FastAPI handler ever touches the model. A request drops a
`Sequence` into the scheduler and waits on its own `asyncio.Queue`; the engine
thread pushes tokens in as they are produced, via
`loop.call_soon_threadsafe`. That queue is the entire coupling.

vLLM and SGLang split these across *processes* with ZeroMQ between them, so
tokenisation and detokenisation overlap the GPU loop on other cores. A thread is
the same idea one notch simpler, and enough while detokenisation is cheap.

Measured end to end over HTTP:

```
1 request     0.54s
8 concurrent  0.89s   ->  4.8x throughput vs running them one at a time
```

That 4.8x is continuous batching visible from outside the process: eight
connections arriving together get merged into one engine step.

### Two bugs worth remembering

**Never leave a subprocess's stdout in an unread pipe.** The engine thread
crashed, no tokens were ever pushed, and every request simply timed out with no
explanation — the traceback was sitting in a pipe nobody read. Two fixes: the
test now logs the server to a file and prints its tail on failure, and
`_run()` wraps the loop so a crash sets `self.fatal`, fails `/health`, and
closes every open stream instead of hanging them.

**`apply_chat_template(tokenize=True)` does not return token ids** in
transformers 5.x — it returns a `BatchEncoding`. Concatenating that into a list
of ids appends its string KEYS, which surfaces much later as
`ValueError: too many dimensions 'str'` inside `torch.tensor`. Use
`tokenize=False` and tokenize the string. `Sequence.__init__` now asserts the
prompt is a flat list of ints so this fails at the boundary with a useful
message.

### Incremental detokenisation

Decoding one token at a time is wrong: a multi-byte character spans several
tokens and emits replacement characters. `Stream.delta` decodes the whole
generated run each time and returns only the new tail. The test asserts the
concatenated stream deltas equal the non-streamed text exactly.

## Benchmarking quality: two gates, never one

`bench/intelligence.py`. Speed benchmarks cannot catch a subtly wrong kernel —
a server that is fast and wrong looks great on tok/s. But "our score is high"
is also the wrong test, because that measures the *model*, not our code.

So: two gates.

**Implementation gate** — ours vs HuggingFace, same prompts, greedy, compare
answers. Greedy for determinism. On GSM8K: **40/40 identical final answers**,
87.5% accuracy on both sides. That is the number that says paged KV, batching,
prefix caching and graphs are right.

Note identical *text* was only 15/40. Same lesson as stage 3 — exact bf16 ties
fork the trajectory. Compare answers, not transcripts.

**Quality gate** — ours with Qwen's sampling, against the published figure.

### What Qwen's docs insist on

- **Thinking is ON by default** and emits `<think>...</think>`. Pass
  `enable_thinking=False`. Thinking mode wants up to 38k output tokens, so the
  published non-thinking numbers are the only ones a short run can reach
  (MATH-500: 87.4 non-thinking vs 97.4 thinking).
- **Do not greedy decode.** Qwen warns it causes repetition loops and score
  collapse. Non-thinking wants `temperature=0.7, top_p=0.8, top_k=20`. Our
  MATH-500 greedy HF run scored 70.0% against 80.0% sampled — consistent with
  that warning. Greedy is still right for the *agreement* gate, where
  determinism matters more than the score.
- Published GSM8K/MMLU numbers are for Qwen3-8B-**Base**, not instruct. The
  instruct chat tables dropped them as saturated.

### The scorer is part of the measurement

GSM8K went from 88.5% to **95.0%** with no change to the model or the server —
only to the answer comparison. The model writes money and percentages the way
a human does:

```
gold '70000'  got '\$70,000'
gold '460'    got '\$460'
gold '60'     got '60\%'
```

Six and a half points of apparent model quality were a harness artefact. Always
print the misses before believing a score. All three of the first three misses
were formatting.

### Truncation looks exactly like stupidity

MATH-500 first scored 70.0% against a published 87.4. The tell was not the
score but the companion metric: **only 76% of answers contained `\boxed{}`**.
The rest were cut off mid-derivation, and the fallback "last number in the
text" then grabbed something arbitrary.

| max_new | accuracy | reached `\boxed{}` |
|---|---|---|
| 1024 | 70.0% | 76% |
| 2048 | 77.5% | 90% |
| 4096 | 81.5% | 96% |

Accuracy tracks the completion rate, which identifies the cause as the output
budget, not the implementation. Always report the fraction of answers that
actually finished alongside the score.

The last few points are the string comparison itself: `\frac{3}{56}` scored
wrong needs symbolic equivalence, not `==`. `normalise_math` folds the harmless
LaTeX variants but the scores here remain a floor. A real harness uses sympy.
