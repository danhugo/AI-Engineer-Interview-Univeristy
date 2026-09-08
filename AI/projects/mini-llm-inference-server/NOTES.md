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
