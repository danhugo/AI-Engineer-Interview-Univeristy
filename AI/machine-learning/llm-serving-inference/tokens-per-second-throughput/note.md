# Tokens Per Second Throughput - Interview Knowledge Sheet

## Intuition

Tokens per second is the serving system's production rate.

For LLMs, always ask which tokens are being counted:

- prompt tokens processed during prefill
- generated tokens produced during decode
- all tokens combined
- one user's perceived output rate
- the whole server's aggregate output rate

For user-facing chat, generated-token throughput is usually the most useful capacity metric:

$$
\text{TPS} = \frac{\text{total generated tokens}}{\text{measurement window seconds}}
$$

---

## 1. Throughput Is Not Latency

Throughput answers:

> How much work can the system finish per second?

Latency answers:

> How long does one request wait?

Batching can increase aggregate TPS while making an individual request slower. That tradeoff is normal in GPU serving.

---

## 2. Output TPS vs Prompt TPS

A decoder-only LLM has two major phases:

1. **Prefill**: process the input prompt and build KV cache.
2. **Decode**: generate one new token per active sequence per iteration.

Prompt-token throughput mostly measures prefill capacity:

$$
\text{prompt TPS} = \frac{\text{prompt tokens processed}}{\Delta t}
$$

Generation-token throughput mostly measures decode capacity:

$$
\text{generation TPS} = \frac{\text{output tokens generated}}{\Delta t}
$$

Long prompts stress prefill. Long answers stress decode.

---

## 3. Measurement Windows Matter

A benchmark should define the time window. Two common choices are:

$$
\text{TPS} = \frac{\text{total output tokens}}{T_y - T_x}
$$

where `Tx` is the first request timestamp and `Ty` is the last response timestamp, or:

$$
\text{TPS} = \frac{\text{total output tokens}}{T_{end} - T_{start}}
$$

where the benchmark setup and teardown are included.

The second number can be lower because it includes client overhead, warmup, request preparation, or result storage.

---

## 4. Per-User TPS

A single user's perceived generation speed is:

$$
\text{TPS}_{user} = \frac{\text{output tokens}}{\text{end-to-end latency}}
$$

For long outputs, this approaches:

$$
\frac{1}{\text{ITL}}
$$

where ITL is inter-token latency.

As concurrency rises, aggregate TPS can rise while per-user TPS falls.

---

## 5. Why TPS Saturates

TPS stops improving when a bottleneck saturates:

- GPU compute during large prefills
- memory bandwidth during decode
- KV cache capacity
- scheduler overhead
- CPU tokenization or detokenization
- network streaming overhead

More concurrent requests help only while there is idle capacity to fill.

---

## Interview Gotchas

- Report whether TPS means prompt, generation, or total tokens.
- Exclude warmup unless the benchmark explicitly includes cold start.
- Compare percentiles for latency, but use a clear window for throughput.
- Higher aggregate TPS can hide worse TTFT or ITL.
- Tokenizers differ, so token counts are not always comparable across models.

---

## References

- NVIDIA NIM LLM benchmarking metrics: https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html
- vLLM production metrics: https://docs.vllm.ai/en/v0.21.0/design/metrics/
- Hugging Face TGI overview: https://huggingface.co/docs/text-generation-inference/index
