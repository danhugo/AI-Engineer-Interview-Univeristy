# Compute TTFT, ITL, TPS from Token Timestamps - Interview Knowledge Sheet

## Intuition

A streamed LLM response is a timeline.

If you record:

- when the request was sent
- when each non-empty output token arrived
- when the response completed

then you can compute the main serving metrics without knowing the server internals.

---

## 1. Time to First Token

Time to first token is the wait until the first generated token is visible:

$$
\text{TTFT} = t_1 - t_{request}
$$

TTFT usually includes:

- client-to-server network time
- queueing
- tokenization
- prefill
- first decode step
- first-token serialization and streaming

---

## 2. Inter-Token Latency

Inter-token latency measures spacing between generated tokens:

$$
\text{ITL}_i = t_i - t_{i-1}
$$

for output tokens after the first token.

The average ITL for one request is:

$$
\overline{\text{ITL}} =
\frac{t_n - t_1}{n - 1}
$$

where `n` is the number of output tokens.

If `n = 1`, ITL is undefined because there is no gap between generated tokens.

---

## 3. End-to-End Latency

End-to-end latency is:

$$
\text{E2E} = t_{done} - t_{request}
$$

For streaming APIs, `t_done` may be the final content token or the terminal done event. Be consistent.

---

## 4. Tokens Per Second

For a single request:

$$
\text{TPS}_{user} = \frac{n}{\text{E2E}}
$$

For a benchmark:

$$
\text{TPS}_{system} =
\frac{\sum_j n_j}{t_{last\ response} - t_{first\ request}}
$$

The second formula measures aggregate system throughput.

---

## 5. Timestamp Hygiene

Good timestamp collection uses:

- monotonic clocks, not wall-clock time
- timestamps at token receipt, not after full response parsing
- exclusion of empty keepalive or done-only chunks
- the same definition across all requests

Percentiles are usually more useful than averages for TTFT and ITL.

---

## Interview Gotchas

- Do not include TTFT inside ITL unless the benchmark definition says so.
- ITL is undefined for one-token outputs.
- Empty streaming chunks should not become fake tokens.
- Client-side timestamps include network and client overhead.
- Server-side metrics and client-side metrics can legitimately differ.

---

## References

- NVIDIA NIM LLM benchmarking metrics: https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html
- vLLM production metrics: https://docs.vllm.ai/en/v0.21.0/design/metrics/
- Hugging Face TGI API and streaming endpoints: https://huggingface.github.io/text-generation-inference/
