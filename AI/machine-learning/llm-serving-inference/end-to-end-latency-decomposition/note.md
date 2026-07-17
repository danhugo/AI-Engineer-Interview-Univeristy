# End-to-End Latency Decomposition - Interview Knowledge Sheet

## Intuition

End-to-end LLM latency is not one thing. It is a chain of small and large waits.

For a streaming text response:

$$
\text{E2E} =
\text{client} +
\text{network} +
\text{queue} +
\text{tokenize} +
\text{prefill} +
\text{decode} +
\text{detokenize} +
\text{stream}
$$

The most important serving skill is knowing which term is dominant.

---

## 1. Request Path

A typical request path is:

1. Client builds request.
2. Request crosses the network.
3. Server validates and tokenizes input.
4. Scheduler queues the request.
5. Model runs prefill over the prompt.
6. Model decodes output tokens.
7. Server detokenizes and streams chunks.
8. Client reads the final response.

Each boundary can add latency.

---

## 2. Prefill vs Decode

Prefill processes many prompt tokens in parallel:

$$
\text{prefill work} \propto \text{input tokens}
$$

Decode generates one token per sequence per step:

$$
\text{decode steps} \propto \text{output tokens}
$$

TTFT is usually dominated by queueing plus prefill. ITL is usually dominated by repeated decode iterations.

---

## 3. Streaming Latency Shape

For `n` output tokens:

$$
\text{E2E} \approx \text{TTFT} + \sum_{i=2}^{n} \text{ITL}_i + \text{finish overhead}
$$

If average ITL is stable:

$$
\text{E2E} \approx \text{TTFT} + (n - 1)\overline{\text{ITL}} + \text{finish overhead}
$$

This explains why short answers care heavily about TTFT and long answers care heavily about ITL.

---

## 4. Queueing

Queueing grows when arrival rate approaches service capacity.

Batching can improve GPU efficiency, but requests may wait for:

- enough peers to form a useful batch
- a free KV-cache block
- the next decode scheduling iteration
- priority or fairness rules

Queue time is often the first sign that the system is overloaded.

---

## 5. How to Debug

Useful decomposition metrics:

- request queue time
- prefill time
- decode time
- TTFT
- ITL or TPOT
- E2E request latency
- prompt and output token counts
- number of running, waiting, and swapped requests

Correlate latency with token counts. A long prompt causing high TTFT is different from high TTFT caused by queueing.

---

## Interview Gotchas

- Do not optimize decode if the issue is queueing or prefill.
- Compare requests with similar input and output lengths.
- Streaming improves perceived latency but does not remove total work.
- P95/P99 latency can be dominated by queueing even when median latency looks fine.
- Network and client overhead matter for user-observed metrics.

---

## References

- vLLM production metrics: https://docs.vllm.ai/en/v0.21.0/design/metrics/
- NVIDIA NIM LLM benchmarking metrics: https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/metrics.html
- DistServe paper: https://arxiv.org/abs/2401.09670
