# Disaggregated Prefill-Decode Serving - Interview Knowledge Sheet

## Intuition

LLM inference has two different phases.

Prefill processes the full prompt and builds the KV cache. It is compute-heavy and parallel across prompt tokens.

Decode generates one new token at a time using the KV cache. It is latency-sensitive and often memory-bandwidth-bound.

Disaggregated serving runs these phases on separate worker pools:

$$
\text{prefill GPUs} \rightarrow \text{KV transfer} \rightarrow \text{decode GPUs}
$$

---

## 1. Why Separate Them

Colocating prefill and decode can cause interference.

A long prompt prefill can delay decode steps for active users, increasing time per output token.

Separating pools lets each phase use a different batching and parallelism strategy:

| Phase | Common bottleneck | User metric |
| --- | --- | --- |
| Prefill | prompt compute | TTFT |
| Decode | KV-cache memory traffic and scheduling | ITL / TPOT |

---

## 2. Core Metrics

Time to first token:

$$
\text{TTFT} = \text{queue}_{p} + \text{prefill} + \text{KV transfer} + \text{first decode}
$$

Inter-token latency:

$$
\text{ITL} \approx \text{decode step interval}
$$

Disaggregation should improve decode stability, but it can hurt TTFT if KV transfer or prefill queueing is too high.

---

## 3. KV Cache Transfer

The decode worker needs the KV cache produced by prefill.

A simplified KV cache size is:

$$
2 \cdot L \cdot T \cdot H \cdot b
$$

where:

- `2`: key and value
- `L`: layers
- `T`: prompt tokens
- `H`: hidden size
- `b`: bytes per element

The transfer path must be fast enough that separation does not erase the scheduling benefit.

---

## 4. Resource Allocation

If prompts are long and outputs are short, prefill may need more capacity.

If outputs are long, decode workers may dominate.

A simple utilization target is:

$$
\rho_p = \lambda \cdot t_p / N_p
$$

$$
\rho_d = \lambda \cdot t_d / N_d
$$

where `lambda` is request rate, `t_p` and `t_d` are per-request service times, and `N_p`, `N_d` are worker counts.

---

## Interview Gotchas

- Prefill and decode stress hardware differently.
- Disaggregation adds KV-transfer overhead.
- It can improve TPOT while not always improving TTFT.
- Separate pools let schedulers tune batching independently.
- The best split depends on prompt length, output length, SLOs, and network bandwidth.

---

## References

- DistServe paper: https://arxiv.org/abs/2401.09670
- Perplexity Research on disaggregated prefill and decode: https://research.perplexity.ai/articles/disaggregated-prefill-and-decode
- vLLM paper: https://arxiv.org/abs/2309.06180
