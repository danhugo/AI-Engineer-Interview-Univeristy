# Continuous Batching Throughput - Interview Knowledge Sheet

## Intuition

Traditional batching waits for a batch, runs it to completion, then starts another batch.

That works poorly for autoregressive LLMs because different requests produce different output lengths. One short request can finish early while its batch slot stays unavailable until the longest request completes.

Continuous batching fixes this by rescheduling at decode-iteration boundaries.

---

## 1. Request-Level vs Iteration-Level Scheduling

In request-level batching:

$$
\text{batch lifetime} = \max(\text{request output lengths})
$$

Finished requests leave unused capacity behind until the whole batch ends.

In iteration-level scheduling, after each decode step:

1. completed requests exit
2. unfinished requests remain
3. new waiting requests can enter
4. the next batch is formed

This is also called continuous batching, in-flight batching, or iteration-level batching.

---

## 2. Why It Improves Throughput

Decode is repeated many times:

$$
\text{decode work} \approx \text{one step} \times \text{output tokens}
$$

If each step can refill free slots, the GPU sees a denser active batch over time.

Higher average active batch size usually means better aggregate token throughput, until another bottleneck appears.

---

## 3. Prefill and Decode Interference

A new request must run prefill before joining decode.

Long prefills can be expensive and may interfere with ongoing decode if they share the same GPU batch. Serving systems use strategies such as:

- chunked prefill
- prefill scheduling limits
- separate prefill/decode workers
- priority rules for decode latency

The goal is to improve throughput without destroying ITL.

---

## 4. Capacity Constraints

Continuous batching still has limits:

- maximum active sequences
- maximum total tokens
- KV cache blocks
- GPU memory
- scheduler overhead
- fairness and admission control

When KV cache is full, new requests wait even if compute could run more tokens.

---

## 5. Simple Throughput Model

If each decode iteration takes `step_time_s` and has `active_sequences` requests, then approximate generation TPS is:

$$
\text{TPS} \approx \frac{\text{active sequences}}{\text{step time seconds}}
$$

Continuous batching improves the numerator by keeping active slots filled.

---

## Interview Gotchas

- Continuous batching improves utilization; it does not make every request faster.
- It can increase queueing under high load if admission is not controlled.
- Prefill and decode have different performance shapes.
- KV cache capacity can be the real bottleneck.
- Fairness matters because short requests should not be starved by long ones, or vice versa.

---

## References

- Orca OSDI paper: https://www.usenix.org/conference/osdi22/presentation/yu
- vLLM documentation: https://docs.vllm.ai/en/latest/index.html
- Triton iterative sequences and continuous batching: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
- Hugging Face TGI overview: https://huggingface.co/docs/text-generation-inference/index
