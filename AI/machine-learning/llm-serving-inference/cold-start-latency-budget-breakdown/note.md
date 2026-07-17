# Cold Start Latency Budget Breakdown - Interview Knowledge Sheet

## Intuition

Cold start is the hidden work paid before a model can serve quickly.

For LLM serving, the first request may pay for:

- container startup
- model download or disk reads
- weight deserialization
- GPU memory allocation
- tensor parallel initialization
- kernel compilation or autotuning
- CUDA graph capture
- KV cache allocation
- tokenizer loading
- first real prefill/decode

Warmup moves much of that cost before live traffic.

---

## 1. Cold vs Warm Latency

Cold latency includes one-time setup:

$$
\text{cold E2E} = \text{startup} + \text{load} + \text{initialize} + \text{first inference}
$$

Warm latency is the steady-state request path:

$$
\text{warm E2E} = \text{queue} + \text{tokenize} + \text{prefill} + \text{decode} + \text{stream}
$$

A cold first token can be seconds or minutes slower than a warm first token for large models.

---

## 2. Budget Categories

A practical cold-start budget might separate:

| Category | Examples |
| --- | --- |
| Image and process | pull image, start container, import Python |
| Model materialization | download, read shards, mmap, deserialize |
| GPU setup | allocate weights, initialize communicators, reserve KV cache |
| Kernel setup | compile, autotune, graph capture |
| Warmup inference | synthetic prefill/decode for expected shapes |
| Readiness | health check, register endpoint, start routing |

This makes it clear which team owns each part.

---

## 3. Warmup

Warmup sends synthetic requests before real traffic.

It can pre-pay:

- lazy allocations
- kernel selection
- graph capture
- tokenizer/model path initialization
- common input and batch shapes

Warmup must resemble expected traffic. Warming only a tiny prompt may not prepare long-prompt or high-batch paths.

---

## 4. Capacity and Autoscaling

Cold start matters most when scaling from zero or reacting to sudden traffic.

If cold start is 90 seconds and traffic spikes now, new replicas cannot help immediately. You may need:

- warm pools
- predictive scaling
- smaller model variants
- preloaded weights
- pinned images
- readiness gates after warmup

---

## 5. Measuring a Budget

Track timestamps for:

1. deploy requested
2. process started
3. model files available
4. weights loaded to GPU
5. warmup started
6. warmup completed
7. endpoint ready
8. first real token emitted

Then compute each segment:

$$
\text{segment}_i = t_{i+1} - t_i
$$

---

## Interview Gotchas

- Do not route production traffic before readiness if warmup is required.
- A health check can pass before the model is truly warm.
- Cold-start budgets should use P95/P99, not only happy-path averages.
- Weight download and GPU initialization are different bottlenecks.
- Warmup shapes should match real prompt lengths, batch sizes, and decode behavior.

---

## References

- Triton model warmup docs: https://triton-inference-server.github.io/model_navigator/0.7.0/triton/warmup/
- PyTriton model config with model warmup: https://triton-inference-server.github.io/pytriton/latest/reference/model_config/
- vLLM documentation: https://docs.vllm.ai/en/latest/index.html
