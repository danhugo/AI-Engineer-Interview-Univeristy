# KV Cache Tiered Offloading - Q&A

---

## Basics

**Q: What is KV cache offloading?**
A: Moving some cached key/value tensors from GPU memory to a slower, larger tier such as CPU RAM or SSD.

**Q: Why offload KV cache?**
A: To serve longer contexts or larger batches when GPU memory is the bottleneck.

**Q: What is the main downside?**
A: Extra data movement can increase latency or reduce throughput.

---

## Calculations

**Q: What is a simple KV cache size formula?**
A: `2 * layers * batch * tokens * hidden_size * bytes_per_element`.

**Q: How do you estimate transfer time?**
A: `bytes_moved / bandwidth_bytes_per_second`.

**Q: What must be true for prefetching to hide transfer cost?**
A: Transfer time must fit within available compute time before the data is needed.

---

## Policy

**Q: Which KV blocks should usually stay on GPU?**
A: Blocks needed soon by active decoding requests.

**Q: What blocks are safer to offload?**
A: Paused requests, low-priority requests, or blocks unlikely to be accessed soon.

**Q: Why is offloading harder for dense full-context attention?**
A: Each decode token can attend to all previous tokens, so old blocks may still be needed frequently.

---

## Gotchas

**Q: Does offloading reduce the amount of KV data?**
A: No. It changes where the data lives.

**Q: Can offloading and KV quantization be combined?**
A: Yes. Quantization reduces bytes; offloading moves bytes across tiers.

**Q: Is bandwidth math the full latency estimate?**
A: No. It is a lower bound before overheads and synchronization.
