# LoRA Adapter Hot-Swap Overhead - Q&A

---

## Basics

**Q: What is LoRA adapter hot-swapping?**
A: Changing the active LoRA adapter for inference without reloading the base model.

**Q: Why serve LoRA adapters this way?**
A: Many tasks or tenants can share one base model while using specialized small adapters.

**Q: What is the main cold-swap overhead?**
A: Loading adapter weights into the memory tier where inference needs them, often GPU memory.

---

## Calculations

**Q: How many LoRA parameters does one adapted linear layer add?**
A: `rank * (in_features + out_features)`.

**Q: How do you estimate adapter bytes?**
A: `adapter_parameters * bytes_per_parameter`.

**Q: How do you estimate ideal adapter transfer time?**
A: `adapter_bytes / bandwidth_bytes_per_second`.

---

## Serving

**Q: What happens if the adapter is already on GPU?**
A: The request avoids cold-load transfer and mostly pays routing plus LoRA compute overhead.

**Q: Why can many adapters hurt batching?**
A: Different requests need different LoRA weights, so kernels and scheduling must handle heterogeneous adapter computation.

**Q: Why can adapter cache memory reduce throughput?**
A: It competes with KV cache and activation buffers for GPU memory.

---

## Gotchas

**Q: Does hot-swapping mean changing base model weights?**
A: No. The base model remains loaded and frozen for serving.

**Q: Why should `max_lora_rank` not be set much higher than needed?**
A: It can waste memory and hurt performance.

**Q: What metric helps diagnose hot-swap overhead?**
A: Adapter cache hit rate, cold-load latency, and per-adapter request volume.
