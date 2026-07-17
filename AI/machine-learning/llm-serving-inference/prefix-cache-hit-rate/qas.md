# Prefix Cache Hit Rate - Q&A

---

## Basics

**Q: What is prefix caching?**
A: Reusing previously computed KV cache for prompt prefixes that appear again.

**Q: Which phase does prefix caching mainly speed up?**
A: Prefill, because the shared prompt tokens do not need to be recomputed.

**Q: Does prefix caching change model outputs?**
A: No, when the cache identity is correct. It reuses the same KV tensors the model would have computed.

---

## Measurement

**Q: What is a block hit rate?**
A: `reused_full_blocks / requested_full_prompt_blocks`.

**Q: What is a token-equivalent hit rate?**
A: `reused_full_blocks * block_size / prompt_tokens`.

**Q: Why can token-equivalent hit rate be lower than common-prefix fraction?**
A: Because many implementations only reuse full blocks.

---

## Correctness

**Q: Why is the parent prefix part of a block hash?**
A: The same block tokens can produce different KV values when earlier context differs.

**Q: What extra values may belong in a prefix-cache key?**
A: LoRA ID, multimodal input hash, cache salt, and any other computation-changing identity.

**Q: Why might tenants use cache salts?**
A: To prevent unintended cross-tenant cache reuse and timing leakage.

---

## System Design

**Q: When is prefix caching most useful?**
A: Long repeated prefixes such as system prompts, tool schemas, few-shot examples, and repeated documents.

**Q: When is it not useful?**
A: Short, unique prompts or traffic that only shares suffixes.

**Q: Can prefix caching increase memory pressure?**
A: Yes. Keeping reusable blocks around consumes KV cache capacity and may evict other useful blocks.
