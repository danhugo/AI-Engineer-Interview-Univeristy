# Prefix Cache Hit Rate - Interview Knowledge Sheet

## Intuition

Prefix caching reuses the KV cache produced by earlier prompt tokens.

If many requests start with the same system prompt, tool schema, retrieved document, or chat history, the server should not recompute those shared tokens every time.

The important detail is that most serving systems cache KV data in blocks, not individual tokens. A request can only reuse the already computed full blocks that exactly match its prefix.

---

## 1. What Gets Cached

During prefill, each prompt token produces key and value tensors for every transformer layer.

For a prompt of length $T$, the server stores KV cache roughly proportional to:

$$
2 \cdot L \cdot T \cdot H \cdot \text{bytes}
$$

where `2` means key and value, $L$ is layers, and $H$ is hidden size across heads.

Prefix caching stores those computed KV blocks so a later request with the same prefix can skip part of prefill.

---

## 2. Block Granularity

Paged KV cache managers usually split tokens into fixed-size blocks.

If `block_size = 16`, then tokens `0..15` form block 0, tokens `16..31` form block 1, and so on.

A cache hit for 31 matching prefix tokens only reuses:

$$
\left\lfloor \frac{31}{16} \right\rfloor = 1
$$

full block. The 15-token tail still needs computation unless the implementation has a separate partial-block mechanism.

---

## 3. Hit Rate Definitions

Be precise about the numerator.

Block hit rate:

$$
\text{block hit rate} =
\frac{\text{reused full blocks}}{\text{requested full prompt blocks}}
$$

Token-equivalent hit rate:

$$
\text{token hit rate} =
\frac{\text{reused full blocks} \cdot \text{block size}}{\text{prompt tokens}}
$$

The first is about cache lookup effectiveness. The second is closer to saved prefill work.

---

## 4. Hashing and Correctness

A cached block is only valid for the same tokens and the same preceding context.

For block $i$, a safe cache key depends on:

$$
\text{hash}(\text{parent hash}, \text{tokens in block}, \text{extra identity})
$$

The extra identity can include LoRA adapter ID, multimodal input hash, tenant salt, or other values that change model computation.

This prevents two requests with the same local block tokens but different earlier context from sharing an invalid KV block.

---

## 5. When Hit Rate Is High

Prefix caching helps most when:

- prompts have long repeated prefixes
- traffic has repeated system prompts, few-shot examples, or tool schemas
- RAG requests reuse the same retrieved document prefix
- requests arrive before cached blocks are evicted
- cache salt and adapter identity allow sharing

It helps less when prompts are short, highly unique, or only share suffixes.

---

## Interview Gotchas

- Prefix caching saves prefill, not autoregressive decode.
- Cache reuse must be exact for all tokens that affect the KV values.
- Full-block caching means matching partial tails may not count.
- Cache salts improve isolation but can reduce cross-tenant hit rate.
- LoRA or multimodal inputs should be part of the cache identity.
- A high request-level hit rate can still save little work if only short prefixes hit.

---

## References

- vLLM automatic prefix caching design: https://docs.vllm.ai/en/stable/design/prefix_caching/
- vLLM prefix caching implementation notes: https://docs.vllm.ai/en/v0.6.6/automatic_prefix_caching/details.html
- vLLM RFC for automatic prefix caching: https://github.com/vllm-project/vllm/issues/2614
