# Grouped Query Attention GQA - Interview Knowledge Sheet

## Intuition

Multi-head attention gives every query head its own key and value head.

That is expressive, but during autoregressive decoding the model must read cached keys and values for every previous token and every KV head.

Grouped-query attention reduces KV cache bandwidth.

It keeps many query heads, but shares fewer key/value heads across groups of query heads.

```
MHA: query heads = KV heads
GQA: query heads > KV heads > 1
MQA: many query heads share 1 KV head
```

---

## 1. Why KV Heads Matter at Inference

In decoder generation, each new token attends to all previous cached keys and values.

The KV cache size is proportional to:

$$
\text{layers} \times \text{sequence length} \times \text{KV heads} \times \text{head dim}
$$

Reducing KV heads lowers memory use and memory bandwidth.

This can speed up decoding, especially for long contexts and batch serving.

---

## 2. GQA Shape Idea

Assume:

- query heads: $h_q$
- key/value heads: $h_{kv}$
- group size: $g = h_q / h_{kv}$

Each KV head is shared by $g$ query heads.

Example:

```
8 query heads, 2 KV heads

Q heads:  0 1 2 3 | 4 5 6 7
KV head:  0 0 0 0 | 1 1 1 1
```

The implementation often repeats KV heads to match the query-head dimension before using the usual attention formula.

---

## 3. Relation to MHA and MQA

GQA includes two important endpoints:

- MHA when $h_{kv} = h_q$
- MQA when $h_{kv} = 1$

So GQA is a middle point between quality and decoding efficiency.

---

## 4. Uptraining From MHA

The GQA paper proposes converting existing multi-head checkpoints by pooling key/value heads into fewer groups and then continuing training.

The paper reports an uptraining recipe using a small fraction of original pretraining compute, with GQA approaching MHA quality while keeping speed closer to MQA.

---

## 5. What Changes and What Does Not

Changes:

- fewer key projection heads
- fewer value projection heads
- smaller KV cache

Does not change:

- number of query heads
- attention softmax idea
- output shape
- autoregressive causal masking

---

## Interview Gotchas

- GQA is mainly an inference efficiency technique for decoder attention.
- It reduces KV heads, not query heads.
- MQA is the special case with one KV head.
- MHA is the special case where KV heads equal query heads.
- KV cache memory scales with KV heads.
- Query heads must be divisible by KV heads in the common grouped layout.

---

## References

- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints": https://arxiv.org/abs/2305.13245
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need": https://arxiv.org/abs/1911.02150
