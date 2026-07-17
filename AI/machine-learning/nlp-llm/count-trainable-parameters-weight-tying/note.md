# Count Trainable Parameters with Weight Tying - Interview Knowledge Sheet

## Intuition

Language models usually have:

1. an input embedding table that maps token IDs to vectors
2. an output projection that maps hidden vectors back to vocabulary logits

Without weight tying, these are two separate large matrices. With weight tying, they share one matrix.

```
token embedding weight == language modeling head weight
```

This can save many parameters because vocabulary matrices are often huge.

---

## 1. Untied Embedding and Output Head

Let:

- `V`: vocabulary size
- `D`: hidden size

Input embedding parameters:

$$
V \cdot D
$$

Output projection parameters, ignoring bias:

$$
D \cdot V
$$

The counts are the same number because the matrices are transposes in the computation.

Untied total:

$$
2VD
$$

---

## 2. Tied Weights

With weight tying, the model learns one shared vocabulary matrix:

$$
VD
$$

The output logits use the same values:

$$
\text{logits} = hE^T
$$

where $E$ is the embedding table.

If the output head has a separate bias, add:

$$
V
$$

---

## 3. Trainable Parameter Counting

Trainable parameter count should count unique trainable tensors, not every module reference.

If two layers point to the same underlying parameter object, count it once.

That distinction matters in code:

```python
lm_head.weight = token_embedding.weight
```

The model has two references, but one trainable tensor.

---

## 4. When Tying Is Valid

The hidden dimension must match the embedding dimension. Otherwise the shared matrix cannot be used both ways.

Weight tying is natural when:

- input embeddings have shape `(V, D)`
- hidden states have size `D`
- logits need shape `(V,)` per token

It is not automatically valid if the output hidden size differs from the embedding size.

---

## Interview Gotchas

- Count shared parameters once.
- Tying usually saves `V * D` parameters.
- A separate output bias still adds `V` parameters.
- Weight tying requires compatible hidden and embedding dimensions.
- Tying changes model capacity and regularization, not just memory use.

---

## References

- Press and Wolf, "Using the Output Embedding to Improve Language Models": https://arxiv.org/abs/1608.05859
- Inan, Khosravi, and Socher, "Tying Word Vectors and Word Classifiers": https://arxiv.org/abs/1611.01462
- Hugging Face GPT-2 docs: https://huggingface.co/docs/transformers/model_doc/gpt2
