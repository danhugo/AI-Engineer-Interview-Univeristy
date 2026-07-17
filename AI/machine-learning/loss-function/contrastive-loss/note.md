# Contrastive Loss — Interview Knowledge Sheet

## Intuition

Contrastive learning trains embeddings by comparison.

It says:

```
pull similar examples together
push different examples apart
```

In SimCLR, each image has two augmented views. Those two views are a positive pair. Other views in the batch are negatives.

---

## 1. Embeddings and Similarity

The model maps inputs to embeddings:

```
x -> z
```

Then we compare embeddings with cosine similarity.

Similar pairs should have high similarity.

Different pairs should have lower similarity.

Embeddings are usually L2-normalized before cosine similarity.

---

## 2. NT-Xent Loss

SimCLR uses NT-Xent:

```
Normalized Temperature-scaled Cross-Entropy
```

For one anchor `i` and its positive `j`:

$$
\ell_{i,j}
= -\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
\sum_{k \ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}
$$

`tau` is temperature.

Lower temperature makes the model care more about the most similar examples.

---

## 3. Batch Setup

For `N` original samples, create two views each.

The batch has `2N` embeddings.

If the first `N` rows are view 1 and the next `N` rows are view 2:

```
positive of i is i + N
positive of i + N is i
```

Every other row is a negative.

---

## 4. Interview Gotchas

- Contrastive loss trains embeddings, not direct class logits.
- Positive pairs should be close.
- Negative pairs should be far apart.
- SimCLR uses in-batch negatives.
- Temperature changes how sharp the softmax is.
- Normalize embeddings before cosine similarity.

---

## References

- SimCLR paper: https://arxiv.org/abs/2002.05709
- PyTorch cosine similarity: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
