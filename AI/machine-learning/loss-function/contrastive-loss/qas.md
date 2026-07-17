# Contrastive Loss — Q&A

---

## Intuition

**Q: What does contrastive learning train?**
A: Embeddings.

**Q: What should happen to similar examples?**
A: Their embeddings should move closer together.

**Q: What should happen to different examples?**
A: Their embeddings should move farther apart.

---

## 1. Embeddings and Similarity

**Q: What is compared in contrastive loss?**
A: Embeddings.

**Q: What similarity is common?**
A: Cosine similarity.

**Q: Why normalize embeddings?**
A: So dot product behaves like cosine similarity.

---

## 2. NT-Xent Loss

**Q: What does NT-Xent stand for?**
A: Normalized Temperature-scaled Cross-Entropy.

**Q: What is the positive pair in SimCLR?**
A: Two augmented views of the same sample.

**Q: What does temperature control?**
A: How sharp the similarity softmax is.

---

## 3. Batch Setup

**Q: How many embeddings come from `N` samples in SimCLR?**
A: `2N`.

**Q: What are negatives?**
A: Other examples in the batch.

---

## 4. Interview Gotchas

**Q: Does contrastive loss require class labels?**
A: Not always. SimCLR uses augmented views as positives.

**Q: What is the core summary?**
A: Pull positives together and push negatives apart.
