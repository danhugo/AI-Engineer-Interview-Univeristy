# Triplet Loss — Q&A

---

## Intuition

**Q: What does triplet loss train?**
A: Embeddings.

**Q: What are the three examples?**
A: Anchor, positive, and negative.

**Q: What is the goal?**
A: Make the anchor closer to the positive than to the negative.

---

## 1. Formula

**Q: What is the triplet loss formula?**
A: `max(0, d(anchor, positive) - d(anchor, negative) + margin)`.

**Q: When is the loss zero?**
A: When the negative is farther than the positive by at least the margin.

**Q: What does the margin control?**
A: How much farther the negative must be.

---

## 2. Why It Works

**Q: What happens when the negative is too close?**
A: The model gets loss and must separate it.

**Q: What happens when the negative is far enough?**
A: The loss is zero.

---

## 3. Triplet Mining

**Q: Why does triplet mining matter?**
A: Easy triplets give no learning signal.

**Q: What kind of triplets are useful?**
A: Hard or semi-hard triplets.

---

## 4. PyTorch Pattern

**Q: What PyTorch class implements triplet margin loss?**
A: `torch.nn.TripletMarginLoss`.

**Q: Does it take logits?**
A: No. It takes embeddings.

---

## 5. Interview Gotchas

**Q: What is the core summary?**
A: Pull anchor-positive together and push anchor-negative apart by a margin.
