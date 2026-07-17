# Triplet Loss — Interview Knowledge Sheet

## Intuition

Triplet loss trains embeddings using three examples:

| Role | Meaning |
|------|---------|
| anchor | the reference example |
| positive | same identity/class as anchor |
| negative | different identity/class |

The goal:

```
anchor closer to positive than to negative
```

by at least a margin.

---

## 1. Formula

Let:

- `d(a, p)` be anchor-positive distance
- `d(a, n)` be anchor-negative distance
- `m` be margin

Triplet margin loss is:

$$
L = \max(0, d(a,p) - d(a,n) + m)
$$

Loss is zero when:

$$
d(a,n) \ge d(a,p) + m
$$

---

## 2. Why It Works

If the negative is already far enough, there is no loss.

If the negative is too close, the model is pushed to:

- pull anchor and positive closer
- push anchor and negative farther

---

## 3. Triplet Mining

Triplet loss depends heavily on which triplets you choose.

Easy triplets already have zero loss.

Hard triplets create learning signal.

In practice, triplet mining is often as important as the loss formula.

---

## 4. PyTorch Pattern

```python
loss = torch.nn.TripletMarginLoss(margin=1.0)(anchor, positive, negative)
```

Inputs are embeddings, not class logits.

---

## 5. Interview Gotchas

- Triplet loss trains embeddings.
- It uses anchor, positive, and negative examples.
- The margin defines how much farther the negative should be.
- Easy triplets give zero loss.
- Mining good triplets matters.

---

## References

- PyTorch `TripletMarginLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
- FaceNet paper: https://arxiv.org/abs/1503.03832
