# Label Smoothing Cross-Entropy — Interview Knowledge Sheet

## Intuition

Normal cross-entropy trains on hard one-hot labels:

```
true class = 1
all other classes = 0
```

Label smoothing softens that target.

It says:

```
be mostly confident in the true class
but not infinitely confident
```

This can reduce overconfidence and improve generalization.

---

## 1. Smoothed Labels

With `K` classes and smoothing `epsilon`, replace one-hot labels with:

$$
y'_k = (1-\epsilon)y_k + \frac{\epsilon}{K}
$$

The true class gets most of the mass.

Wrong classes get a small amount.

---

## 2. Loss

Use cross-entropy with the smoothed target distribution:

$$
\text{CE} = -\sum_{k=1}^{K} y'_k\log(p_k)
$$

This prevents the model from being rewarded for pushing the true class probability all the way to `1`.

---

## 3. PyTorch Pattern

```python
loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
```

PyTorch still expects raw logits and class ID labels.

---

## 4. Interview Gotchas

- Label smoothing is regularization for classification.
- It replaces hard one-hot labels with softer targets.
- It can reduce overconfidence.
- Too much smoothing can underfit.
- PyTorch applies it inside `CrossEntropyLoss`.

---

## References

- Rethinking the Inception Architecture for Computer Vision: https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html
- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
