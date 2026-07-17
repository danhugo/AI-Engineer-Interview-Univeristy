# Cross-Entropy Loss — Interview Knowledge Sheet

## Intuition

Cross-entropy measures the difference between:

- the true label distribution
- the predicted probability distribution

For classification, the correct class should get high probability.

Cross-entropy is small when that happens.

It is large when the correct class gets low probability.

---

## 1. Multi-Class Setting

Use cross-entropy when exactly one class is correct:

```
cat OR dog OR bird
```

The model outputs one logit per class:

```
logits = [score_cat, score_dog, score_bird]
```

Softmax turns logits into probabilities.

---

## 2. Formula

For one-hot labels:

$$
\text{CE} = -\sum_{k=1}^{K} y_k\log(p_k)
$$

Only the true class has `y_k = 1`, so for class ID `y`:

$$
\text{CE} = -\log(p_y)
$$

It is the negative log probability of the true class.

---

## 3. PyTorch Pattern

PyTorch `CrossEntropyLoss` expects raw logits and class IDs:

```python
loss = torch.nn.CrossEntropyLoss()(logits, labels)
```

Do not apply softmax first.

`CrossEntropyLoss` combines:

```
log_softmax -> NLLLoss
```

in one numerically stable operation.

---

## 4. Binary vs Multi-Class

Use binary cross-entropy for binary or independent yes/no labels.

Use cross-entropy for one correct class out of many.

| Problem | Loss |
|---------|------|
| spam or not spam | `BCEWithLogitsLoss` |
| cat or dog or bird | `CrossEntropyLoss` |
| multiple tags can be true | `BCEWithLogitsLoss` |

---

## 5. Interview Gotchas

- Cross-entropy is for classification.
- It compares true and predicted distributions.
- For one-hot labels, it becomes `-log(probability of true class)`.
- PyTorch expects logits, not softmax probabilities.
- PyTorch targets are usually class IDs, not one-hot vectors.

---

## References

- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- scikit-learn `log_loss`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
