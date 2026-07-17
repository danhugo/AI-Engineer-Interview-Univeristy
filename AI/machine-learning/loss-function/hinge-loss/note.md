# Hinge Loss — Interview Knowledge Sheet

## Intuition

Hinge loss is a margin loss.

It does not only ask:

```
Was the class correct?
```

It also asks:

```
Was it correct by a large enough margin?
```

This is the classic loss behind support vector machines.

---

## 1. Binary Hinge Loss

For labels `y` in `{-1, +1}` and score `s`:

$$
\text{hinge}(y, s) = \max(0, 1 - ys)
$$

If `ys >= 1`, the prediction is correct with enough margin, so loss is `0`.

If `ys < 1`, the model pays loss.

---

## 2. Margin Intuition

The value `ys` combines correctness and confidence.

| `ys` | Meaning | Loss |
|------|---------|------|
| large positive | correct with margin | 0 |
| small positive | correct but too close | positive |
| negative | wrong side | large |

Hinge loss keeps pushing until the example is beyond the margin.

---

## 3. Multi-Class Hinge Loss

For class scores `s` and true class `y`:

$$
L_i = \sum_{j \ne y_i}\max(0, s_j - s_{y_i} + \Delta)
$$

The true class score should be higher than every wrong class score by at least margin `Delta`.

---

## 4. When to Use

Use hinge loss for margin-based classifiers such as SVMs.

For modern neural classification, cross-entropy is more common.

Hinge loss works with raw scores, not probabilities.

---

## 5. Interview Gotchas

- Binary hinge labels are usually `-1` and `+1`, not `0` and `1`.
- Hinge loss uses raw scores.
- Correct predictions can still have loss if the margin is too small.
- Once the margin is satisfied, the loss is zero.
- Cross-entropy keeps rewarding higher true-class probability; hinge stops after the margin.

---

## References

- PyTorch margin losses: https://docs.pytorch.org/docs/stable/nn.html#loss-functions
- scikit-learn SVM user guide: https://scikit-learn.org/stable/modules/svm.html
