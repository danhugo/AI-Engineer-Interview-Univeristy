# AdaBoost Fit Method - Interview Knowledge Sheet

## Intuition

AdaBoost trains weak learners one after another. Each round increases weight on examples the current ensemble gets wrong, so the next weak learner focuses on hard cases.

---

## 1. Learner Weight

Classic binary AdaBoost uses labels in `{ -1, +1 }`. Weighted error is:

$$
\epsilon_t = \sum_i w_i \mathbf{1}[h_t(x_i) \ne y_i]
$$

The learner weight is:

$$
\alpha_t = \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}
$$

---

## 2. Sample Weight Update

$$
w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))
$$

Correct examples shrink. Wrong examples grow. Then weights are normalized. The final prediction is the sign of weighted learner votes.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
