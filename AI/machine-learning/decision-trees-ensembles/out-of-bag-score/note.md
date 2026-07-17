# Out-of-Bag Score - Interview Knowledge Sheet

## Intuition

Each bootstrap sample leaves out some training rows. Those left-out rows are out-of-bag for that tree and can act like validation data.

For each training example, collect predictions only from trees that did not train on that example.

---

## 1. Why Rows Are Left Out

A bootstrap sample draws `n` times with replacement from `n` rows. The probability a row is never selected is approximately:

$$
\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368
$$

So each tree leaves out about 36.8% of rows.

---

## 2. OOB Prediction

$$
\hat{y}_{OOB,i} = \operatorname{vote}\{T_b(x_i): i \notin bootstrap_b\}
$$

OOB is a cheap validation signal for bagging and random forests, but not a replacement for a final test set.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
