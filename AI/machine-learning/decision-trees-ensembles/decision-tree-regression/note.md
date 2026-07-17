# Decision Tree Regression - Interview Knowledge Sheet

## Intuition

A regression tree partitions feature space into regions and predicts a constant value in each region. The constant is usually the mean target value of training examples in that leaf.

---

## 1. Split Criterion

Regression trees commonly choose splits that reduce squared error. For targets in a node:

$$
SSE = \sum_i (y_i - \bar{y})^2
$$

A split is useful when:

$$
\Delta = SSE_{parent} - (SSE_L + SSE_R)
$$

---

## 2. Leaf Values

The mean minimizes squared error inside the leaf:

$$
\hat{y}_{leaf} = \frac{1}{n_{leaf}}\sum_{i \in leaf} y_i
$$

This makes regression trees piecewise-constant models. Shallow trees may underfit; deep trees can memorize noise.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
