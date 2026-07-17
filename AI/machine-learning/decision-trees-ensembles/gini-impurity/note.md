# Gini Impurity - Interview Knowledge Sheet

## Intuition

Gini impurity asks: if a random sample from this node were labeled according to the node class mixture, how often would the label be wrong?

A pure node has zero impurity. A mixed node has higher impurity.

For class probabilities $p_k$:

$$
G = 1 - \sum_k p_k^2
$$

The squared term rewards concentration. If one class dominates, $\sum p_k^2$ is large, so impurity is small.

---

## 1. Split Quality

A tree chooses splits by comparing weighted child impurity:

$$
G_{split} = \frac{n_L}{n}G_L + \frac{n_R}{n}G_R
$$

$$
\text{gain} = G_{parent} - G_{split}
$$

A useful split makes children more class-pure than the parent.

---

## 2. Interview Notes

Gini is a local split criterion, not the final model loss. It is popular because it is fast, simple, and usually ranks splits similarly to entropy.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
