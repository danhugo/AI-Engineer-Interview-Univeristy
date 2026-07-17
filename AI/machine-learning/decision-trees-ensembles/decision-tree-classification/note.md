# Decision Tree Classification - Interview Knowledge Sheet

## Intuition

A classification tree is a sequence of if/else questions. Each internal node asks a feature-threshold question. Each leaf stores the class mixture of training samples that reached it.

Prediction is: follow the questions, arrive at a leaf, then return the majority class or class probabilities.

---

## 1. Greedy Tree Growth

At each node, the tree chooses the candidate split with the largest impurity reduction:

$$
\Delta = I(parent) - \left(\frac{n_L}{n}I_L + \frac{n_R}{n}I_R\right)
$$

This is greedy: the best current split is not guaranteed to create the globally best whole tree.

---

## 2. Leaf Prediction

For class `k` in a leaf:

$$
P(y=k \mid x \in leaf) = \frac{\#\{y_i=k \text{ in leaf}\}}{\#\{i \text{ in leaf}\}}
$$

---

## 3. Regularization

Common controls are `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, and pruning with `ccp_alpha`. Deep trees reduce bias but can sharply increase variance.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
