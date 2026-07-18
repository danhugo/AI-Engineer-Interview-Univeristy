# Random Search vs Grid Search - Interview Knowledge Sheet

## Intuition

Grid search spends trials evenly over every dimension.

Random search spends trials across the whole space. This is often better when only a few hyperparameters strongly affect performance.

If only learning rate matters, a grid may waste most trials varying less important parameters.

---

## 1. Trial Budget

Grid search budget is fixed by the grid size:

$$
N_{grid} = \prod_j m_j
$$

Random search lets you choose any budget `N`.

This makes random search easier when training is expensive.

---

## 2. Distributions Matter

Random search is only as good as the sampling distributions.

For scale-sensitive parameters such as learning rate, sample log-uniform rather than uniform.

---

## 3. Practical Rule

Use grid search for small, known search spaces.

Use random search for larger spaces or when you want a fixed compute budget.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
