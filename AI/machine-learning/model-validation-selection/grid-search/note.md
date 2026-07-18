# Grid Search - Interview Knowledge Sheet

## Intuition

Grid search tries every combination from a manually chosen hyperparameter grid.

It is simple and exhaustive inside that grid:

```text
for each parameter combination -> cross-validate -> keep best score
```

---

## 1. Search Space

If one parameter has `a` choices and another has `b` choices, grid search evaluates:

$$
a \times b
$$

combinations.

That grows quickly as you add parameters.

---

## 2. Correct Validation

Grid search should evaluate each combination using cross-validation or a validation set.

The final test set should not be used to choose hyperparameters. Once you optimize on validation scores, those validation scores become biased upward.

---

## 3. When It Works

Grid search is good when there are few important hyperparameters and reasonable candidate values.

It is weak when many parameters matter or when only a tiny part of the grid is useful.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
