# Bayesian Hyperparameter Optimization - Interview Knowledge Sheet

## Intuition

Bayesian optimization treats hyperparameter tuning as expensive black-box optimization.

Instead of trying points blindly, it builds a surrogate model of:

```text
hyperparameters -> validation score
```

Then it chooses the next trial by balancing exploration and exploitation.

---

## 1. Surrogate and Acquisition

The surrogate predicts both expected score and uncertainty.

The acquisition function chooses the next candidate. A common intuition is:

```text
try places that look good or places where uncertainty is high
```

---

## 2. Expected Improvement Intuition

Expected improvement asks how much a candidate is expected to beat the current best score.

A point can be attractive because it has a high predicted mean or high uncertainty.

---

## 3. When To Use

Use Bayesian optimization when each training run is expensive and the number of trials is limited.

Random search is often enough when training is cheap or the search space is simple.
## References

- scikit-learn model selection user guide: https://scikit-learn.org/stable/model_selection.html
- scikit-learn learning curve docs: https://scikit-learn.org/stable/modules/learning_curve.html
