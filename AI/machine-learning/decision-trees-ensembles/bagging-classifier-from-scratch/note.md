# Bagging Classifier from Scratch - Interview Knowledge Sheet

## Intuition

Bagging trains many copies of the same high-variance learner on bootstrap samples, then averages their predictions.

For classification, averaging usually means majority vote.

---

## 1. Algorithm

For `B` estimators: sample rows with replacement, fit a base classifier, store it, and vote at prediction time.

$$
\hat{y} = \operatorname{mode}(h_1(x), \ldots, h_B(x))
$$

---

## 2. Bagging vs Random Forest

Bagging can use any base learner. A random forest is bagging of decision trees plus random feature selection at splits. Bagging mainly reduces variance, especially for unstable learners like deep trees.
## References

- scikit-learn Decision Trees user guide: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Ensemble Methods user guide: https://scikit-learn.org/stable/modules/ensemble.html
