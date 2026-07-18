# Train/Val/Test Split with No Leakage - Interview Knowledge Sheet

## Intuition

The purpose of a validation/test split is to simulate future unseen data.

Leakage happens when information from validation or test data influences training.

The safe order is:

```text
split first -> fit preprocessing on train -> tune on validation -> report once on test
```

---

## 1. Roles

Training set fits model parameters.

Validation set selects hyperparameters and modeling choices.

Test set estimates final generalization after decisions are frozen.

---

## 2. Common Leakage Examples

- scaling before splitting
- imputing using full-data statistics
- selecting features using all labels
- duplicate users across train and test
- random splits for time-dependent data

---

## 3. Pipelines

A pipeline helps because each preprocessing step is fit only on the training fold during cross-validation.
## References

- scikit-learn preprocessing API: https://scikit-learn.org/stable/api/sklearn.preprocessing.html
- scikit-learn OneHotEncoder docs: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- scikit-learn impute user guide: https://scikit-learn.org/stable/modules/impute.html
