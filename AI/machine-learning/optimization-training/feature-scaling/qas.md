# Feature Scaling — Q&A

---

## Intuition

**Q: What is feature scaling?**
A: Transforming input columns so their numeric scales are comparable.

**Q: Why can unscaled features be a problem?**
A: Large-scale features can dominate distances, gradients, or regularization.

**Q: Does scaling change labels?**
A: No. It changes input features.

---

## 1. Standardization

**Q: What is the standardization formula?**
A: `(x - mean) / std`.

**Q: What mean and standard deviation should validation data use?**
A: The training set mean and standard deviation.

**Q: What does standardized training data usually have?**
A: Mean near `0` and standard deviation near `1` per feature.

---

## 2. Min-Max Scaling

**Q: What does min-max scaling to `[0, 1]` compute?**
A: `(x - min) / (max - min)`.

**Q: Does min-max scaling preserve order?**
A: Yes.

**Q: What is a downside of min-max scaling?**
A: It is sensitive to outliers.

---

## 3. Optimization

**Q: Why does scaling help gradient descent?**
A: It makes feature directions more balanced, so one learning rate works better.

**Q: What can happen without scaling?**
A: Slow convergence or unstable updates.

**Q: Why does scaling matter for regularization?**
A: Penalties act on coefficient sizes, which depend on feature scale.

---

## 4. Model Types

**Q: Which models often need scaling?**
A: KNN, k-means, SVMs, neural networks, and gradient descent linear models.

**Q: Which models usually care less?**
A: Tree-based models.

**Q: Why do trees care less?**
A: Their splits depend mainly on ordering, not Euclidean distance or gradient scale.

---

## 5. Leakage

**Q: When should a scaler be fit?**
A: After the train/test split, using training data only.

**Q: Why not fit on the full dataset?**
A: It leaks test-set information.

**Q: What should be saved from training?**
A: The scaler parameters such as means, standard deviations, mins, and maxes.

---

## 6. Edge Cases

**Q: What if a feature has zero standard deviation?**
A: Use a safe scale like `1` to avoid division by zero.

**Q: What scaler is often better with strong outliers?**
A: Robust scaling.

**Q: Should categorical one-hot columns always be standardized?**
A: Not always; it depends on model and pipeline design.
