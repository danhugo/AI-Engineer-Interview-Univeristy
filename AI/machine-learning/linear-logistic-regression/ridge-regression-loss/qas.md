# Ridge Regression Loss — Q&A

---

## Intuition

**Q: What is ridge regression?**
A: Linear regression with an L2 penalty on the weights.

**Q: Does ridge change the prediction formula?**
A: No. Prediction is still `Xw + b`.

**Q: What changes in ridge?**
A: The loss includes a penalty for large weights.

---

## 1. Why Regularize?

**Q: When can ordinary linear regression become unstable?**
A: When features are correlated, numerous, or the data is small or noisy.

**Q: Why are very large weights risky?**
A: They make predictions sensitive to small input changes.

**Q: What does ridge prefer?**
A: Smaller, more stable weights.

---

## 2. The Ridge Objective

**Q: What two parts are in the ridge loss?**
A: Prediction error and an L2 weight penalty.

**Q: What does `alpha` control?**
A: The strength of the regularization penalty.

**Q: What happens when `alpha = 0`?**
A: Ridge becomes ordinary linear regression.

---

## 3. What L2 Does

**Q: Why does L2 discourage large weights?**
A: Squaring makes large weights expensive.

**Q: Does ridge usually make weights exactly zero?**
A: No. It usually shrinks weights smoothly toward zero.

**Q: What model can create exact zero weights?**
A: Lasso.

---

## 4. Bias Term

**Q: Should ridge regularize the bias?**
A: Usually no.

**Q: Why not regularize the bias?**
A: The bias is the baseline prediction, not a feature strength.

**Q: If `theta = [b, w1, w2]`, what should be penalized?**
A: Only `theta[1:]`.

---

## 5. Gradient

**Q: What is added to the weight gradient by ridge?**
A: `2 * alpha * w`.

**Q: Does the bias gradient get a ridge term?**
A: No.

**Q: What is the MSE weight gradient?**
A: `(2 / n) * X.T @ error`.

---

## 6. NumPy Pattern

**Q: How do you compute ridge loss in NumPy?**
A: Compute MSE, then add `alpha * np.sum(w ** 2)`.

**Q: How do you avoid regularizing the bias?**
A: Keep `w` and `b` separate or exclude `theta[0]`.

---

## 7. PyTorch Pattern

**Q: What should be regularized in `torch.nn.Linear`?**
A: `model.weight`.

**Q: Should `model.bias` be included?**
A: No.

**Q: When do you add the ridge penalty?**
A: Add it to the loss before `backward()`.

---

## 8. Ridge vs Lasso

**Q: What penalty does ridge use?**
A: L2, `sum(w^2)`.

**Q: What penalty does lasso use?**
A: L1, `sum(abs(w))`.

**Q: When is ridge a good choice?**
A: When you want stable weights and do not need exact feature selection.

---

## 9. Interview Gotchas

**Q: Why does feature scaling matter?**
A: The penalty acts on coefficient size, so scale affects the penalty.

**Q: What happens if `alpha` is too large?**
A: The model can underfit.

**Q: What is the core ridge summary?**
A: Same linear model, MSE plus L2 weight penalty.
