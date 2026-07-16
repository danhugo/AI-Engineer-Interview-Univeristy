# Lasso Regression with ISTA — Q&A

---

## Intuition

**Q: What is lasso regression?**
A: Linear regression with an L1 penalty on the weights.

**Q: What can lasso do that ridge usually does not?**
A: It can set some weights exactly to zero.

**Q: What are the two moves in ISTA?**
A: A gradient step, then soft-thresholding.

---

## 1. The Lasso Objective

**Q: Does lasso change the prediction formula?**
A: No. Prediction is still `Xw + b`.

**Q: What penalty does lasso use?**
A: The L1 norm, `sum(abs(w))`.

**Q: Should lasso regularize the bias?**
A: No. Regularize `w`, not `b`.

---

## 2. Why L1 Creates Sparsity

**Q: Why does L1 create sparse weights?**
A: It can pull small weights all the way to exactly zero.

**Q: What does a zero weight mean?**
A: The model ignores that feature.

**Q: How is lasso different from ridge?**
A: Ridge shrinks smoothly. Lasso can create exact zeros.

---

## 3. Why ISTA?

**Q: Why is plain gradient descent awkward for lasso?**
A: L1 is not differentiable at zero.

**Q: What does ISTA stand for?**
A: Iterative Shrinkage-Thresholding Algorithm.

**Q: What part of lasso does soft-thresholding handle?**
A: The L1 penalty.

---

## 4. Soft-Thresholding

**Q: What does soft-thresholding do?**
A: It shrinks values toward zero and sets small values to zero.

**Q: What happens to `0.2` with threshold `0.5`?**
A: It becomes `0`.

**Q: Why is soft-thresholding useful?**
A: It creates exact zeros naturally.

---

## 5. ISTA Update

**Q: What is the MSE gradient for weights?**
A: `(2 / n) * X.T @ error`.

**Q: How is `w` updated in ISTA?**
A: Take a gradient step, then apply soft-thresholding.

**Q: How is `b` updated?**
A: With a normal gradient step.

---

## 6. NumPy and PyTorch Pattern

**Q: What is the NumPy soft-threshold pattern?**
A: `np.sign(x) * np.maximum(np.abs(x) - t, 0)`.

**Q: What is the PyTorch soft-threshold pattern?**
A: `torch.sign(x) * torch.clamp(torch.abs(x) - t, min=0)`.

**Q: Why use manual updates here?**
A: They show the lasso/ISTA idea clearly.

---

## 7. Complexity

**Q: What is the cost of one ISTA step?**
A: `O(nd)`.

**Q: What operations dominate the cost?**
A: `X @ w` and `X.T @ error`.

---

## 8. Interview Gotchas

**Q: Why does feature scaling matter?**
A: L1 penalizes coefficient size, so scale affects the penalty.

**Q: What happens when `alpha` increases?**
A: More weights are pushed toward zero.

**Q: What is the core lasso summary?**
A: Linear regression plus L1 penalty, trained here with ISTA.
