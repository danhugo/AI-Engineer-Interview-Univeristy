# Elastic Net Gradient Descent — Q&A

---

## Intuition

**Q: What is Elastic Net?**
A: Linear regression with both L1 and L2 penalties.

**Q: What does it combine?**
A: Lasso-style sparsity and ridge-style stability.

**Q: Does Elastic Net change the prediction formula?**
A: No. Prediction is still `Xw + b`.

---

## 1. The Objective

**Q: What are the three parts of the Elastic Net loss?**
A: MSE, an L1 penalty, and an L2 penalty.

**Q: What does `lambda_1` control?**
A: The L1 penalty strength.

**Q: What does `lambda_2` control?**
A: The L2 penalty strength.

---

## 2. Why Combine L1 and L2?

**Q: What does L1 add?**
A: Feature selection through exact zero weights.

**Q: What does L2 add?**
A: Smooth shrinkage and stability.

**Q: Why is Elastic Net useful with correlated features?**
A: It can be more stable than pure lasso, which may choose only one feature from a correlated group.

---

## 3. Relation to Ridge and Lasso

**Q: When does Elastic Net become lasso?**
A: When `lambda_1 > 0` and `lambda_2 = 0`.

**Q: When does Elastic Net become ridge?**
A: When `lambda_1 = 0` and `lambda_2 > 0`.

**Q: When is it true Elastic Net?**
A: When both penalties are positive.

---

## 4. Gradients and Subgradients

**Q: What is the MSE gradient for weights?**
A: `(2 / n) * X.T @ error`.

**Q: What is the L2 gradient?**
A: `2 * lambda_2 * w`.

**Q: What does the L1 term use?**
A: A subgradient, usually `lambda_1 * sign(w)` in simple code.

---

## 5. Gradient Descent Update

**Q: What terms are in the weight gradient?**
A: MSE gradient, L1 subgradient, and L2 gradient.

**Q: What terms are in the bias gradient?**
A: Only the MSE bias gradient.

**Q: Why is the bias not regularized?**
A: It is the baseline prediction, not a feature strength.

---

## 6. Subgradient vs Proximal Update

**Q: Why is subgradient descent simple?**
A: It lets you write one gradient-like update for all terms.

**Q: What is the downside?**
A: It may not create exact zeros as cleanly as proximal methods.

**Q: What handles L1 penalties better?**
A: Proximal methods such as soft-thresholding.

---

## 7. Complexity

**Q: What is the cost of one gradient step?**
A: `O(nd)`.

**Q: What operations dominate the cost?**
A: `X @ w` and `X.T @ error`.

---

## 8. Interview Gotchas

**Q: Why does feature scaling matter?**
A: Regularization penalizes coefficient size.

**Q: What happens if the learning rate is too large?**
A: The loss can diverge.

**Q: What is the core Elastic Net summary?**
A: MSE plus L1 and L2 penalties on the weights.
