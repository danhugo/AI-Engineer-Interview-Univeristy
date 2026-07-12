# Elastic Net Gradient Descent — Q&A

---

## Core Idea

**Q: What is Elastic Net?**
A: Linear regression with both L1 and L2 regularization.

**Q: What is the objective?**
A: `MSE + l1 * sum(abs(w)) + l2 * sum(w^2)`.

**Q: Do we regularize the bias?**
A: No. Only the feature weights are regularized.

---

## Intuition

**Q: What does the L1 part do?**
A: It can push some weights exactly to zero.

**Q: What does the L2 part do?**
A: It discourages large weights smoothly.

**Q: Why use Elastic Net instead of only Lasso?**
A: Elastic Net is often more stable when features are correlated.

---

## Gradients

**Q: What is the MSE gradient for weights?**
A: `(2 / n) * X.T @ (pred - y)`.

**Q: What is the MSE gradient for bias?**
A: `(2 / n) * sum(pred - y)`.

**Q: What is the L2 gradient if the penalty is `l2 * sum(w^2)`?**
A: `2 * l2 * w`.

**Q: What do we use for the L1 subgradient in simple code?**
A: `l1 * sign(w)`.

---

## Training

**Q: What is the full weight gradient?**
A: `mse_grad_w + l1 * sign(w) + 2 * l2 * w`.

**Q: What is the full bias gradient?**
A: Only the MSE bias gradient, because the bias is not regularized.

**Q: What can happen if the learning rate is too large?**
A: The loss can bounce around or diverge.

---

## Gotchas

**Q: Why should features be scaled before Elastic Net?**
A: Regularization penalizes coefficient size, so unscaled features can be penalized unfairly.

**Q: Does basic subgradient descent create exact zeros as reliably as ISTA?**
A: No. Proximal methods are usually better for exact sparsity, but subgradient descent is simple and interview-friendly.

**Q: What is the cost of one gradient step?**
A: O(nd), where `n` is the number of examples and `d` is the number of features.

