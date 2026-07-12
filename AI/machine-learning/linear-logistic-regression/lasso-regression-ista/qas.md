# Lasso Regression with ISTA — Q&A

---

## Core Idea

**Q: What is Lasso regression?**
A: Linear regression with an L1 penalty on the weights.

**Q: What is the Lasso objective?**
A: `mean((Xw + b - y)^2) + alpha * sum(abs(w))`.

**Q: Do we regularize the bias?**
A: No. The penalty is applied to the feature weights, not the bias.

---

## L1 Intuition

**Q: Why does Lasso create sparse models?**
A: The L1 penalty can push small weights exactly to zero.

**Q: What does a zero weight mean?**
A: The model is ignoring that feature.

**Q: How is Lasso different from Ridge?**
A: Ridge shrinks weights smoothly. Lasso can shrink some weights all the way to zero.

---

## Soft-Thresholding

**Q: What does soft-thresholding do?**
A: It moves values toward zero and sets small values exactly to zero.

**Q: What is the formula?**
A: `sign(x) * max(abs(x) - threshold, 0)`.

**Q: Why does ISTA need soft-thresholding?**
A: It handles the L1 part of the objective, which is not differentiable at zero.

---

## ISTA

**Q: What does ISTA stand for?**
A: Iterative Shrinkage-Thresholding Algorithm.

**Q: What are the two parts of one ISTA step?**
A: A gradient step on MSE, then soft-thresholding on the weights.

**Q: What update is used for the bias?**
A: A normal gradient descent update, because the bias is not L1-regularized.

---

## Implementation

**Q: What is the MSE gradient for weights?**
A: `(2 / n) * X.T @ (pred - y)`.

**Q: What is the MSE gradient for bias?**
A: `(2 / n) * sum(pred - y)`.

**Q: What happens when `alpha` increases?**
A: More weights are pushed toward zero.

**Q: Why should features be scaled before Lasso?**
A: Because the penalty acts directly on coefficient size, so feature scale affects which weights are punished most.

---

## Complexity

**Q: What is the cost of one ISTA step?**
A: O(nd) for `n` examples and `d` features.

