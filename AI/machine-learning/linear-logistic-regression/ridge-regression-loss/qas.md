# Ridge Regression Loss — Q&A

---

## Core Idea

**Q: What is Ridge regression?**
A: Linear regression with an L2 penalty on the weights.

**Q: What is the Ridge loss?**
A: `MSE + alpha * sum(w²)`.

**Q: Does Ridge change the prediction formula?**
A: No. Prediction is still `y_hat = Xw + b`.

---

## Regularization

**Q: Why add a weight penalty?**
A: To discourage very large weights and reduce overfitting.

**Q: What does `alpha` control?**
A: The strength of the regularization penalty.

**Q: What happens when `alpha = 0`?**
A: Ridge becomes ordinary linear regression.

**Q: What can happen when `alpha` is too large?**
A: The model may underfit because weights are shrunk too much.

---

## Bias

**Q: Should Ridge regularize the bias?**
A: Usually no.

**Q: Why not regularize the bias?**
A: The bias only shifts predictions up or down; penalizing it usually does not help control feature complexity.

**Q: If `theta = [b, w1, w2]`, which values are regularized?**
A: Only `theta[1:]`, not `theta[0]`.

---

## NumPy

**Q: What is the NumPy Ridge-loss pattern?**
A: Compute predictions, compute MSE, add `alpha * np.sum(w ** 2)`.

**Q: Why keep `w` and `b` separate?**
A: It makes it harder to accidentally regularize the bias.

---

## PyTorch

**Q: What PyTorch loss is used for the prediction part?**
A: `torch.nn.MSELoss`.

**Q: For `torch.nn.Linear`, what should be regularized?**
A: `model.weight`.

**Q: Should `model.bias` be included in the Ridge penalty?**
A: No.

**Q: Where do you add the Ridge penalty during training?**
A: Add it to the MSE loss before calling `loss.backward()`.

---

## Comparison

**Q: How is Ridge different from Lasso?**
A: Ridge uses squared weights and shrinks weights smoothly. Lasso uses absolute weights and can set weights exactly to zero.

**Q: When is Ridge useful?**
A: When features are correlated, noisy, or numerous and you want a more stable linear model.
