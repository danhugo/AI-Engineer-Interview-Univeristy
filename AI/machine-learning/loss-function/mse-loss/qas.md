# MSE Loss — Q&A

---

## Intuition

**Q: What does MSE measure?**
A: The average squared distance between predictions and true values.

**Q: Why does MSE punish large errors strongly?**
A: It squares the error.

**Q: What task type is MSE for?**
A: Regression.

---

## 1. Formula

**Q: What is the MSE formula?**
A: `mean((y_hat - y) ** 2)`.

**Q: What is the best possible MSE?**
A: `0`.

**Q: What does one sample contribute?**
A: Its squared error.

---

## 2. Why Square the Error?

**Q: What does squaring do to error signs?**
A: It makes negative and positive errors both positive.

**Q: What is the downside of squaring?**
A: Outliers can dominate the loss.

**Q: Why is MSE convenient for gradient descent?**
A: It is smooth and has a simple gradient.

---

## 3. Gradient

**Q: What is the gradient of one squared error with respect to prediction?**
A: `2 * (y_hat - y)`.

**Q: What factor appears for mean MSE over `n` samples?**
A: `2 / n`.

**Q: What does the gradient point toward?**
A: Reducing prediction error.

---

## 4. When to Use

**Q: When is MSE a good choice?**
A: When large regression errors should be punished strongly.

**Q: When is MSE risky?**
A: When outliers are noisy and should not dominate.

**Q: What losses can be more robust?**
A: MAE or Huber.

---

## 5. PyTorch Pattern

**Q: What PyTorch class implements MSE?**
A: `torch.nn.MSELoss`.

**Q: What reduction is common by default?**
A: Mean.

---

## 6. Interview Gotchas

**Q: Is MSE for classification?**
A: Usually no.

**Q: Why mention mean vs sum?**
A: They scale the loss and gradient differently.

**Q: Why is RMSE often reported?**
A: It has the same units as the target.
