# Huber Loss — Q&A

---

## Intuition

**Q: What does Huber loss combine?**
A: MSE behavior for small errors and MAE behavior for large errors.

**Q: Why use Huber instead of MSE?**
A: It is less sensitive to outliers.

**Q: Why use Huber instead of MAE?**
A: It is smooth near zero.

---

## 1. Formula

**Q: What variable does Huber use?**
A: The prediction error `y_hat - y`.

**Q: What happens when the error is small?**
A: Huber uses a squared penalty.

**Q: What happens when the error is large?**
A: Huber uses a linear penalty.

---

## 2. Why It Helps

**Q: What is the MSE weakness?**
A: Outliers can dominate.

**Q: What is the MAE weakness?**
A: It has a sharp corner at zero.

**Q: What is Huber's compromise?**
A: Smooth near zero, robust for large errors.

---

## 3. Delta

**Q: What does `delta` control?**
A: The switch point between squared and linear behavior.

**Q: What does small `delta` do?**
A: Makes the loss more MAE-like.

**Q: What does large `delta` do?**
A: Makes the loss more MSE-like.

---

## 4. PyTorch Pattern

**Q: What PyTorch class implements Huber?**
A: `torch.nn.HuberLoss`.

**Q: What argument controls the switch?**
A: `delta`.

---

## 5. Interview Gotchas

**Q: What task type is Huber for?**
A: Regression.

**Q: When is Huber useful?**
A: When outliers exist but stable gradients still matter.
