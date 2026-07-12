# Linear Regression: Gradient Descent — Q&A

---

## Core Idea

**Q: What does gradient descent do?**
A: It repeatedly updates parameters in the direction that lowers the loss.

**Q: What is the linear-regression prediction formula?**
A: `y_hat = Xw + b`.

**Q: What loss does linear regression usually optimize?**
A: Mean squared error: `mean((y_hat - y)²)`.

---

## Gradients

**Q: What does a gradient tell you?**
A: The direction that increases the loss fastest.

**Q: Why subtract the gradient?**
A: Because subtracting moves parameters toward lower loss.

**Q: What is the vectorized MSE gradient with a bias column?**
A: `grad = (2 / n) * X_bias.T @ (pred - y)`.

**Q: Why use vectorized operations?**
A: They are shorter, faster, and closer to how ML libraries work.

---

## Learning Rate

**Q: What does the learning rate control?**
A: The size of each update step.

**Q: What happens if the learning rate is too small?**
A: Training can be very slow.

**Q: What happens if the learning rate is too large?**
A: Training can overshoot, bounce around, or diverge.

**Q: Why can feature scaling help?**
A: It makes the loss surface easier for gradient descent to move across.

---

## NumPy

**Q: Where is the bias stored when using a bias column?**
A: `theta[0]`.

**Q: Where are the weights stored?**
A: `theta[1:]`.

**Q: What are the basic NumPy training steps?**
A: Predict, compute error, compute gradient, subtract `lr * grad`.

---

## PyTorch

**Q: How does PyTorch compute gradients?**
A: Autograd tracks operations and computes gradients when you call `loss.backward()`.

**Q: What PyTorch layer represents linear regression?**
A: `torch.nn.Linear`.

**Q: What PyTorch loss should you use?**
A: `torch.nn.MSELoss`.

**Q: Why call `optimizer.zero_grad()`?**
A: PyTorch accumulates gradients by default, so old gradients must be cleared each step.

---

## Comparison

**Q: Why use gradient descent if least squares has a direct solver?**
A: Gradient descent scales to large data and works for many models that have no direct solution.

**Q: When is a direct least-squares solver simpler?**
A: For small ordinary linear-regression problems that fit easily in memory.
