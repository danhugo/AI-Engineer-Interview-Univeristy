# Logistic Regression Gradient Descent — Q&A

---

## Core Training Loop

**Q: How does gradient descent train logistic regression?**
A: It predicts probabilities, measures binary cross-entropy loss, computes gradients, and updates weights to reduce the loss.

**Q: Why not use the linear regression normal equation?**
A: Logistic regression optimizes binary cross-entropy through sigmoid, so there is no simple normal-equation solution like ordinary least squares.

**Q: What parameters does logistic regression learn?**
A: Feature weights `w` and a bias `b`.

---

## NumPy Gradients

**Q: What are the vectorized NumPy steps for one update?**
A: Compute `logits = X @ w + b`, `probs = sigmoid(logits)`, `error = probs - y`, then update with `dw = X.T @ error / n` and `db = mean(error)`.

**Q: Why is vectorization important?**
A: It is simpler, faster, and closer to how ML libraries operate internally.

**Q: What does `probs - y` represent?**
A: The prediction error used in the gradient.

---

## Logits vs Probabilities

**Q: What is a logit?**
A: The raw score before sigmoid.

**Q: What is a probability?**
A: The sigmoid of the logit, interpreted as `P(y=1|x)`.

**Q: Which value should be passed to PyTorch `BCEWithLogitsLoss`?**
A: Raw logits.

**Q: When do you threshold probabilities?**
A: During prediction or evaluation, not inside the training loss.

---

## PyTorch Autograd

**Q: What does autograd do?**
A: It computes gradients automatically from the loss.

**Q: Why call `optimizer.zero_grad()` each step?**
A: PyTorch accumulates gradients, so old gradients must be cleared before a new update.

**Q: What does `optimizer.step()` do?**
A: It updates model parameters using the computed gradients.

---

## Gotchas

**Q: What happens if the learning rate is too large?**
A: Training may overshoot and the loss may fail to decrease.

**Q: Why can feature scaling help?**
A: It makes the loss surface easier for gradient descent to navigate.

**Q: Does logistic regression learn a nonlinear boundary because of sigmoid?**
A: No. The decision boundary is still linear in the input features.

