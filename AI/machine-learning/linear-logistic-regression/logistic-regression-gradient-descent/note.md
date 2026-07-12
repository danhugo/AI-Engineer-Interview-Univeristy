# Logistic Regression Gradient Descent — Interview Knowledge Sheet

## Intuition

**Gradient descent trains logistic regression by nudging weights so predicted probabilities get closer to binary labels.**

The model is simple:

```
logits = Xw + b
probs = sigmoid(logits)
```

Training means finding `w` and `b`.

---

## 1. Why Gradient Descent?

Logistic regression does not use the same closed-form normal equation as ordinary linear regression.

Instead, we repeatedly:

1. Compute logits.
2. Convert logits to probabilities.
3. Measure binary cross-entropy loss.
4. Compute gradients.
5. Update weights in the opposite direction of the gradients.

```
w = w - learning_rate * dw
b = b - learning_rate * db
```

---

## 2. Vectorized NumPy Training

For a batch of `n` examples:

```
logits = X @ w + b
probs = sigmoid(logits)
error = probs - y
dw = X.T @ error / n
db = mean(error)
```

This is vectorized because it works on the whole matrix at once. No per-example Python loop is needed.

Interview point: the gradient has a clean form because sigmoid and binary cross-entropy combine nicely.

---

## 3. Logits vs Probabilities

Keep the difference clear:

| Value | Meaning | Used for |
|-------|---------|----------|
| logits | raw scores | stable loss functions |
| probabilities | sigmoid(logits) | interpretation and thresholding |
| classes | thresholded probabilities | final predictions |

In NumPy, you often compute probabilities yourself for binary cross-entropy.

In PyTorch, use `BCEWithLogitsLoss` with raw logits.

---

## 4. PyTorch Autograd Training

PyTorch can compute gradients for you:

```
logits = model(X)
loss = BCEWithLogitsLoss()(logits, y)
loss.backward()
optimizer.step()
```

The important order:

1. `optimizer.zero_grad()`
2. forward pass
3. loss
4. `loss.backward()`
5. `optimizer.step()`

`zero_grad()` matters because PyTorch accumulates gradients by default.

---

## 5. Learning Rate

The learning rate controls update size.

- Too small: training is slow.
- Too large: loss can bounce around or diverge.

For interview examples, use simple scaled data and a modest learning rate.

---

## 6. Interview Gotchas

- Use vectorized matrix operations in NumPy.
- Do not apply sigmoid before `BCEWithLogitsLoss`.
- Convert probabilities to classes only after training or during evaluation.
- Feature scaling often makes gradient descent easier.
- Logistic regression learns a linear decision boundary, even though sigmoid makes outputs nonlinear.

