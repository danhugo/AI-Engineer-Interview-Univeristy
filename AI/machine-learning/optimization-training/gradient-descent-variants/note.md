# Gradient Descent Variants — Interview Knowledge Sheet

## Intuition

Gradient descent means:

```
look at the slope -> step downhill -> repeat
```

The variants mainly differ in **how much data** they use to estimate that slope.

- Batch gradient descent uses the whole dataset.
- Stochastic gradient descent uses one sample at a time.
- Mini-batch gradient descent uses a small batch, such as 32 or 256 samples.

The update is always the same shape:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

where `eta` is the learning rate.

---

## 1. Batch Gradient Descent

Batch gradient descent computes the gradient over all `n` training examples before every update.

For mean squared error:

$$
L(\theta) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

One update:

```python
pred = X @ theta
grad = (2 / n) * X.T @ (pred - y)
theta = theta - lr * grad
```

### Tradeoff

The gradient is stable because it uses all data.

But each step can be expensive for large datasets.

---

## 2. Stochastic Gradient Descent

Stochastic gradient descent uses one example per update.

For example `i`:

$$
g_i = \nabla_\theta L_i(\theta)
$$

Then:

$$
\theta \leftarrow \theta - \eta g_i
$$

### Tradeoff

Each step is cheap and noisy.

The noise can help escape shallow bad regions, but the loss curve jumps around.

---

## 3. Mini-Batch Gradient Descent

Mini-batch gradient descent uses a small group of examples per update.

This is the default in deep learning.

It balances:

- cheaper updates than full-batch
- less noisy gradients than one-sample SGD
- efficient matrix operations on CPU/GPU

Common training code:

```python
for xb, yb in loader:
    optimizer.zero_grad()
    loss = model_loss(xb, yb)
    loss.backward()
    optimizer.step()
```

---

## 4. Epochs and Shuffling

An **epoch** means one pass through the training dataset.

For SGD and mini-batch training, shuffle examples each epoch.

Shuffling prevents the optimizer from seeing the same ordered pattern every pass.

---

## 5. Learning Rate

The learning rate controls update size.

- Too small: training is slow.
- Too large: loss may bounce or diverge.
- Smaller batches often need more careful learning-rate tuning because gradients are noisier.

---

## 6. Interview Gotchas

- "SGD" is often used loosely to mean mini-batch SGD in deep learning.
- Batch size changes gradient noise, memory use, and update frequency.
- Full-batch gradients are deterministic for a fixed dataset and parameters.
- Mini-batch gradients are estimates of the full gradient.
- Always clear old PyTorch gradients with `optimizer.zero_grad()`.

---

## References

- scikit-learn Stochastic Gradient Descent user guide: https://scikit-learn.org/stable/modules/sgd.html
- PyTorch `DataLoader`: https://docs.pytorch.org/docs/stable/data.html
- PyTorch optimization recipe: https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
