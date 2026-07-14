# Linear Regression: Gradient Descent — Interview Knowledge Sheet

## Intuition

We learned about Linear Regression in [Linear Regression — Normal Equation](../linear-regression-normal-equation/note.md).

Now we'll find out how gradient descent is used in Linear Regression specifically and in ML/DL generally.

**The error (loss) hill**

The goal of ML is to minimize the error between predictions and real data. The error (loss) function is a hilly surface. Each point on it is one set of model parameters, and its height is how wrong the model is. You start at a random point (random parameters), somewhere on the hill — not the lowest.

**The gradient**

The gradient is like a compass. It is a vector pointing in the uphill direction, and its size represents the steepness at where you stand (for each model parameter). It tells you which way is up.

**The descent**

To reach the lowest point, go in the opposite direction of the gradient — this is the descent. Take a step downhill, feel the slope again, and repeat until you reach the bottom.

Different problems use different loss functions — that is, different hill shapes. For Linear Regression, we use Mean Squared Error (MSE). The reason for choosing MSE is explained in the next sections.

---

## 1. The Model

Linear regression predicts:

```
y_hat = Xw + b
```

The goal is to choose `w` and `b` so predictions are close to the true labels.

---

## 2. The Loss

The standard loss is mean squared error:

```
MSE = mean((y_hat - y)²)
```

Intuition:

- 

---

## 3. Gradient Descent Intuition

A gradient tells us which direction increases the loss.

To lower the loss, move the parameters in the opposite direction:

```
parameter = parameter - learning_rate * gradient
```

Think of walking downhill on a loss surface.

---

## 4. Vectorized NumPy Update

With a bias column:

```
pred = X_bias @ theta
error = pred - y
grad = (2 / n) * X_bias.T @ error
theta = theta - lr * grad
```

This updates all weights at once.

`theta[0]` is the bias.

`theta[1:]` are the feature weights.

---

## 5. Learning Rate

The learning rate controls step size.

Small learning rate:

- safer
- slower

Large learning rate:

- faster at first
- can overshoot or diverge

In interviews, mention that feature scaling usually helps gradient descent.

---

## 6. PyTorch Pattern

PyTorch can compute gradients automatically:

```python
model = torch.nn.Linear(num_features, 1)
loss = torch.nn.MSELoss()(model(X), y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

This is the same training idea used for neural networks.

---

## 7. When Use Gradient Descent?

Use gradient descent when:

- the dataset is large
- you want minibatch training
- the model has no simple closed-form solution
- you are using PyTorch or deep learning tools

For small ordinary linear regression, `lstsq` is often simpler.

---

## Interview Gotchas

- The gradient for MSE includes the factor `2 / n`.
- Use vectorized matrix operations, not nested Python loops.
- Feature scaling can make training faster and more stable.
- Too large a learning rate can make loss increase.
- In PyTorch, call `zero_grad()` so gradients do not accumulate across steps.
