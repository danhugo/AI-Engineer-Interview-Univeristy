# Logistic Regression Gradient Descent — Interview Knowledge Sheet

## Intuition

Logistic regression is a **classification** model. It predicts the probability that a sample belongs to class 1.

Linear regression can't do this. It outputs any real number, not a bounded probability. So we take a linear score and squash it into `[0, 1]`.

The model:

```
logits = Xw + b
probs  = sigmoid(logits)
```

Training means finding `w` and `b` by **gradient descent**.

### Logit vs logistic — they are inverses

- **Logit**: maps probability → real number (unbounded score)
- **Logistic** (sigmoid): maps real number → probability in `[0, 1]`

We compute a logit, then turn it into a probability with sigmoid:

$$
p = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}
$$

| Logit | Probability |
|-------|------------|
| large negative | near 0 |
| 0 | 0.5 |
| large positive | near 1 |

A probability is not a label until you pick a threshold (usually 0.5): `p ≥ 0.5 → class 1`, else class 0.

### Why "logistic" / "regression"

"Logistic regression" is a **classification** method, despite the word "regression." The "regression" part refers to the underlying linear model (it regresses the log-odds onto features). The "logistic" part is the sigmoid link function.

---

## 1. Why Gradient Descent?

Logistic regression has **no closed-form solution**.

In linear regression, the normal equation solves least squares in one shot. That works because the loss is quadratic in the weights. Here the loss goes through the nonlinear sigmoid, so the math does not simplify to an exact formula. We must search for the weights iteratively with gradient descent.

---

## 2. Binary Cross-Entropy Loss

### Why not MSE?

Linear regression uses MSE, which comes from MLE assuming Gaussian errors. But our labels are 0 or 1, not continuous. A Gaussian assumption does not fit a coin-flip target.

### BCE is the MLE loss for a Bernoulli target

Each label is a Bernoulli trial: value 1 with probability `p`, value 0 with probability `1-p`. Its probability mass function:

$$
P(Y = y) = p^y(1-p)^{1-y}
$$

Step 1 — **Likelihood** (product over all samples, since labels are independent):

$$
L = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}
$$

Step 2 — **Log-likelihood** (turn products into sums, avoid underflow from multiplying many small numbers):

$$
\log L = \sum_{i=1}^{n} \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

Step 3 — **Flip to minimization** (gradient descent minimizes, so multiply by -1):

$$
\text{BCE} = -\sum_{i=1}^{n} \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

That is binary cross-entropy.

### Intuition

The loss is always `-log(probability assigned to the correct class)`:

- True label 1 → loss = `-log(p)`. Want `p` near 1.
- True label 0 → loss = `-log(1-p)`. Want `p` near 0.

Confident right predictions are cheap. Confident wrong predictions blow up toward infinity.

---

## 3. The Clean Gradient

### Why sigmoid pairs so well with BCE

Sigmoid has a tidy derivative:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

When we differentiate BCE w.r.t. the logit `z`, chain rule gives:

$$
\frac{dL}{dz}
= \frac{dL}{dp} \cdot \frac{dp}{dz}
= \frac{p-y}{p(1-p)} \cdot p(1-p)
= p-y
$$

The `p(1-p)` denominator from BCE cancels exactly with the `p(1-p)` from the sigmoid derivative. The result is wonderfully clean:

$$
\frac{dL}{dz} = p-y
$$

So the prediction error `p - y` is the gradient w.r.t. each logit.

### Vectorized update (one full batch)

With `n` samples:

```
logits = X @ w + b
probs  = sigmoid(logits)
error  = probs - y          # gradient w.r.t. logits
dw = X.T @ error / n        # gradient w.r.t. weights
db = mean(error)            # gradient w.r.t. bias

w -= lr * dw
b -= lr * db
```

This mirrors the linear regression gradient, with `error = probs - y` instead of `error = y_hat - y`. The only difference is that predictions come through sigmoid.

---

## 4. Logits vs Probabilities

Keep these distinct:

| Value | Meaning | Used for |
|-------|---------|----------|
| logits | raw scores | stable loss functions |
| probabilities | sigmoid(logits) | interpretation and thresholding |
| classes | thresholded probabilities | final predictions |

In NumPy, you compute probabilities yourself for BCE.

In PyTorch, pass raw logits to `BCEWithLogitsLoss`. It fuses sigmoid + BCE in a numerically stable way.

---

## 5. PyTorch Autograd Training

PyTorch computes gradients for you:

```
logits = model(X)
loss  = BCEWithLogitsLoss()(logits, y)
loss.backward()
optimizer.step()
```

The correct order each step:

1. `optimizer.zero_grad()` — clear old gradients (PyTorch accumulates by default)
2. forward pass
3. compute loss
4. `loss.backward()` — fill in gradients
5. `optimizer.step()` — apply the update

`zero_grad()` matters: without it, gradients from previous steps pile up and updates drift.

---

## 6. Learning Rate

The learning rate controls update size.

- Too small: training is slow.
- Too large: loss bounces or diverges.

For interview examples, use simple scaled data and a modest learning rate.

---

## 7. Interview Gotchas

- Use vectorized matrix operations in NumPy.
- Do not apply sigmoid before `BCEWithLogitsLoss` (it expects logits).
- Convert probabilities to classes only after training or during evaluation.
- Feature scaling often makes gradient descent easier.
- Logistic regression learns a **linear** decision boundary in the input features, even though sigmoid makes the outputs nonlinear.
