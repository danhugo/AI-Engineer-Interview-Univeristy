# Softmax Regression Gradient Descent — Interview Knowledge Sheet

## Intuition

Softmax regression is **logistic regression for more than two classes**.

Binary logistic regression makes one logit and turns it into one probability:

```
logit -> sigmoid -> probability of class 1
```

Softmax regression makes one logit per class and turns them into a probability distribution:

```
logits -> softmax -> probabilities over all classes
```

The model:

```
logits = XW + b
probs  = softmax(logits)
```

Training means finding `W` and `b` by **gradient descent**.

---

## 1. Logits Are Class Scores

For `K` classes, each sample gets `K` raw scores.

For `3` classes:

```
logits = [2.0, 0.5, -1.0]
```

These are not probabilities. They do not need to be positive. They do not need to sum to `1`.

They only say how strongly the model likes each class before normalization.

---

## 2. Softmax Turns Scores Into Probabilities

Softmax exponentiates each logit and divides by the total:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Where:

- `z_k` is the logit for class `k`
- `p_k` is the probability for class `k`
- `K` is the number of classes

All probabilities sum to `1`:

$$
\sum_{k=1}^{K} p_k = 1
$$

The largest logit gets the largest probability.

Prediction is:

```
class_id = argmax(probs)
```

---

## 3. Stable Softmax

Naive softmax can overflow when logits are large.

Use the stable version:

```
shifted = logits - max(logits)
probs = exp(shifted) / sum(exp(shifted))
```

Subtracting the same value from every logit does not change the softmax result:

$$
\frac{e^{z_k-c}}{\sum_{j=1}^{K} e^{z_j-c}}
=
\frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

In a batch, subtract the row max for each sample.

---

## 4. Categorical Distribution Assumption

Binary logistic regression assumes each label is a Bernoulli trial.

Softmax regression assumes each label is a **categorical** trial:

- there are `K` possible classes
- exactly one class is correct
- the model predicts probabilities `p_1, p_2, ..., p_K`

For one sample with class ID `y`, the probability of the observed label is:

$$
P(Y = y) = p_y
$$

With a one-hot label vector, the same probability is:

$$
P(Y = y) = \prod_{k=1}^{K} p_k^{y_k}
$$

Only the true class contributes because its `y_k` is `1`. All other exponents are `0`.

For all samples, the likelihood is:

$$
L = \prod_{i=1}^{n} \prod_{k=1}^{K} p_{ik}^{y_{ik}}
$$

The log-likelihood is:

$$
\log L = \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik}\log(p_{ik})
$$

Gradient descent minimizes, so we use the negative log-likelihood. That gives cross-entropy.

---

## 5. Cross-Entropy Loss

For one sample, cross-entropy is:

$$
L = -\log(p_y)
$$

Where `y` is the true class.

So the loss is the negative log probability assigned to the correct class.

For one-hot labels, the same loss is:

$$
L = -\sum_{k=1}^{K} y_k \log(p_k)
$$

Only the true class contributes because only one `y_k` is `1`.

### Intuition

- Correct class gets high probability → low loss
- Correct class gets low probability → high loss

This is the multi-class version of binary cross-entropy.

---

## 6. The Clean Gradient

Softmax and cross-entropy also pair cleanly.

For each class logit:

$$
\frac{\partial L}{\partial z_k} = p_k - y_k
$$

So the gradient is:

```
predicted probability - true one-hot label
```

This matches binary logistic regression.

Binary case:

$$
\frac{dL}{dz} = p-y
$$

Multi-class case:

$$
\frac{\partial L}{\partial z} = p-y
$$

Same idea. More classes.

---

## 7. Vectorized Update

With:

- `X` shape `(n, d)`
- `W` shape `(d, K)`
- `b` shape `(K,)`
- `Y` shape `(n, K)` as one-hot labels

One full-batch update is:

```
logits = X @ W + b
probs = softmax(logits)
error = probs - Y

dW = X.T @ error / n
db = mean(error, axis=0)

W -= lr * dW
b -= lr * db
```

This mirrors logistic regression. The only difference is that `sigmoid` becomes `softmax`, and one binary label becomes one row of one-hot labels.

---

## 8. One-Hot Labels vs Class IDs

Two common label formats:

| Format | Example | Used by |
|--------|---------|---------|
| class IDs | `[0, 2, 1]` | PyTorch `CrossEntropyLoss` |
| one-hot | `[[1,0,0], [0,0,1], [0,1,0]]` | manual NumPy formulas |

In manual NumPy code, one-hot labels make the gradient easy:

$$
\text{error} = \text{probs} - Y
$$

In PyTorch, use class IDs.

---

## 9. PyTorch Autograd Training

PyTorch computes gradients for you:

```
logits = model(X)
loss = CrossEntropyLoss()(logits, y_class_ids)
loss.backward()
optimizer.step()
```

The correct order each step:

1. `optimizer.zero_grad()` — clear old gradients
2. forward pass
3. compute loss
4. `loss.backward()` — fill in gradients
5. `optimizer.step()` — apply the update

`CrossEntropyLoss` expects raw logits. Do not apply softmax first.

For the difference between `CrossEntropyLoss` and `NLLLoss`, see [Negative Log-Likelihood Loss vs Cross-Entropy](../negative-log-likelihood-vs-cross-entropy/note.md).

---

## 10. Softmax vs Sigmoid

Use softmax when classes are mutually exclusive.

Example:

```
cat OR dog OR bird
```

Use sigmoid when labels are independent.

Example:

```
is_red AND is_round AND is_large
```

Softmax probabilities compete because they must sum to `1`. Sigmoid outputs do not compete.

---

## 11. Interview Gotchas

- Softmax regression is multinomial logistic regression.
- Use one logit per class.
- It assumes a categorical target: exactly one correct class.
- Use stable softmax by subtracting the row max.
- Cross-entropy only cares about the probability of the true class.
- The clean gradient is `probs - one_hot`.
- Use class IDs with PyTorch `CrossEntropyLoss`.
- Do not apply softmax before PyTorch `CrossEntropyLoss`.
- Softmax is for mutually exclusive classes, not multi-label classification.
