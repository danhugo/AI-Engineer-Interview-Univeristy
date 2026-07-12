# Softmax Regression Gradient Descent — Interview Knowledge Sheet

## Intuition

**Softmax regression is logistic regression for more than two classes.**

Instead of one score, the model outputs one logit per class:

```
logits = XW + b
```

For `3` classes, one example might have logits:

```
[2.0, 0.5, -1.0]
```

These are raw scores, not probabilities.

---

## 1. Softmax Turns Class Scores Into Probabilities

Softmax converts logits into probabilities that sum to `1`:

```
prob[class k] = exp(logit[k]) / sum(exp(all logits))
```

The largest logit gets the largest probability.

Prediction is:

```
argmax(probabilities)
```

---

## 2. Stable Softmax

Naive softmax can overflow when logits are large.

Use the stable version:

```
shifted = logits - max(logits)
softmax = exp(shifted) / sum(exp(shifted))
```

Subtracting the max does not change the final probabilities. It only makes the computation safer.

In matrix form, subtract the row max for each example.

---

## 3. Cross-Entropy

For multi-class classification, the usual loss is cross-entropy:

```
loss = -mean(log(probability assigned to the true class))
```

If the true class is `2`, only the probability of class `2` matters for that example.

Confident correct prediction → low loss.

Confident wrong prediction → high loss.

---

## 4. One-Hot Labels vs Class IDs

Two common label formats:

| Format | Example | Used by |
|--------|---------|---------|
| class IDs | `[0, 2, 1]` | PyTorch `CrossEntropyLoss` |
| one-hot | `[[1,0,0], [0,0,1], [0,1,0]]` | manual NumPy formulas |

In interviews, be clear about which format your function expects.

PyTorch `CrossEntropyLoss` expects:

- raw logits
- class ID labels
- no softmax applied first

---

## 5. NumPy Gradient Descent

For softmax regression:

```
logits = X @ W + b
probs = softmax(logits)
one_hot = one_hot(y)
error = probs - one_hot
dW = X.T @ error / n
db = mean(error, axis=0)
```

Then update:

```
W = W - lr * dW
b = b - lr * db
```

---

## 6. Interview Gotchas

- Softmax is for mutually exclusive classes.
- Use sigmoid/BCE for independent multi-label problems.
- Use stable softmax by subtracting the max logit.
- Do not apply softmax before PyTorch `CrossEntropyLoss`.
- PyTorch labels should be class IDs, not one-hot vectors.

