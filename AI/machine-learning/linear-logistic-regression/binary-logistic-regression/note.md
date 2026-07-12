# Binary Logistic Regression — Interview Knowledge Sheet

## Intuition

**Binary logistic regression turns a linear score into a probability for class 1.**

It is used when the answer has two classes:

- spam or not spam
- fraud or not fraud
- click or no click
- disease or no disease

---

## 1. Linear Score First

The model starts with the same score as linear regression:

```
logit = Xw + b
```

This raw score is called a **logit**.

- Large positive logit → class 1 is likely
- Large negative logit → class 0 is likely
- Logit near 0 → unsure

The logit is not a probability yet. It can be any real number.

---

## 2. Sigmoid Turns Scores Into Probabilities

Sigmoid squashes any real number into the range `[0, 1]`:

```
p = sigmoid(logit) = 1 / (1 + exp(-logit))
```

Interpretation:

| Logit | Probability |
|-------|-------------|
| large negative | near 0 |
| 0 | 0.5 |
| large positive | near 1 |

So logistic regression predicts:

```
P(y = 1 | x)
```

---

## 3. Probabilities Become Classes With a Threshold

A probability is not a class label until we choose a threshold.

The common default is:

```
if p >= 0.5: predict 1
else:        predict 0
```

In interviews, mention that `0.5` is not always best.

For fraud or medical screening, you may lower the threshold to catch more positives. That often increases recall but may create more false positives.

---

## 4. Binary Cross-Entropy

Logistic regression usually trains with **binary cross-entropy**:

```
loss = -mean(y * log(p) + (1 - y) * log(1 - p))
```

Intuition:

- If the true label is `1`, we want `p` close to `1`.
- If the true label is `0`, we want `p` close to `0`.
- Confident wrong predictions get punished strongly.

In NumPy, clip probabilities before taking logs:

```
p = clip(p, eps, 1 - eps)
```

This avoids `log(0)`.

---

## 5. PyTorch: BCEWithLogitsLoss

In PyTorch, prefer:

```
torch.nn.BCEWithLogitsLoss()
```

It expects raw logits, not sigmoid probabilities.

Correct:

```
loss = BCEWithLogitsLoss()(logits, labels)
```

Avoid:

```
loss = BCEWithLogitsLoss()(sigmoid(logits), labels)
```

Why? `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable way.

---

## 6. Interview Gotchas

- Logistic regression is a classification model, even though it has "regression" in the name.
- The output of sigmoid is a probability for class `1`.
- The model learns a linear decision boundary in feature space.
- Use logits for PyTorch loss functions when the loss name says "with logits."
- Use probabilities for human-facing interpretation and thresholding.

