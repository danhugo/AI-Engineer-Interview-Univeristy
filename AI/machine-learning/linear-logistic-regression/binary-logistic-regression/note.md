# Binary Logistic Regression — Interview Knowledge Sheet

## Intuition

Binary logistic regression is a **classification** model for two classes.

Linear regression predicts any real number. Binary classification needs a probability between `0` and `1`.

So logistic regression does two steps:

```
linear score -> sigmoid -> probability of class 1
```

The model:

```
logit = Xw + b
prob  = sigmoid(logit)
```

The probability is not a class yet. You still need a threshold.

---

## 1. The Model

The linear part is:

$$
z = Xw + b
$$

`z` is the **logit**.

The sigmoid turns the logit into a probability:

$$
p = P(y=1 \mid x) = \frac{1}{1 + e^{-z}}
$$

Interpretation:

| Logit | Probability |
|-------|-------------|
| large negative | near 0 |
| 0 | 0.5 |
| large positive | near 1 |

---

## 2. Log-Odds Meaning

Logistic regression is linear in the **log-odds**:

$$
\log\left(\frac{p}{1-p}\right) = Xw + b
$$

The left side is called the logit.

That is why the raw score is called a logit. The model learns a linear rule for log-odds, then converts it to probability with sigmoid.

---

## 3. Bernoulli Assumption

Binary labels are usually modeled as Bernoulli trials.

For one sample:

$$
P(Y=y) = p^y(1-p)^{1-y}
$$

If `y = 1`, this becomes `p`.

If `y = 0`, this becomes `1-p`.

For the whole dataset, the likelihood is:

$$
L = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}
$$

Taking negative log gives binary cross-entropy.

---

## 4. Binary Cross-Entropy

The loss is:

$$
\text{BCE} =
-\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

Intuition:

- true label `1` -> want `p` close to `1`
- true label `0` -> want `p` close to `0`
- confident wrong predictions get punished strongly

In NumPy, clip probabilities before taking logs:

```
p = np.clip(p, eps, 1 - eps)
```

This avoids `log(0)`.

---

## 5. Thresholding

A probability becomes a class only after thresholding:

```
if p >= 0.5: predict 1
else:        predict 0
```

The `0.5` threshold is common, but not always best.

For fraud detection or medical screening, false negatives may be expensive. You might lower the threshold to catch more positives. That can increase recall but also increases false positives.

---

## 6. Decision Boundary

With threshold `0.5`, the boundary happens when:

$$
p = 0.5
$$

Sigmoid gives `0.5` when:

$$
Xw + b = 0
$$

So logistic regression has a **linear decision boundary** in the input features.

The probabilities are nonlinear because of sigmoid. The boundary is still linear.

---

## 7. PyTorch Pattern

Use raw logits with `BCEWithLogitsLoss`:

```python
logits = model(X)
loss = torch.nn.BCEWithLogitsLoss()(logits, y)
```

Do not apply sigmoid before this loss:

```python
# Wrong
loss = torch.nn.BCEWithLogitsLoss()(torch.sigmoid(logits), y)
```

`BCEWithLogitsLoss` combines sigmoid and BCE in a numerically stable way.

Use sigmoid later for interpretation:

```python
probs = torch.sigmoid(logits)
preds = (probs >= 0.5).long()
```

---

## 8. Interview Gotchas

- Logistic regression is classification, despite the name.
- The model predicts `P(y=1 | x)`.
- The raw output is a logit, not a probability.
- Use BCE for binary labels.
- Use logits with PyTorch `BCEWithLogitsLoss`.
- Choose the threshold based on the cost of mistakes.
- The decision boundary is linear when using the original features.
