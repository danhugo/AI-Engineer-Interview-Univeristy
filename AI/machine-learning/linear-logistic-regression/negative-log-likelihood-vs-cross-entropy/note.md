# Negative Log-Likelihood Loss vs Cross-Entropy — Interview Knowledge Sheet

## Intuition

Cross-entropy measures the difference between:

- the **true label distribution**
- the **predicted probability distribution**

For a normal class label, the true distribution is one-hot:

```
cat = 1
dog = 0
bird = 0
```

The model predicts a distribution:

```
cat = 0.80
dog = 0.15
bird = 0.05
```

The closer the predicted distribution is to the true label distribution, the smaller the loss.

With one-hot labels, this becomes the same as asking:

**How much probability did the model give to the observed labels overall?**

The main difference in PyTorch is the input:

| Loss | Input |
|------|-------|
| `CrossEntropyLoss` | raw logits |
| `NLLLoss` | log-probabilities |

---

## 1. Two Views of the Same Signal

There are two useful ways to explain the same loss.

### Distribution view

Cross-entropy asks:

**How different is the predicted distribution from the true distribution?**

If the true label is `cat`, the ideal distribution is:

```
cat = 1
dog = 0
bird = 0
```

The model should move its predicted distribution toward that target.

### Likelihood view

NLL asks:

**How likely was the whole observed dataset under the model?**

For independent samples, the dataset likelihood multiplies the probability of each observed label.

Then NLL takes the negative log of that product.

The single `-\log(p_y)` term is one sample's contribution to the full NLL.

With one-hot labels, the cross-entropy view and likelihood view lead to the same objective.

---

## 2. Negative Log-Likelihood Loss

Likelihood measures how plausible a set of model parameters is, given the data you actually observed. Probability asks how likely data is under a known model. Likelihood fixes the data and asks which model parameters most likely generated it.

For independent samples, the full likelihood is a product:

$$
L(\theta) = \prod_{i=1}^{n} P_\theta(Y_i = y_i \mid x_i)
$$

For multi-class classification:

$$
L(\theta) = \prod_{i=1}^{n} p_{i,y_i}
$$

Where `p_{i,y_i}` is the model probability for the true class of sample `i`.

Negative log-likelihood turns that product into a sum:

$$
\text{NLL}(\theta)
= -\log L(\theta)
= -\sum_{i=1}^{n} \log(p_{i,y_i})
$$

So NLL means:

```
multiply the probabilities of all observed labels
take log of that product
negate it
```

In practice, we compute the sum of negative logs directly. This avoids multiplying many small probabilities.

For one sample, the contribution is:

$$
\text{NLL}_i = -\log(p_{i,y_i})
$$

That single-sample term is what often appears in PyTorch loss formulas before reduction.

---

## 3. Cross-Entropy

Cross-entropy compares the true label distribution with the predicted distribution.

For one-hot labels:

$$
\text{CE} = -\sum_{k=1}^{K} y_k \log(p_k)
$$

Only the true class has `y_k = 1`. All other classes have `y_k = 0`.

For one sample, it becomes:

$$
\text{CE} = -\log(p_y)
$$

Across the full dataset:

$$
\text{CE}
= -\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\log(p_{ik})
= -\sum_{i=1}^{n}\log(p_{i,y_i})
$$

That is the same objective as NLL for one-hot labels. Some libraries return the mean instead of the sum.

---

## 4. When They Are the Same

For normal multi-class classification with class ID labels:

$$
\text{CrossEntropyLoss}(\text{logits}, y)
=
\text{NLLLoss}(\log\text{softmax}(\text{logits}), y)
$$

They are the same when:

- each example has one correct class
- labels are class IDs
- `CrossEntropyLoss` receives raw logits
- `NLLLoss` receives log-probabilities

PyTorch documents this directly: `CrossEntropyLoss` is equivalent to `LogSoftmax` followed by `NLLLoss` for class-index targets.

---

## 5. Input Difference

### CrossEntropyLoss

Input is raw logits:

```
logits = model(X)
loss = torch.nn.CrossEntropyLoss()(logits, y)
```

Logits are raw scores. They do not need to be positive. They do not need to sum to `1`.

`CrossEntropyLoss` applies the log-softmax step internally.

### NLLLoss

Input is already log-probabilities:

```
log_probs = torch.nn.LogSoftmax(dim=1)(logits)
loss = torch.nn.NLLLoss()(log_probs, y)
```

Do not pass raw logits directly to `NLLLoss`.

`log_softmax` means:

```
logits -> softmax probabilities -> log probabilities
```

It does this in one stable step. It keeps the class ranking from the logits, but converts the scores into log-probabilities for `NLLLoss`.

---

## 6. PyTorch Rule

Use `CrossEntropyLoss` by default for multi-class classification:

```
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, y_class_ids)
```

Use `NLLLoss` only when your model already outputs log-probabilities:

```
log_probs = model(X)
loss_fn = torch.nn.NLLLoss()
loss = loss_fn(log_probs, y_class_ids)
```

The common safe pattern is:

```
log_probs = torch.nn.functional.log_softmax(logits, dim=1)
loss = torch.nn.functional.nll_loss(log_probs, y)
```

---

## 7. Common Mistakes

- Do not pass raw logits to `NLLLoss`.
- Do not apply softmax before `CrossEntropyLoss`.
- Do not apply log-softmax before `CrossEntropyLoss`.
- Use class IDs for the common PyTorch case.
- Use probability targets only when you really need soft labels.

---

## 8. Interview Gotchas

- NLL is the likelihood view: punish low probability on the observed labels across the dataset.
- Formal NLL starts from the full dataset likelihood, a product over samples.
- Cross-entropy is the distribution view: compare true distribution and predicted distribution.
- With one-hot labels, cross-entropy reduces to NLL.
- In PyTorch, `CrossEntropyLoss = LogSoftmax + NLLLoss` for class ID targets.
- The difference is usually input format, not the core idea.

---

## References

- PyTorch `CrossEntropyLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch `NLLLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
- PyTorch `LogSoftmax`: https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
- Multinomial logistic regression likelihood: https://en.wikipedia.org/wiki/Multinomial_logistic_regression#Likelihood_function
- Maximum likelihood estimation: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
- Extra reading: https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81/
