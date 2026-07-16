# Negative Log-Likelihood Loss vs Cross-Entropy — Q&A

---

## Intuition

**Q: What does cross-entropy measure?**
A: It measures the difference between the true label distribution and the predicted probability distribution.

**Q: Why does this become true-class probability for one-hot labels?**
A: Only the true class has value `1`, so the loss only keeps the predicted probability for that class.

**Q: What is the main PyTorch difference?**
A: `CrossEntropyLoss` expects raw logits. `NLLLoss` expects log-probabilities.

---

## 1. Two Views of the Same Signal

**Q: What is the distribution view?**
A: Cross-entropy asks how different the predicted distribution is from the true distribution.

**Q: What is the likelihood view?**
A: NLL asks how likely the whole observed dataset was under the model.

**Q: Why do these views match for one-hot labels?**
A: The true distribution has `1` only on the correct class, so only that class probability matters.

---

## 2. Negative Log-Likelihood Loss

**Q: What does likelihood mean?**
A: Likelihood is the probability of the observed dataset under the model.

**Q: What is the full likelihood for independent samples?**
A: It is the product of the probabilities assigned to all observed labels.

**Q: What happens when we take negative log of the likelihood product?**
A: The product becomes a sum: `NLL = -sum(log(p_true_i))`.

**Q: What is `-log(p_y)` then?**
A: It is one sample's contribution to the full NLL.

---

## 3. Cross-Entropy

**Q: What does cross-entropy compare?**
A: It compares the true label distribution with the predicted distribution.

**Q: What is cross-entropy for one-hot labels?**
A: `-sum(y_k * log(p_k))`.

**Q: Why does cross-entropy become NLL for one-hot labels?**
A: Only the true class has `y_k = 1`, so across the dataset it becomes `-sum(log(p_true_i))`.

---

## 4. When They Are the Same

**Q: When are cross-entropy and NLL the same?**
A: They are the same for one-label multi-class classification when NLL receives `log_softmax(logits)`.

**Q: What is the PyTorch equivalence?**
A: `CrossEntropyLoss(logits, y)` equals `NLLLoss(LogSoftmax(logits), y)` for class ID targets.

**Q: What labels are used in this common case?**
A: Class ID labels, like `[0, 2, 1]`.

---

## 5. Input Difference

**Q: What input does `CrossEntropyLoss` expect?**
A: Raw logits.

**Q: What input does `NLLLoss` expect?**
A: Log-probabilities.

**Q: Should raw logits go directly into `NLLLoss`?**
A: No. Apply `log_softmax` first.

**Q: What does `log_softmax` do?**
A: It converts logits into log-probabilities in a numerically stable way.

---

## 6. PyTorch Rule

**Q: What should you use by default for multi-class classification?**
A: Use `CrossEntropyLoss`.

**Q: When should you use `NLLLoss`?**
A: Use it when the model already outputs log-probabilities.

**Q: What is the safe manual pattern?**
A: `log_probs = log_softmax(logits, dim=1)`, then `nll_loss(log_probs, y)`.

---

## 7. Common Mistakes

**Q: Should you apply softmax before `CrossEntropyLoss`?**
A: No. Pass raw logits.

**Q: Should you apply log-softmax before `CrossEntropyLoss`?**
A: No. `CrossEntropyLoss` does that internally.

**Q: What is a common `NLLLoss` mistake?**
A: Passing raw logits directly to it.

---

## 8. Interview Gotchas

**Q: What is the probability view?**
A: NLL punishes low probability on the observed labels across the dataset.

**Q: What is the distribution view?**
A: Cross-entropy compares true and predicted distributions.

**Q: What is the shortest PyTorch summary?**
A: `CrossEntropyLoss = LogSoftmax + NLLLoss`.
