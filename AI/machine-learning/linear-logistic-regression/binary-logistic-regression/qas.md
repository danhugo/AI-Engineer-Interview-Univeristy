# Binary Logistic Regression — Q&A

---

## Core Idea

**Q: What is binary logistic regression used for?**
A: Binary classification, where each example belongs to class `0` or class `1`.

**Q: What does logistic regression output?**
A: A probability that the example belongs to class `1`.

**Q: Why is it called logistic regression if it is used for classification?**
A: It models log-odds with a linear function, then turns that score into a probability.

---

## Logits and Sigmoid

**Q: What is a logit?**
A: The raw linear score `Xw + b` before sigmoid.

**Q: Can a logit be outside `[0, 1]`?**
A: Yes. A logit can be any real number.

**Q: What does sigmoid do?**
A: It maps any real-valued logit to a probability between `0` and `1`.

**Q: What probability does logit `0` map to?**
A: `0.5`.

**Q: What does a large positive logit mean?**
A: The predicted probability for class `1` is close to `1`.

---

## Thresholding

**Q: How do you convert a probability into a class prediction?**
A: Compare it to a threshold, usually `0.5`.

**Q: Is `0.5` always the best threshold?**
A: No. The best threshold depends on the cost of false positives and false negatives.

**Q: What happens if you lower the threshold?**
A: The model predicts more positives, which can increase recall but also false positives.

---

## Binary Cross-Entropy

**Q: What loss is commonly used for binary logistic regression?**
A: Binary cross-entropy.

**Q: What is the intuition for binary cross-entropy?**
A: It rewards high probability on the true class and strongly punishes confident wrong predictions.

**Q: Why do we clip probabilities in a NumPy BCE implementation?**
A: To avoid `log(0)`, which is undefined.

---

## PyTorch

**Q: What PyTorch loss should you use for binary logistic regression logits?**
A: `torch.nn.BCEWithLogitsLoss`.

**Q: Should you apply sigmoid before `BCEWithLogitsLoss`?**
A: No. It already includes sigmoid internally.

**Q: When should you apply sigmoid in PyTorch?**
A: During inference, when you want probabilities or thresholded class predictions.

---

## Gotchas

**Q: What is the decision boundary for logistic regression?**
A: A linear boundary where the logit is `0`, which corresponds to probability `0.5`.

**Q: What shape should binary labels usually have for `BCEWithLogitsLoss`?**
A: The same shape as the logits, often `(n, 1)` for a batch of `n` examples.

