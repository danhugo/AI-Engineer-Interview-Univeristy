# Binary Logistic Regression — Q&A

---

## Intuition

**Q: What is binary logistic regression used for?**
A: Binary classification, where each example belongs to class `0` or class `1`.

**Q: What does the model output after sigmoid?**
A: A probability for class `1`.

**Q: What are the two main steps?**
A: Compute a linear score, then pass it through sigmoid.

---

## 1. The Model

**Q: What is a logit?**
A: The raw linear score `Xw + b`.

**Q: Can a logit be any real number?**
A: Yes. It is not a probability yet.

**Q: What does sigmoid do?**
A: It maps the logit to a probability between `0` and `1`.

---

## 2. Log-Odds Meaning

**Q: What does logistic regression model linearly?**
A: The log-odds of class `1`.

**Q: What are odds?**
A: Odds are `p / (1-p)`.

**Q: Why is the raw score called a logit?**
A: It is the log of the odds.

---

## 3. Bernoulli Assumption

**Q: What distribution is used for binary labels?**
A: Bernoulli.

**Q: What does the likelihood multiply?**
A: The probabilities assigned to the observed labels.

**Q: What loss comes from negative log-likelihood?**
A: Binary cross-entropy.

---

## 4. Binary Cross-Entropy

**Q: What does BCE reward?**
A: High probability on the true class.

**Q: What does BCE punish strongly?**
A: Confident wrong predictions.

**Q: Why clip probabilities in NumPy?**
A: To avoid `log(0)`.

---

## 5. Thresholding

**Q: How do you convert probability to a class?**
A: Compare it to a threshold, often `0.5`.

**Q: Is `0.5` always best?**
A: No. It depends on the cost of false positives and false negatives.

**Q: What happens when you lower the threshold?**
A: The model predicts more positives, often increasing recall and false positives.

---

## 6. Decision Boundary

**Q: Where is the `0.5` decision boundary?**
A: Where the logit `Xw + b` equals `0`.

**Q: Is the decision boundary linear?**
A: Yes, if the original input features are used.

**Q: Are the probabilities linear?**
A: No. Sigmoid makes probabilities nonlinear.

---

## 7. PyTorch Pattern

**Q: What PyTorch loss should you use with logits?**
A: `BCEWithLogitsLoss`.

**Q: Should you apply sigmoid before `BCEWithLogitsLoss`?**
A: No. The loss includes sigmoid internally.

**Q: When should you apply sigmoid?**
A: During inference, when you want probabilities or thresholded predictions.

---

## 8. Interview Gotchas

**Q: Why is it called regression?**
A: It uses a linear model for the log-odds, even though the task is classification.

**Q: What is the most common PyTorch mistake?**
A: Passing sigmoid probabilities into `BCEWithLogitsLoss`.

**Q: What should the threshold depend on?**
A: The cost of different mistakes.
