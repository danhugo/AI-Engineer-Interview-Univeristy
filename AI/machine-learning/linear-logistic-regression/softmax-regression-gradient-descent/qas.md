# Softmax Regression Gradient Descent — Q&A

---

## Core Idea

**Q: What is softmax regression?**
A: Multi-class logistic regression. It predicts one probability distribution across several classes.

**Q: How many logits does softmax regression output per example?**
A: One logit per class.

**Q: How do you choose the predicted class?**
A: Take the class with the largest probability, usually with `argmax`.

---

## Softmax

**Q: What does softmax do?**
A: It converts class logits into probabilities that sum to `1`.

**Q: Why do we subtract the max logit before exponentiating?**
A: For numerical stability. It prevents overflow without changing the probabilities.

**Q: Does the largest logit always get the largest softmax probability?**
A: Yes.

---

## Cross-Entropy

**Q: What loss is used for softmax regression?**
A: Multi-class cross-entropy.

**Q: What is the intuition for cross-entropy?**
A: It punishes the model when it assigns low probability to the true class.

**Q: In NumPy, how do you get the true-class probabilities from `probs` and class IDs?**
A: Use advanced indexing: `probs[np.arange(n), labels]`.

---

## Labels

**Q: What is a class ID label?**
A: A single integer class index, like `0`, `1`, or `2`.

**Q: What is a one-hot label?**
A: A vector with `1` at the true class and `0` elsewhere.

**Q: What label format does PyTorch `CrossEntropyLoss` expect?**
A: Class IDs, not one-hot vectors.

---

## Gradient Descent

**Q: What is the softmax regression gradient error term?**
A: `probs - one_hot(labels)`.

**Q: What are the shapes for `W` and `b`?**
A: `W` is `(num_features, num_classes)` and `b` is `(num_classes,)`.

**Q: Why is full-batch vectorized training useful for interviews?**
A: It shows the whole algorithm clearly without hiding the math in loops.

---

## PyTorch

**Q: What should you pass into `torch.nn.CrossEntropyLoss`?**
A: Raw logits and class ID labels.

**Q: Should you call softmax before `CrossEntropyLoss`?**
A: No. It applies log-softmax internally in a stable way.

**Q: When should you call softmax in PyTorch?**
A: During inference, when you want probabilities for interpretation.

