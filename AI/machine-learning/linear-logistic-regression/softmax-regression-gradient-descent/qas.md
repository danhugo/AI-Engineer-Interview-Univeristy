# Softmax Regression Gradient Descent — Q&A

---

## Intuition

**Q: What is softmax regression?**
A: Softmax regression is logistic regression for more than two classes.

**Q: How is softmax regression different from binary logistic regression?**
A: Binary logistic regression outputs one logit. Softmax regression outputs one logit per class.

**Q: What does softmax regression predict?**
A: It predicts one probability distribution across all classes.

---

## 1. Logits Are Class Scores

**Q: What is a logit in softmax regression?**
A: A logit is a raw class score before softmax.

**Q: Do logits need to be positive or sum to 1?**
A: No. Logits are raw scores, not probabilities.

**Q: How many logits does each sample get?**
A: One logit per class.

---

## 2. Softmax Turns Scores Into Probabilities

**Q: What does softmax do?**
A: It converts logits into probabilities that sum to `1`.

**Q: Which class gets the largest softmax probability?**
A: The class with the largest logit.

**Q: How do you choose the predicted class?**
A: Use `argmax(probs)`.

---

## 3. Stable Softmax

**Q: Why can naive softmax overflow?**
A: Exponentials of large logits can become too large to represent.

**Q: How do you make softmax stable?**
A: Subtract the max logit before exponentiating.

**Q: Does subtracting the max change the probabilities?**
A: No. It shifts all logits by the same value, so the softmax result stays the same.

---

## 4. Categorical Distribution Assumption

**Q: What distribution assumption does softmax regression use?**
A: It assumes the label is categorical. There are `K` possible classes, and exactly one class is correct.

**Q: How is this like binary logistic regression?**
A: Binary logistic regression uses a Bernoulli target. Softmax regression uses the multi-class version, a categorical target.

**Q: What is the probability of the observed class?**
A: For class ID `y`, it is `p_y`, the probability assigned to the true class.

---

## 5. Cross-Entropy Loss

**Q: What loss is used for softmax regression?**
A: Multi-class cross-entropy.

**Q: What is the intuition for cross-entropy?**
A: It is the negative log probability assigned to the correct class.

**Q: What happens when the model gives the true class low probability?**
A: The loss becomes high.

---

## 6. The Clean Gradient

**Q: What is the gradient with respect to each class logit?**
A: `p_k - y_k`.

**Q: What is the vector error term?**
A: `probs - one_hot`.

**Q: How does this match binary logistic regression?**
A: Both have the same idea: predicted probability minus true label.

---

## 7. Vectorized Update

**Q: What are the shapes of `W` and `b`?**
A: `W` is `(num_features, num_classes)` and `b` is `(num_classes,)`.

**Q: What is the full-batch gradient for `W`?**
A: `dW = X.T @ error / n`.

**Q: What is the full-batch gradient for `b`?**
A: `db = mean(error, axis=0)`.

---

## 8. One-Hot Labels vs Class IDs

**Q: What is a class ID label?**
A: A single integer class index, like `0`, `1`, or `2`.

**Q: What is a one-hot label?**
A: A vector with `1` at the true class and `0` elsewhere.

**Q: Which label format is easiest for manual NumPy gradients?**
A: One-hot labels.

---

## 9. PyTorch Autograd Training

**Q: What should you pass to PyTorch `CrossEntropyLoss`?**
A: Raw logits and class ID labels.

**Q: Should you apply softmax before `CrossEntropyLoss`?**
A: No. `CrossEntropyLoss` expects logits.

**Q: Why call `optimizer.zero_grad()` each step?**
A: PyTorch accumulates gradients by default. `zero_grad()` clears old gradients.

**Q: Where is the `CrossEntropyLoss` vs `NLLLoss` distinction explained?**
A: See `../negative-log-likelihood-vs-cross-entropy/note.md`.

---

## 10. Softmax vs Sigmoid

**Q: When should you use softmax?**
A: Use softmax when classes are mutually exclusive.

**Q: When should you use sigmoid?**
A: Use sigmoid when labels are independent and multiple labels can be true at once.

**Q: Why do softmax probabilities compete?**
A: They must sum to `1`, so increasing one class probability lowers others.

---

## 11. Interview Gotchas

**Q: What is another name for softmax regression?**
A: Multinomial logistic regression.

**Q: What is the key NumPy gradient shortcut?**
A: `error = probs - one_hot`.

**Q: What is the most common PyTorch mistake?**
A: Applying softmax before `CrossEntropyLoss`.
