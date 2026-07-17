# Weight Decay vs L2 Regularization — Q&A

---

## Intuition

**Q: What does L2 regularization do?**
A: It adds a squared-weight penalty to the loss.

**Q: What does weight decay do?**
A: It shrinks weights during the optimizer update.

**Q: What is the shared goal?**
A: Discourage large weights and improve generalization.

---

## 1. L2 Regularization

**Q: What penalty does L2 add?**
A: Usually `lambda / 2 * sum(w ** 2)` or `lambda * sum(w ** 2)`.

**Q: What is the gradient of `lambda / 2 * sum(w ** 2)`?**
A: `lambda * w`.

**Q: Does L2 change the model prediction formula?**
A: No. It changes the training objective.

---

## 2. Weight Decay

**Q: What is the SGD weight decay update?**
A: `w = (1 - lr * weight_decay) * w - lr * grad`.

**Q: Why is it called decay?**
A: Because weights are multiplied by a factor below `1`.

**Q: What happens with larger weight decay?**
A: Stronger shrinkage and higher underfitting risk.

---

## 3. SGD Equivalence

**Q: When are L2 and weight decay equivalent?**
A: For plain SGD, up to coefficient conventions.

**Q: Why do they match for SGD?**
A: The L2 gradient term becomes the same `-lr * lambda * w` shrinkage term.

**Q: What common convention changes exact constants?**
A: Whether the penalty uses `lambda * sum(w ** 2)` or `lambda / 2 * sum(w ** 2)`.

---

## 4. Adam Difference

**Q: Why can L2 and weight decay differ in Adam?**
A: Adam rescales gradients, so an L2 penalty inside the gradient is adaptively scaled too.

**Q: What is decoupled weight decay?**
A: Shrinking weights separately from the adaptive gradient update.

**Q: Which optimizer is the common Adam variant with decoupled decay?**
A: AdamW.

---

## 5. Parameters to Exclude

**Q: Should the bias usually be weight-decayed?**
A: Usually no.

**Q: What other parameters are often excluded?**
A: Batch norm and layer norm scale/shift parameters.

**Q: Why exclude them?**
A: They do not control model complexity like large kernel weights do.

---

## 6. Interview Gotchas

**Q: Is every framework's `weight_decay` decoupled?**
A: No. Check the optimizer implementation.

**Q: Is AdamW just Adam with an L2 loss penalty?**
A: No. Its weight decay is decoupled from Adam's adaptive gradient step.

**Q: What should you mention in an interview?**
A: Equivalence for SGD, non-equivalence for Adam-style optimizers, and bias/norm exclusions.
