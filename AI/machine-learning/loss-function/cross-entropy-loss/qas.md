# Cross-Entropy Loss — Q&A

---

## Intuition

**Q: What does cross-entropy compare?**
A: The true label distribution and the predicted probability distribution.

**Q: When is cross-entropy small?**
A: When the true class gets high probability.

**Q: When is it large?**
A: When the true class gets low probability.

---

## 1. Multi-Class Setting

**Q: When do you use multi-class cross-entropy?**
A: When exactly one class is correct.

**Q: What does the model output before softmax?**
A: One logit per class.

**Q: What turns logits into probabilities?**
A: Softmax.

---

## 2. Formula

**Q: What is the one-hot formula?**
A: `-sum(y_k * log(p_k))`.

**Q: Why does only one term matter?**
A: Only the true class has label value `1`.

**Q: What is the class-ID version?**
A: `-log(p_true_class)`.

---

## 3. PyTorch Pattern

**Q: What input does `CrossEntropyLoss` expect?**
A: Raw logits.

**Q: What target format is common?**
A: Class IDs.

**Q: Should you apply softmax first?**
A: No.

---

## 4. Binary vs Multi-Class

**Q: What loss fits binary classification with logits?**
A: `BCEWithLogitsLoss`.

**Q: What loss fits exactly one class out of many?**
A: `CrossEntropyLoss`.

**Q: What loss fits multi-label classification?**
A: `BCEWithLogitsLoss`.

---

## 5. Interview Gotchas

**Q: What does PyTorch combine inside `CrossEntropyLoss`?**
A: `log_softmax` and `NLLLoss`.

**Q: What is a common mistake?**
A: Applying softmax before `CrossEntropyLoss`.
