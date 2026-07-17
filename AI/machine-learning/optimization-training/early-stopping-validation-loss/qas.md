# Early Stopping with Validation Loss - Q&A

---

## Intuition

**Q: What problem does early stopping address?**
A: It stops training before the model keeps improving training loss while validation performance stops improving.

**Q: Why is early stopping a regularizer?**
A: It limits how long optimization can adapt to training-specific noise.

**Q: Which curve should usually drive early stopping?**
A: Validation loss or another validation metric.

---

## 1. Train vs Validation Loss

**Q: Why is training loss alone not enough?**
A: Training loss can keep decreasing even while generalization gets worse.

**Q: What does rising validation loss with falling training loss suggest?**
A: Overfitting.

**Q: Which epoch is often the best model?**
A: The epoch with the best validation metric, not necessarily the last epoch.

---

## 2. Basic Rule

**Q: For validation loss, what counts as better?**
A: A lower value.

**Q: What is the best validation loss after several epochs?**
A: The minimum validation loss observed so far.

**Q: What does `min_delta` control?**
A: The minimum improvement required to reset the patience counter.

---

## 3. Patience

**Q: What is `patience`?**
A: The number of epochs with no improvement allowed before stopping.

**Q: Why not stop after the first bad validation epoch?**
A: Validation curves are noisy, so one bad epoch may not mean overfitting.

**Q: With `patience = 2`, when do you stop?**
A: After two consecutive epochs without a qualifying improvement.

---

## 4. Restore Best Weights

**Q: Why restore best weights?**
A: The last epoch may be worse than the best validation epoch.

**Q: What should be saved during training?**
A: The weights from the best validation epoch.

---

## 5. Evaluation Discipline

**Q: Why keep a separate test set?**
A: Early stopping uses validation data for training decisions, so final evaluation should use untouched data.

**Q: What is a common interview mistake?**
A: Saying early stopping uses the test set.
