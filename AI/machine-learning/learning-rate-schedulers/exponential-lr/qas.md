# ExponentialLR — Q&A

---

## Intuition

**Q: What does ExponentialLR do?**
A: It multiplies the learning rate by the same factor every step.

**Q: Is the decay abrupt?**
A: No. It is steady multiplicative decay.

---

## 1. The Schedule

**Q: What is the formula?**
A: `base_lr * gamma ** step`.

**Q: What happens when `gamma < 1`?**
A: The learning rate decays.

---

## 2. When to Use

**Q: When is ExponentialLR useful?**
A: When you want simple steady decay.

**Q: What can go wrong?**
A: The learning rate can shrink too fast if `gamma` is too small.

---

## 3. PyTorch Pattern

**Q: What PyTorch class implements it?**
A: `torch.optim.lr_scheduler.ExponentialLR`.

---

## 4. Interview Gotchas

**Q: Why does step frequency matter?**
A: Multiplying every batch decays much faster than multiplying every epoch.
