# CosineAnnealingLR — Q&A

---

## Intuition

**Q: What does cosine annealing do?**
A: It smoothly lowers the learning rate.

**Q: Why use it?**
A: It gives training a smooth landing.

---

## 1. The Schedule

**Q: What is `eta_min`?**
A: The minimum learning rate.

**Q: What is `T_max`?**
A: The length of the cosine decay.

---

## 2. Why Cosine?

**Q: How is it different from StepLR?**
A: It decays smoothly instead of dropping abruptly.

---

## 3. PyTorch Pattern

**Q: What PyTorch class implements it?**
A: `torch.optim.lr_scheduler.CosineAnnealingLR`.

**Q: Does it restart by itself?**
A: No.

---

## 4. Interview Gotchas

**Q: What scheduler adds restarts?**
A: `CosineAnnealingWarmRestarts`.
