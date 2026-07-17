# Cosine Annealing with Warm Restarts — Q&A

---

## Intuition

**Q: What does warm restart mean?**
A: The learning rate jumps back high at the start of a new cosine cycle.

**Q: Why restart?**
A: It can help the optimizer explore again.

---

## 1. The Schedule

**Q: What is `T_cur`?**
A: Position inside the current cycle.

**Q: What happens at restart?**
A: `T_cur` resets to `0`.

---

## 2. Cycle Length

**Q: What is `T_0`?**
A: The length of the first cycle.

**Q: What is `T_mult`?**
A: The multiplier for future cycle lengths.

---

## 3. PyTorch Pattern

**Q: What PyTorch class implements warm restarts?**
A: `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.

---

## 4. Interview Gotchas

**Q: How is this different from CosineAnnealingLR?**
A: Warm restarts repeat cycles; plain CosineAnnealingLR does not restart.
