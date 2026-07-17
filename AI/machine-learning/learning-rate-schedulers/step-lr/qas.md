# StepLR — Q&A

---

## Intuition

**Q: What does StepLR do?**
A: It drops the learning rate by a fixed factor every fixed interval.

**Q: Is the drop smooth or abrupt?**
A: Abrupt.

---

## 1. The Schedule

**Q: What does `gamma` do?**
A: It multiplies the learning rate at each drop.

**Q: What does `step_size` do?**
A: It controls how often the learning rate drops.

---

## 2. When to Use

**Q: Why use StepLR?**
A: It is simple and easy to debug.

**Q: What is a downside?**
A: The drops are sudden.

---

## 3. PyTorch Pattern

**Q: What PyTorch class implements StepLR?**
A: `torch.optim.lr_scheduler.StepLR`.

**Q: When do you usually call `scheduler.step()`?**
A: After `optimizer.step()`.

---

## 4. Interview Gotchas

**Q: What changes if you step per batch instead of per epoch?**
A: `step_size` is interpreted in batches instead of epochs.
