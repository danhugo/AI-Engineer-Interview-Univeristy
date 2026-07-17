# Linear Warmup Schedule — Q&A

---

## Intuition

**Q: What does linear warmup do?**
A: It gradually increases the learning rate from a small value to the base learning rate.

**Q: Why use warmup?**
A: Early training can be unstable, so warmup avoids large first updates.

**Q: What is the short intuition?**
A: Start cautious, then ramp up.

---

## 1. The Schedule

**Q: What happens during warmup?**
A: The learning rate increases linearly.

**Q: What happens after warmup in the basic schedule?**
A: The learning rate stays at the base value.

**Q: What does `warmup_steps` control?**
A: How many steps the ramp takes.

---

## 2. Why Warmup Helps

**Q: Why can early updates be risky?**
A: Weights are random and optimizer statistics are not stable yet.

**Q: What kind of models often use warmup?**
A: Large deep learning models.

---

## 3. PyTorch Pattern

**Q: What PyTorch scheduler can implement custom warmup?**
A: `LambdaLR`.

**Q: When should `scheduler.step()` usually be called?**
A: After `optimizer.step()`.

---

## 4. Interview Gotchas

**Q: Is warmup usually alone?**
A: Often no. It is usually paired with a decay schedule.

**Q: What happens if warmup is too long?**
A: Training may spend too many steps at a low learning rate.
