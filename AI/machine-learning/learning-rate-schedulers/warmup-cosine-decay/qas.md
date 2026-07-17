# Warmup + Cosine Decay — Q&A

---

## Intuition

**Q: What are the two phases?**
A: Linear warmup, then cosine decay.

**Q: What does warmup help with?**
A: Early training stability.

**Q: What does cosine decay help with?**
A: Smoothly lowering the learning rate later in training.

---

## 1. Warmup Phase

**Q: What happens during warmup?**
A: The learning rate ramps from `0` to `base_lr`.

**Q: What controls warmup length?**
A: `warmup_steps`.

---

## 2. Cosine Decay Phase

**Q: What does progress measure?**
A: How far we are through the decay phase.

**Q: Where does the cosine decay end?**
A: At `min_lr`, or near zero if `min_lr = 0`.

---

## 3. Why Cosine?

**Q: Why use cosine instead of a sharp drop?**
A: It changes the learning rate smoothly.

**Q: What is the intuition near the end?**
A: The learning rate gently settles.

---

## 4. PyTorch Pattern

**Q: What PyTorch scheduler can implement this custom schedule?**
A: `LambdaLR`.

**Q: What Hugging Face helper implements it?**
A: `get_cosine_schedule_with_warmup`.

---

## 5. Interview Gotchas

**Q: Should this schedule be based on total steps or only epochs?**
A: Usually total training steps.

**Q: When should scheduler step happen?**
A: Usually after `optimizer.step()`.
