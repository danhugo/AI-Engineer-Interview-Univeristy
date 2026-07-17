# Dropout — Q&A

---

## Intuition

**Q: What does dropout do during training?**
A: It randomly zeros some activations.

**Q: Why use dropout?**
A: To reduce overfitting and discourage units from relying too heavily on each other.

**Q: Does dropout remove neurons permanently?**
A: No. It only masks activations during a training forward pass.

---

## 1. Forward Pass

**Q: What is `p` in many APIs?**
A: The probability of dropping an activation.

**Q: What is the keep probability if `p = 0.2`?**
A: `0.8`.

**Q: What is inverted dropout?**
A: Scaling kept activations by `1 / (1 - p)` during training.

---

## 2. Train vs Eval

**Q: Is dropout active during inference?**
A: No.

**Q: What should dropout return in eval mode?**
A: The input unchanged.

**Q: What mistake makes validation outputs stochastic?**
A: Forgetting to switch the model to eval mode.

---

## 3. Scaling

**Q: Why scale kept activations?**
A: To keep the expected activation the same as without dropout.

**Q: If `p = 0.5`, what are kept activations multiplied by?**
A: `2.0`.

**Q: What is the expected output of inverted dropout for one activation `x`?**
A: `x`.

---

## 4. Usage

**Q: Where is dropout commonly placed?**
A: After activations in dense or MLP blocks.

**Q: Can too much dropout hurt?**
A: Yes. It can cause underfitting.

**Q: Does dropout directly penalize weight magnitude?**
A: No.

---

## 5. Interview Gotchas

**Q: Is dropout deterministic in train mode?**
A: No, because the mask is random.

**Q: Does PyTorch `Dropout(p=0.3)` keep 30% or drop 30%?**
A: It drops 30%.

**Q: What is the key phrase from the original intuition?**
A: Preventing units from co-adapting too much.
