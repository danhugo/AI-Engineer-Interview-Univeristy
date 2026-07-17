# Temperature Sampling - Q&A

---

## Intuition

**Q: What does temperature sampling control?**
A: The sharpness of the next-token probability distribution during decoding.

**Q: Does temperature retrain the model?**
A: No. It only changes how logits are converted into sampling probabilities.

---

## 1. Formula

**Q: Where does temperature enter the softmax?**
A: Logits are divided by temperature before softmax.

**Q: What is the formula?**
A: `softmax(logits / T)`.

**Q: What values are valid for temperature?**
A: Positive values only.

---

## 2. Effects

**Q: What does `T < 1` do?**
A: It sharpens the distribution and makes high-logit tokens more likely.

**Q: What does `T > 1` do?**
A: It flattens the distribution and makes sampling more diverse.

**Q: What does `T -> 0` approach?**
A: Greedy argmax decoding.

---

## 3. Sampling

**Q: How is sampling different from greedy decoding?**
A: Sampling draws from probabilities; greedy always picks the highest-probability token.

**Q: Why can sampling be useful?**
A: It can reduce repetitive outputs and improve diversity for open-ended generation.

**Q: Why can high temperature be risky?**
A: It gives more probability to weak candidates, increasing randomness and possible errors.

---

## 4. Practical Use

**Q: What Hugging Face flag enables multinomial sampling?**
A: `do_sample=True`.

**Q: What is a common stable softmax trick?**
A: Subtract the maximum scaled logit before exponentiating.

**Q: Should temperature be used for deterministic factual extraction?**
A: Usually no; lower temperature or deterministic decoding is often preferred.
