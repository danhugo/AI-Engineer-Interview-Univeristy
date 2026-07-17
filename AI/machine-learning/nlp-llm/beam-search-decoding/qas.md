# Beam Search Decoding - Q&A

---

## Intuition

**Q: What is a beam?**
A: A partial candidate sequence with an accumulated score.

**Q: What does beam width mean?**
A: The number of active candidate prefixes kept after each expansion.

**Q: How is beam search different from greedy decoding?**
A: Greedy keeps one prefix; beam search keeps several.

---

## 1. Scoring

**Q: Why use log probabilities?**
A: They avoid underflow and convert probability products into sums.

**Q: How is a beam score updated?**
A: Add the next token's log probability to the current beam score.

**Q: Why are log scores usually negative?**
A: Probabilities are in `[0, 1]`, so their logs are less than or equal to zero.

---

## 2. Algorithm

**Q: What happens at each decoding step?**
A: Expand beams, score candidates, keep the top beam-width candidates.

**Q: What should happen when a beam emits EOS?**
A: It should be marked complete and not expanded further.

**Q: When can decoding stop early?**
A: When all kept beams are finished or the max step limit is reached.

---

## 3. Tradeoffs

**Q: Why can beam search beat greedy decoding?**
A: It can preserve a lower-scoring prefix that later becomes the best full sequence.

**Q: What is the compute cost of larger beams?**
A: More beams require more model evaluations or larger batched expansions.

**Q: What is a common downside of beam search?**
A: It can produce less diverse or overly generic text.

---

## 4. Practical Use

**Q: What Hugging Face argument enables beam search?**
A: `num_beams` greater than `1`.

**Q: What does `num_beams=1` become without sampling?**
A: Greedy decoding.

**Q: Why use length normalization?**
A: To reduce the bias toward very short sequences.
