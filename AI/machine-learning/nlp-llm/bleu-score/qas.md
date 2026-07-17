# BLEU Score - Q&A

---

## Intuition

**Q: What task is BLEU most associated with?**
A: Machine translation.

**Q: Is BLEU precision-oriented or recall-oriented?**
A: Precision-oriented.

**Q: Why is BLEU usually reported on a corpus?**
A: Corpus-level n-gram statistics are more stable than sentence-level statistics.

---

## 1. Modified Precision

**Q: What makes BLEU precision "modified"?**
A: Candidate n-gram counts are clipped by reference counts.

**Q: Why does BLEU clip counts?**
A: To stop repeated words or phrases from inflating precision.

**Q: What n-gram orders are common in BLEU?**
A: 1-grams through 4-grams.

---

## 2. Geometric Mean

**Q: How does BLEU combine n-gram precisions?**
A: With a weighted geometric mean.

**Q: What happens if an unsmoothed n-gram precision is zero?**
A: The BLEU score becomes zero.

**Q: Why is smoothing often used for sentence BLEU?**
A: Short sentences may have zero matches for high-order n-grams.

---

## 3. Brevity Penalty

**Q: Why does BLEU need a brevity penalty?**
A: Precision alone can reward overly short candidates.

**Q: When is the brevity penalty equal to 1?**
A: When the candidate length is greater than the effective reference length.

**Q: What does BLEU multiply by the brevity penalty?**
A: The geometric mean of modified n-gram precisions.

---

## 4. Limitations

**Q: Can BLEU judge semantic equivalence perfectly?**
A: No.

**Q: What preprocessing choices affect BLEU?**
A: Tokenization, casing, punctuation handling, and smoothing.

**Q: How does BLEU differ from ROUGE?**
A: BLEU is precision-oriented for translation; ROUGE is recall-oriented for summarization.
