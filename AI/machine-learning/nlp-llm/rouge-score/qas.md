# ROUGE Score - Q&A

---

## Intuition

**Q: What does ROUGE measure?**
A: Overlap between a generated summary and human reference summaries.

**Q: What task is ROUGE most associated with?**
A: Summarization.

**Q: What does recall-oriented mean here?**
A: The score emphasizes how much reference content the candidate recovered.

---

## 1. ROUGE-N

**Q: What does ROUGE-1 compare?**
A: Unigram overlap.

**Q: What does ROUGE-2 compare?**
A: Bigram overlap.

**Q: Why are counts clipped?**
A: To prevent repeated candidate n-grams from receiving more credit than the reference contains.

---

## 2. Precision, Recall, and F1

**Q: What is ROUGE recall denominator?**
A: The number of reference n-grams.

**Q: What is ROUGE precision denominator?**
A: The number of candidate n-grams.

**Q: Why report F1?**
A: It balances reference coverage with candidate conciseness.

---

## 3. ROUGE-L

**Q: What sequence does ROUGE-L use?**
A: The longest common subsequence.

**Q: Does an LCS require adjacent tokens?**
A: No. It preserves order but allows gaps.

**Q: Why can ROUGE-L be more flexible than ROUGE-2?**
A: It can credit ordered matches even when words are not adjacent.

---

## 4. Limitations

**Q: Can ROUGE detect factual hallucinations?**
A: Not reliably.

**Q: Can ROUGE reward invalid summaries?**
A: Yes, if they share many surface tokens with the reference.

**Q: What implementation detail can change ROUGE scores?**
A: Tokenization and text normalization.
