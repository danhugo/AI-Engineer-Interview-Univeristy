# N-Gram Speculation Dictionary Construction - Q&A

---

## Intuition

**Q: What is n-gram speculation?**
A: A draft-token method that proposes continuations by matching recent tokens against repeated n-grams in the prompt or cache.

**Q: Does it need a smaller draft model?**
A: No. It uses lookup from observed token sequences.

**Q: When does it work well?**
A: When generated output repeats or closely follows prompt text, such as extraction, summarization, and templates.

---

## Dictionary

**Q: What is the dictionary key?**
A: A tuple of `n` consecutive tokens.

**Q: What is the dictionary value?**
A: The next few tokens that followed that key in the prompt or cached text.

**Q: Why try longer n-grams first?**
A: Longer matches are more specific and usually produce cleaner proposals.

---

## Verification

**Q: Does the dictionary decide final output tokens?**
A: No. It only proposes tokens; the target model verifies them.

**Q: What happens after a wrong proposal?**
A: Verification accepts the matching prefix and rejects at the first bad token.

**Q: Why is this considered lossless when paired with correct speculative decoding?**
A: The target model's distribution is preserved by the verifier and rejection-sampling rule.

---

## Tradeoffs

**Q: What happens if `prompt_lookup_min` is too small?**
A: Matches are frequent but may be low quality.

**Q: What happens if `prompt_lookup_max` is too large?**
A: Matches may be rare.

**Q: What limits memory?**
A: Caps on key sizes, continuation length, and number of cached entries.
