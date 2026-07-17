# Extend BPE Tokenizer - Q&A

---

## Intuition

**Q: What does it mean to extend a tokenizer?**
A: Add new vocabulary entries after the tokenizer has already been trained.

**Q: Why extend a BPE tokenizer?**
A: To add domain terms, control markers, or placeholders that should be encoded as units.

---

## 1. BPE

**Q: What does BPE repeatedly merge?**
A: Frequent adjacent token pairs.

**Q: Why is BPE useful for rare words?**
A: Rare words can be represented as smaller subword pieces instead of becoming unknown tokens.

**Q: Does adding a token retrain all BPE merges?**
A: No. It appends entries or special handling without relearning the full merge table.

---

## 2. Added Tokens

**Q: What is the difference between normal and special added tokens?**
A: Special tokens are control markers that should not be split and can be skipped during decoding.

**Q: What should happen to `<|assistant|>` during encoding?**
A: It should remain one atomic token.

**Q: What ID does a new token usually get?**
A: A new ID after the existing vocabulary IDs.

---

## 3. Model Alignment

**Q: Why resize token embeddings after adding tokens?**
A: The model needs embedding rows for the new token IDs.

**Q: What shape is a token embedding matrix?**
A: `vocab_size x hidden_dim`.

**Q: Are new token embeddings meaningful immediately?**
A: No. They are initialized and usually need fine-tuning.

---

## 4. Gotchas

**Q: What happens if tokenizer size exceeds model embedding rows?**
A: New token IDs can index missing embedding rows and crash.

**Q: Can tokenization alone teach a model a new concept?**
A: No. It only changes representation; learning requires training or fine-tuning.

**Q: When should special tokens be used?**
A: For structural markers such as chat roles, EOS, BOS, tools, or modality placeholders.
