# Count Trainable Parameters with Weight Tying - Q&A

---

## Intuition

**Q: What does weight tying share in a language model?**
A: The input embedding matrix and the output vocabulary projection matrix.

**Q: Why is this parameter-efficient?**
A: Vocabulary matrices are large, and tying replaces two matrices with one shared matrix.

**Q: Does weight tying mean two equal copies?**
A: No. It means one shared trainable tensor.

---

## Counting

**Q: What is the size of an embedding table with vocabulary `V` and hidden size `D`?**
A: `V * D`.

**Q: What is the untied count for input embedding plus output head weights?**
A: `2 * V * D`, ignoring bias.

**Q: What is the tied count for the shared weight?**
A: `V * D`.

---

## Bias

**Q: If the output head has a separate bias, how many parameters does it add?**
A: `V`.

**Q: Is the output bias usually tied to the embedding?**
A: No. It is a separate vector if present.

**Q: With tied weights and output bias, what is the total for embedding/head parameters?**
A: `V * D + V`.

---

## Code Gotchas

**Q: When counting trainable parameters, what should be counted once?**
A: Each unique trainable parameter tensor.

**Q: What common bug overcounts tied parameters?**
A: Summing every module's parameters without deduplicating shared objects.

**Q: What shape compatibility is required for tying?**
A: The model hidden size must match the embedding dimension.

---

## Concepts

**Q: Does tying affect only disk size?**
A: No. It also changes the model's parameterization and can act as regularization.

**Q: What logits formula uses the tied embedding table?**
A: `logits = hidden @ embedding.T`.

**Q: What parameter saving comes from tying?**
A: Usually `V * D` fewer parameters.
