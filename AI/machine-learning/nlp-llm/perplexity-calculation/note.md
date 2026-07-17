# Perplexity Calculation - Interview Knowledge Sheet

## Intuition

Perplexity measures how surprised a language model is by a sequence.

Lower perplexity means the model assigned higher probability to the observed tokens. A useful intuition is:

```text
perplexity = average branching factor
```

If a model has perplexity `10`, it is behaving as if it had about 10 equally likely choices at each token.

---

## 1. From Token Probabilities

For a token sequence with probabilities:

$$
p(x_1), p(x_2), \ldots, p(x_N)
$$

the average negative log likelihood is:

$$
\text{NLL}_{avg} = -\frac{1}{N}\sum_{i=1}^{N}\log p(x_i)
$$

Perplexity is:

$$
\text{PPL} = \exp(\text{NLL}_{avg})
$$

Equivalently:

$$
\text{PPL} =
\left(\prod_{i=1}^{N}\frac{1}{p(x_i)}\right)^{1/N}
$$

So perplexity is the geometric mean inverse probability of the true tokens.

---

## 2. Cross-Entropy Connection

For next-token prediction, cross-entropy is usually the average negative log probability of the true token.

If cross-entropy uses natural logs:

$$
\text{PPL} = e^{H}
$$

If cross-entropy uses log base 2:

$$
\text{PPL} = 2^{H}
$$

Most deep learning libraries use natural logs, so `exp(loss)` is the common pattern.

---

## 3. From Logits

Models usually output logits, not probabilities.

For logits `z` and target token index `y`:

$$
\log p(y) = z_y - \log\sum_j e^{z_j}
$$

The stable implementation uses log-softmax or log-sum-exp, not `softmax` followed by `log`.

---

## 4. Masking and Token Counts

Perplexity must divide by the number of evaluated tokens.

Common masks:

- ignore padding tokens
- ignore prompt tokens when evaluating only completions
- ignore labels set to `-100` in PyTorch-style language modeling

If you divide by batch size or sequence length including padding, the perplexity is wrong.

---

## 5. Interpretation

Perplexity is useful for comparing language models on the same dataset with the same tokenizer.

Be careful comparing across:

- different tokenizers
- different text normalization
- different languages
- different evaluation masks

A model can have lower perplexity and still be worse for a downstream task, especially for instruction following or factuality.

---

## Interview Gotchas

- Perplexity is exponentiated average NLL.
- Use natural-log `exp(loss)` if the loss is computed with natural logs.
- Lower is better.
- Count only evaluated target tokens.
- Use log-softmax/log-sum-exp for numerical stability.
- Do not compare perplexity across different tokenizers without caveats.

---

## References

- Jurafsky and Martin, "Speech and Language Processing", Chapter 3: https://web.stanford.edu/~jurafsky/slp3/
