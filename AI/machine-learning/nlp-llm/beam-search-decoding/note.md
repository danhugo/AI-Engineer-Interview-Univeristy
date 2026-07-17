# Beam Search Decoding - Interview Knowledge Sheet

## Intuition

Beam search keeps several partial generations alive instead of committing to one token at a time.

Greedy decoding keeps only the best current prefix.

Beam search keeps the best `k` prefixes, called beams, and expands them at each step.

---

## 1. Objective

Autoregressive decoding scores a sequence by multiplying conditional probabilities:

$$
P(y_{1:T} \mid x) = \prod_{t=1}^{T} P(y_t \mid x, y_{<t})
$$

In practice, use log probabilities:

$$
\log P(y_{1:T} \mid x) = \sum_{t=1}^{T} \log P(y_t \mid x, y_{<t})
$$

Log scores avoid numerical underflow and turn products into sums.

---

## 2. Algorithm

At every step:

1. Expand each active beam with candidate next tokens.
2. Add each next-token log probability to the beam score.
3. Keep only the top `beam_width` candidates.
4. Stop beams that emit EOS.

The final answer is usually the finished sequence with the best normalized score.

---

## 3. Why It Helps

Greedy decoding can pick a locally strong first token that leads to a weak full sequence.

Beam search can keep a slightly worse prefix if it may lead to a better complete sequence.

This is why beam search is common for input-grounded tasks such as translation, summarization, captioning, and speech recognition.

---

## 4. Length Bias

Raw log probabilities are negative.

Longer sequences add more negative terms, so raw beam search can prefer short outputs.

A simple normalization is:

$$
\text{score}(y) = \frac{\sum_t \log P(y_t)}{|y|^\alpha}
$$

where `alpha` controls the strength of length normalization.

---

## 5. Hugging Face Pattern

In Transformers generation:

```python
model.generate(..., num_beams=4, do_sample=False)
```

`num_beams=1` is equivalent to greedy decoding when sampling is disabled.

Beam search can also be combined with sampling, but low-probability branches are still pruned between steps.

---

## 6. Interview Gotchas

- Beam search is approximate; it does not enumerate all possible sequences.
- Scores are summed log probabilities, not raw probabilities.
- Larger beams cost more compute and memory.
- Beam search often reduces diversity compared with sampling.
- EOS handling and length normalization strongly affect output quality.

---

## References

- Hugging Face Transformers generation strategies: https://huggingface.co/docs/transformers/en/generation_strategies
- Graves, "Sequence Transduction with Recurrent Neural Networks": https://arxiv.org/abs/1211.3711
