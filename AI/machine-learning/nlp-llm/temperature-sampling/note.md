# Temperature Sampling - Interview Knowledge Sheet

## Intuition

Language models output logits for the next token.

Temperature changes how sharp or flat the next-token distribution is before sampling.

It does not change the model weights. It changes decoding behavior.

---

## 1. Formula

Given logits `z` and temperature `T`:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Low temperature makes the largest logit dominate.

High temperature spreads probability mass across more tokens.

---

## 2. What Temperature Means

| Temperature | Effect |
|-------------|--------|
| `T < 1` | sharper, safer, less diverse |
| `T = 1` | original softmax distribution |
| `T > 1` | flatter, more diverse |
| `T -> 0` | approaches greedy argmax |

Temperature must be positive.

---

## 3. Sampling vs Greedy

Greedy decoding chooses:

$$
\arg\max_i p_i
$$

Sampling draws from the distribution:

$$
x \sim \text{Categorical}(p)
$$

With sampling, a lower-probability token can be chosen. That is useful for open-ended generation, but risky for factual or constrained outputs.

---

## 4. Numerical Stability

Use a stable softmax:

```python
scaled = logits / temperature
shifted = scaled - max(scaled)
probs = exp(shifted) / sum(exp(shifted))
```

Subtracting the max does not change the probabilities because softmax is shift-invariant.

---

## 5. Hugging Face Pattern

In Transformers generation, sampling is enabled with:

```python
model.generate(..., do_sample=True, temperature=0.7)
```

If `do_sample=False`, generation uses deterministic search behavior, so temperature is not the main control knob.

---

## 6. Interview Gotchas

- Temperature rescales logits before softmax.
- Low temperature reduces entropy.
- High temperature increases entropy.
- `T = 0` is not a valid softmax temperature; use greedy decoding for argmax.
- Temperature is usually combined with other sampling controls such as top-k or top-p.

---

## References

- Hugging Face Transformers generation strategies: https://huggingface.co/docs/transformers/en/generation_strategies
- Hugging Face `GenerationConfig`: https://huggingface.co/docs/transformers/main_classes/text_generation
