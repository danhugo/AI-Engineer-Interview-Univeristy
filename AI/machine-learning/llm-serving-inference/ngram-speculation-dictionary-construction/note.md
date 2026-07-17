# N-Gram Speculation Dictionary Construction - Interview Knowledge Sheet

## Intuition

N-gram speculation proposes future tokens by looking for repeated text patterns that already appeared in the prompt.

Instead of running a small draft model, the server builds a lookup table:

$$
\text{recent } n\text{-gram} \rightarrow \text{tokens that followed this } n\text{-gram before}
$$

This is useful when the answer is likely to copy or transform prompt text, such as summarization, extraction, code editing, or templated outputs.

---

## 1. Dictionary Construction

Given prompt tokens:

```text
[a, b, c, a, b, d, a, b, e]
```

For `n = 2`, the key `(a, b)` appears three times. The following continuations are:

```text
(a, b) -> [c, d, e]
```

In serving systems, the value is usually a short continuation window, not just one token:

$$
(x_i, \ldots, x_{i+n-1}) \rightarrow (x_{i+n}, \ldots, x_{i+n+k-1})
$$

where `k` is the maximum number of speculative tokens to propose.

---

## 2. Runtime Lookup

At decode time:

1. Take the most recent `n` generated tokens.
2. Look for that key in the dictionary.
3. Propose the stored continuation.
4. Let the target model verify the proposal.

If several n-gram sizes are allowed, try the longest key first. Longer matches are usually more specific.

---

## 3. Why It Is Safe

The n-gram table only proposes draft tokens. The target model still verifies them with speculative decoding.

If the proposal is wrong, the target model rejects at the first mismatch and generation continues from the verified distribution.

So the dictionary affects speed, not the model distribution, assuming the verifier uses a lossless speculative decoding algorithm.

---

## 4. Practical Knobs

| Knob | Meaning |
| --- | --- |
| `prompt_lookup_min` | smallest n-gram key to try |
| `prompt_lookup_max` | largest n-gram key to try |
| `num_speculative_tokens` | maximum continuation length to draft |

Small n-grams match often but are noisy.

Large n-grams match less often but are more precise.

---

## 5. Cost

For a prompt of length `T`, max n-gram size `N`, and continuation length `K`, a simple construction cost is:

$$
O(T \cdot N \cdot K)
$$

The memory footprint is bounded by the number of observed n-gram positions times the stored continuation length.

In practice, serving systems cap `N`, `K`, and per-request cache size.

---

## Interview Gotchas

- N-gram speculation does not require a draft model.
- It works best when future output overlaps with prompt or previous output.
- Longest-match lookup is usually better than first-match lookup.
- It is a proposer, not an acceptance rule.
- Low match rate means low benefit, but correctness is still protected by target verification.

---

## References

- vLLM N-gram speculation docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/
- vLLM speculative decoding overview: https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM blog on prompt lookup decoding: https://vllm-project.github.io/2024/10/17/spec-decode.html
