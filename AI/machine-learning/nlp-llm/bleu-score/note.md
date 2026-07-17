# BLEU Score - Interview Knowledge Sheet

## Intuition

BLEU evaluates a machine translation by asking:

```text
Does the candidate use the same short phrases as one or more human references?
```

It is precision-oriented. A candidate gets credit for n-grams that appear in references, but repeated words are clipped so they cannot inflate the score.

BLEU is usually a corpus-level metric. Sentence-level BLEU is common in exercises, but it is less stable.

---

## 1. Modified N-Gram Precision

Plain precision can be gamed by repeating a good word:

```text
the the the the
```

BLEU uses clipped counts:

$$
p_n =
\frac{\sum_{g \in C}\min(\text{count}_C(g), \max_R \text{count}_R(g))}
{\sum_{g \in C}\text{count}_C(g)}
$$

For each candidate n-gram `g`, the allowed count is capped by the maximum count of that n-gram in any reference.

Common BLEU uses `n = 1, 2, 3, 4`.

---

## 2. Geometric Mean

BLEU combines modified precisions with a geometric mean:

$$
\exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

With uniform weights:

$$
w_n = \frac{1}{N}
$$

The geometric mean is strict. If any `p_n` is zero, the unsmoothed BLEU score becomes zero.

That is why sentence-level BLEU often uses smoothing.

---

## 3. Brevity Penalty

A very short candidate can have high precision by saying only a few safe words.

BLEU adds a brevity penalty:

$$
BP =
\begin{cases}
1 & \text{if } c > r \\
e^{1 - r/c} & \text{if } c \le r
\end{cases}
$$

where `c` is candidate length and `r` is effective reference length.

The full BLEU score is:

$$
BLEU = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

---

## 4. Effective Reference Length

For multiple references, BLEU uses an effective reference length for the brevity penalty.

A common sentence-level choice is the reference length closest to the candidate length, breaking ties by choosing the shorter reference.

Corpus BLEU sums candidate lengths and effective reference lengths across the corpus before applying the brevity penalty.

---

## 5. BLEU vs ROUGE

BLEU:

- historically used for machine translation
- precision-oriented
- has brevity penalty
- usually reported at corpus level

ROUGE:

- historically used for summarization
- recall-oriented in its classic form
- often reports ROUGE-1, ROUGE-2, and ROUGE-L

---

## Interview Gotchas

- BLEU uses modified precision, not ordinary precision.
- BLEU clips repeated n-grams using reference counts.
- BLEU has a brevity penalty because precision alone favors short candidates.
- Corpus BLEU is more meaningful than sentence BLEU.
- Tokenization, casing, and smoothing can change scores.
- BLEU does not directly check meaning or factual correctness.

---

## References

- Papineni et al., "Bleu: a Method for Automatic Evaluation of Machine Translation": https://aclanthology.org/P02-1040/
