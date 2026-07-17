# ROUGE Score - Interview Knowledge Sheet

## Intuition

ROUGE evaluates generated summaries by overlap with one or more reference summaries.

The name comes from "Recall-Oriented Understudy for Gisting Evaluation." The important word is recall: classic ROUGE asks how much of the human reference summary the system summary recovered.

If the reference says:

```text
the cat sat on the mat
```

and the candidate says:

```text
cat sat mat
```

the candidate captured several important words, but missed some reference words. ROUGE turns that intuition into overlap counts.

---

## 1. ROUGE-N

ROUGE-N compares n-grams.

For a candidate summary `C` and reference summary `R`:

$$
\text{ROUGE-N}_{recall} =
\frac{\sum_{g \in R}\min(\text{count}_C(g), \text{count}_R(g))}
{\sum_{g \in R}\text{count}_R(g)}
$$

The numerator is clipped overlap: if the candidate repeats an n-gram many times, it cannot get more credit than the reference count.

Common choices:

- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap

ROUGE-1 is broad and forgiving. ROUGE-2 is stricter because word order must match locally.

---

## 2. Recall, Precision, and F1

The original ROUGE framing emphasizes recall:

$$
\text{recall} = \frac{\text{overlap}}{\text{reference n-grams}}
$$

Many libraries also report precision and F1:

$$
\text{precision} = \frac{\text{overlap}}{\text{candidate n-grams}}
$$

$$
F_1 = \frac{2PR}{P + R}
$$

Recall rewards covering reference content. Precision punishes extra candidate content. F1 balances both.

---

## 3. ROUGE-L

ROUGE-L uses the longest common subsequence (LCS), not exact adjacent n-grams.

An LCS keeps token order but allows gaps.

Example:

```text
reference: the cat sat on the mat
candidate: the cat is on mat
LCS:       the cat on mat
```

For LCS length `L`:

$$
R_{lcs} = \frac{L}{m}
$$

$$
P_{lcs} = \frac{L}{n}
$$

where `m` is the reference length and `n` is the candidate length.

ROUGE-L F1 is often reported as:

$$
F_{lcs} = \frac{2P_{lcs}R_{lcs}}{P_{lcs} + R_{lcs}}
$$

ROUGE-L is useful when a summary uses the right content in roughly the right order but not exactly the same adjacent phrases.

---

## 4. Multiple References

Summarization datasets may have multiple human references.

Common handling:

- compute score against each reference
- take the best score, or average depending on the benchmark/library

Always state the aggregation rule in an interview or evaluation report.

---

## 5. Limitations

ROUGE is cheap and reproducible, but it only sees surface overlap.

It can miss:

- paraphrases
- factual contradictions
- hallucinated details mixed with overlapping words
- readability and coherence

A high ROUGE score does not prove a summary is faithful. A low ROUGE score does not always prove it is bad if it uses valid paraphrases.

---

## Interview Gotchas

- ROUGE is mostly used for summarization, while BLEU is historically used for machine translation.
- ROUGE is recall-oriented in its classic form.
- Use clipped counts so repeated candidate words do not get unlimited credit.
- ROUGE-2 is usually stricter than ROUGE-1.
- ROUGE-L uses longest common subsequence, not longest common substring.
- Tokenization and normalization can change scores.

---

## References

- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries": https://aclanthology.org/W04-1013/
- Microsoft Research summary of ROUGE: https://www.microsoft.com/en-us/research/publication/rouge-a-package-for-automatic-evaluation-of-summaries/
