"""BLEU and perplexity from scratch. STUDY ONLY.

The real pipeline uses  . test_metrics.py asserts this matches it.

Perplexity
----------
    ppl = exp(mean cross-entropy)

Read it as "how many tokens is the model effectively choosing between". A
perplexity of 1 means it always knows the answer. A perplexity equal to the
vocabulary size means it learned nothing. It is just the loss on a more
intuitive axis — but note it is exp of the *unsmoothed* loss, so evaluate with
label_smoothing=0 or the number is not comparable to published figures.

BLEU
----
Translation has no single right answer, so exact-match accuracy is useless.
BLEU asks a softer question: of the n-grams the model produced, how many appear
in the reference?

Three parts:

1. Modified n-gram precision, for n = 1..4.
   "Modified" handles the degenerate case. Output "the the the the the" against
   reference "the cat sat" would score 5/5 on raw unigram precision. So each
   n-gram is clipped to the number of times it appears in the reference:
   count_clipped = min(count_in_candidate, count_in_reference). Now it is 1/5.

2. Geometric mean of the four precisions.
   Geometric, not arithmetic, so a zero at any n drives the whole score to
   zero. A translation with no correct 4-grams should not be rescued by good
   unigrams.

3. Brevity penalty.
   Precision alone rewards being short — emit one word you are sure of and
   score 1.0. BP punishes candidates shorter than the reference:

       BP = 1                    if c > r
       BP = exp(1 - r/c)         if c <= r

   There is no penalty for being too long because precision already handles
   that: extra words are unmatched and dilute the score.

    BLEU = BP * exp( sum_n w_n * log p_n ),  w_n = 1/4

Corpus BLEU sums counts across all sentences before dividing, rather than
averaging per-sentence scores. The two differ, and corpus-level is the standard
figure people report.
"""
import math
from collections import Counter

import torch
from torch import Tensor


def perplexity(loss: float | Tensor) -> float:
    """exp of the mean cross-entropy. Use an unsmoothed loss."""
    value = float(loss)
    # exp overflows past ~709; a loss that high means the model is broken
    # anyway, so report inf rather than crashing an eval run.
    if value > 709:
        return float("inf")
    return math.exp(value)


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Count every contiguous run of n tokens."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    candidates: list[list[str]],
    references: list[list[str]],
    max_n: int = 4,
) -> float:
    """Corpus-level BLEU, returned on the 0-100 scale like sacrebleu.

    Args:
        candidates: model outputs, each a list of tokens
        references: gold outputs, each a list of tokens (one reference each)

    Counts are accumulated across the whole corpus and divided once at the end.
    """
    if len(candidates) != len(references):
        raise ValueError(
            f"{len(candidates)} candidates vs {len(references)} references"
        )

    # numerator and denominator of each p_n, summed over all sentences
    matches = [0] * max_n
    totals = [0] * max_n
    cand_len = 0
    ref_len = 0

    for cand, ref in zip(candidates, references):
        cand_len += len(cand)
        ref_len += len(ref)

        for n in range(1, max_n + 1):
            cand_ngrams = _ngrams(cand, n)
            ref_ngrams = _ngrams(ref, n)

            # clip each n-gram to how often it actually occurs in the reference
            overlap = {
                gram: min(count, ref_ngrams[gram])
                for gram, count in cand_ngrams.items()
            }
            matches[n - 1] += sum(overlap.values())
            totals[n - 1] += max(sum(cand_ngrams.values()), 0)

    # any p_n == 0 zeroes the geometric mean
    if any(m == 0 for m in matches) or any(t == 0 for t in totals):
        return 0.0

    log_precision_sum = sum(
        (1.0 / max_n) * math.log(matches[i] / totals[i]) for i in range(max_n)
    )

    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len)

    return 100.0 * bp * math.exp(log_precision_sum)


def sentence_bleu(
    candidate: list[str], reference: list[str], max_n: int = 4
) -> float:
    """BLEU for a single pair. Noisy — short sentences often have no 4-gram
    match at all and score 0. Useful for spot checks, not for reporting."""
    return corpus_bleu([candidate], [reference], max_n=max_n)
