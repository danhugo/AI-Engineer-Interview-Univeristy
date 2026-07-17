"""Reference solutions for BLEU Score."""

from collections import Counter
import math
import re


def tokenize(text):
    """Lowercase and split text into word tokens."""
    return re.findall(r"\w+", text.lower())


def ngrams(tokens, n):
    """Return n-grams as tuples."""
    if n <= 0:
        raise ValueError("n must be positive")
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def modified_precision(candidate_tokens, reference_tokens_list, n):
    """Return BLEU clipped n-gram precision."""
    cand_counts = Counter(ngrams(candidate_tokens, n))
    total = sum(cand_counts.values())
    if total == 0:
        return 0.0

    max_ref_counts = Counter()
    for ref_tokens in reference_tokens_list:
        ref_counts = Counter(ngrams(ref_tokens, n))
        for gram, count in ref_counts.items():
            max_ref_counts[gram] = max(max_ref_counts[gram], count)

    clipped = sum(min(count, max_ref_counts[gram]) for gram, count in cand_counts.items())
    return clipped / total


def closest_reference_length(candidate_len, reference_lens):
    """Return the reference length closest to candidate_len; tie goes shorter."""
    return min(reference_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))


def brevity_penalty(candidate_len, reference_len):
    """Return BLEU brevity penalty."""
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return math.exp(1.0 - reference_len / candidate_len)


def sentence_bleu(candidate, references, max_n=4, smooth=False):
    """Return sentence BLEU for one candidate and one or more references."""
    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(ref) for ref in references]
    if not candidate_tokens or not reference_tokens_list:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        precision = modified_precision(candidate_tokens, reference_tokens_list, n)
        if precision == 0.0:
            if not smooth:
                return 0.0
            precision = 1.0 / (2 ** n * max(1, len(candidate_tokens) - n + 1))
        precisions.append(precision)

    ref_len = closest_reference_length(len(candidate_tokens), [len(ref) for ref in reference_tokens_list])
    bp = brevity_penalty(len(candidate_tokens), ref_len)
    log_precision_mean = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_precision_mean)


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


def test_modified_precision():
    cand = tokenize("the the the the")
    refs = [tokenize("the cat is on the mat")]
    check(close(modified_precision(cand, refs, 1), 2 / 4), "unigram clipping failed")
    check(close(modified_precision(tokenize("the cat sat"), refs, 2), 1 / 2), "bigram precision failed")
    print("PASS  modified precision")


def test_brevity_penalty():
    check(close(brevity_penalty(5, 5), 1.0), "equal length BP should be 1")
    check(close(brevity_penalty(4, 6), math.exp(1 - 6 / 4)), "short candidate BP wrong")
    check(closest_reference_length(5, [3, 7, 6]) == 6, "closest reference length wrong")
    print("PASS  brevity penalty")


def test_sentence_bleu():
    score = sentence_bleu(
        "the cat is on the mat",
        ["the cat is on the mat"],
        max_n=4,
    )
    check(close(score, 1.0), f"perfect BLEU should be 1: {score}")

    short = sentence_bleu("the cat", ["the cat is on the mat"], max_n=2)
    check(close(short, math.exp(1 - 6 / 2)), f"brevity penalty not applied: {short}")

    smoothed = sentence_bleu("the cat sat", ["the dog barked"], max_n=4, smooth=True)
    check(smoothed > 0.0, f"smoothed BLEU should be positive: {smoothed}")
    print("PASS  sentence BLEU")


if __name__ == "__main__":
    test_modified_precision()
    test_brevity_penalty()
    test_sentence_bleu()
    print("All tests passed.")
