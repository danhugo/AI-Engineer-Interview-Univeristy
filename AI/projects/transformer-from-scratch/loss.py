"""Cross-entropy with label smoothing and padding mask. STUDY ONLY.

The real pipeline uses nn.CrossEntropyLoss(ignore_index=..., label_smoothing=...).
test_loss.py asserts this matches it.

Cross-entropy
-------------
The model outputs logits over the vocabulary. Softmax turns them into
probabilities. Cross-entropy is the negative log probability the model assigned
to the correct token:

    loss = -log p[correct]

Perfect prediction (p=1) gives loss 0. p=0.5 gives 0.69. p -> 0 gives infinity.

Why log_softmax and not log(softmax(x))
---------------------------------------
softmax exponentiates. Large logits overflow to inf, small ones underflow to 0,
and log(0) is -inf. log_softmax uses the identity

    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

Subtracting the max keeps every exponent <= 0, so exp never overflows. Same
value, no NaN.

Padding
-------
Batched sequences are padded to equal length. Those positions carry no
information. Counting them in the loss teaches the model to predict <pad>,
which is both useless and easy — it will happily drive the loss down by getting
padding right. ignore_index drops them from both the sum and the count.

Label smoothing
---------------
The hard target says p[correct] should be 1.0 and everything else 0.0. To reach
that the model must push the correct logit to +inf: it becomes overconfident,
and confident mistakes are expensive.

Smoothing spreads a small mass eps across all classes:

    target[correct] = 1 - eps + eps/V
    target[other]   = eps/V

The model is now rewarded for being right but not certain. The paper uses
eps=0.1. It slightly worsens perplexity and reliably improves BLEU, because
translation has many valid outputs and absolute confidence in one is wrong.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Mean cross-entropy over non-ignored positions.

    Args:
        logits: (N, V) raw scores, unnormalized
        targets: (N,) correct class indices
        ignore_index: target value to skip entirely (padding)
        label_smoothing: eps in [0, 1)

    Returns:
        scalar loss, averaged over positions that were not ignored
    """
    if logits.dim() != 2:
        raise ValueError(f"expected logits (N, V), got {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"expected targets (N,), got {tuple(targets.shape)}")

    vocab_size = logits.size(-1)

    # (N, V) log probabilities, computed stably
    log_probs = F.log_softmax(logits, dim=-1)

    mask = targets != ignore_index
    if mask.sum() == 0:
        return logits.sum() * 0.0  # keeps the graph connected, value 0

    # gather() would fail on ignore_index (often -100, out of range), so clamp
    # first and drop those rows with the mask afterwards.
    safe_targets = targets.clamp(min=0)

    # -log p[correct] for every position: (N,)
    nll = -log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

    if label_smoothing > 0:
        # mean over the vocabulary of -log p, i.e. the loss against a target
        # that is uniform over all classes
        smooth = -log_probs.mean(dim=-1)
        # torch's convention: (1 - eps) * nll + eps * uniform_loss
        per_position = (1 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_position = nll

    # average over real tokens only — the count is mask.sum(), not len(targets)
    return per_position[mask].sum() / mask.sum()


def sequence_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Same loss for sequence output, flattening batch and time.

    Args:
        logits: (batch, seq_len, vocab)
        targets: (batch, seq_len)

    The flatten is the only difference. Every (batch, position) pair is an
    independent next-token prediction, so they all go in one pool.
    """
    vocab_size = logits.size(-1)
    return cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
