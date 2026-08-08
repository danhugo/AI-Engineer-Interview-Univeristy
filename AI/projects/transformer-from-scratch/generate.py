"""Greedy decoding. DRAFT.

Training runs the decoder once over the whole target with teacher forcing.
Generation cannot: there is no target. Tokens come out one at a time, and each
one is fed back in.

    step 1:  decoder(<bos>)              -> 5
    step 2:  decoder(<bos> 5)            -> 1
    step 3:  decoder(<bos> 5 1)          -> 4
    ...until <eos> or max_len

This is where teacher forcing gets audited. If training had the target shift
wrong, loss still looked fine but generation collapses — usually into repeating
one token. Always eyeball a few samples before trusting a loss curve.

Greedy takes the argmax every step. It is not optimal — the highest-probability
token now can lead to a worse sequence overall, which is what beam search
addresses — but it is deterministic and enough to tell working from broken.
"""
import torch
from torch import Tensor

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


@torch.no_grad()
def greedy_decode(
    model,
    src: Tensor,
    max_len: int = 64,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    device: str | torch.device = "cpu",
) -> list[list[int]]:
    """Decode a batch greedily.

    Args:
        model: the Transformer
        src: (batch, src_len) encoder input, already padded
        max_len: stop after this many generated tokens

    Returns one list of token IDs per row, <bos> and everything from <eos>
    onward stripped.

    Note this re-runs the full decoder on the growing prefix every step, which
    is O(n^2) work over the sequence. A KV cache is the standard fix; it is
    left out here because it changes the model's forward signature and this is
    a correctness tool, not a serving path.
    """
    model.eval()
    src = src.to(device)
    batch_size = src.size(0)

    # every sequence starts with <bos>
    tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model(src, tokens)          # (batch, cur_len, vocab)
        next_token = logits[:, -1, :].argmax(dim=-1)  # only the last position

        # once a sequence has emitted <eos>, keep it padded so it stops
        # contributing new content while the rest of the batch continues
        next_token = torch.where(
            finished, torch.full_like(next_token, PAD_ID), next_token
        )
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)

        if bool(finished.all()):
            break

    out: list[list[int]] = []
    for row in tokens.tolist():
        seq = row[1:]  # drop <bos>
        if eos_id in seq:
            seq = seq[: seq.index(eos_id)]
        out.append([t for t in seq if t != PAD_ID])
    return out


@torch.no_grad()
def translate(
    model, sentences: list[str], tokenizer, max_len: int = 64, device="cpu"
) -> list[str]:
    """End-to-end: text in, text out. For Multi30k."""
    from collate import pad_to_max

    encoded = tokenizer.encode_batch(sentences)
    src = pad_to_max(
        [torch.tensor(e.ids[:max_len], dtype=torch.long) for e in encoded]
    )
    token_ids = greedy_decode(model, src, max_len=max_len, device=device)
    return [tokenizer.decode(ids) for ids in token_ids]
