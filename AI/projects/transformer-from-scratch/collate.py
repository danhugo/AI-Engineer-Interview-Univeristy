"""Batching for seq2seq: pad to length, then shift targets.

Shared by the toy task and Multi30k. This is the single most bug-prone file in
the pipeline, so it is small and separately tested.

The shift
---------
The decoder is trained with teacher forcing: it sees the correct prefix and
predicts the next token. That means one target sequence produces two tensors.

    stored tgt:  <bos>  5  1  4  <eos>

    tgt_in:      <bos>  5  1  4          what the decoder reads
    tgt_out:        5   1  4  <eos>      what it must predict

They are the same tensor offset by one. At every position the decoder reads
tgt_in[t] and is scored against tgt_out[t], which is the next real token.

Get this backwards and the model learns to echo its own input. Loss drops,
training looks healthy, and generation produces garbage — which is why the
overfit-one-batch check in train.py exists.
"""
import torch
from torch import Tensor

PAD_ID = 0


def pad_to_max(sequences: list[Tensor], pad_id: int = PAD_ID) -> Tensor:
    """Stack variable-length 1D tensors into (batch, max_len), right-padded."""
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = seq
    return out


def seq2seq_collate(
    batch: list[tuple[Tensor, Tensor]], pad_id: int = PAD_ID
) -> dict[str, Tensor]:
    """Collate (src, tgt) pairs into a padded, shifted batch.

    Returns a dict rather than a tuple — three tensors positionally is exactly
    the kind of thing that gets silently swapped at the call site.

    Keys:
        src:     (batch, src_len)   encoder input
        tgt_in:  (batch, tgt_len-1) decoder input, teacher forced
        tgt_out: (batch, tgt_len-1) what the decoder must predict
    """
    srcs = [item[0] for item in batch]
    tgts = [item[1] for item in batch]

    src = pad_to_max(srcs, pad_id)
    tgt = pad_to_max(tgts, pad_id)

    # Pad first, then shift. The other order would need per-sequence slicing
    # and leaves ragged lengths for the loss to deal with.
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out}
