# Seq2Seq Training Pipeline — Design

Date: 2026-08-08
Project: `AI/projects/transformer-from-scratch/`

## Goal

Train the hand-written Transformer in `transformer.py` on a real translation
task, end to end.

Today the project has a tokenizer stub, two dataset implementations, and a model
architecture. Nothing calls `.backward()`. This design adds the training
pipeline.

## Core rule

Same split the project already uses for `data.py` (study) and `dataset.py`
(real):

- **From-scratch code is for study.** It gets a test proving it matches the
  library equivalent, and the real pipeline never imports it.
- **The real pipeline uses libraries.**
- **The model architecture is the one exception.** `transformer.py` stays
  hand-written. That is the point of the project.

## Decisions

| Question | Decision |
|---|---|
| Architecture | Encoder-decoder seq2seq (matches existing `Transformer.forward`) |
| Dataset | Synthetic toy task first, then Multi30k de→en |
| Tokenizer | HuggingFace `tokenizers`, byte-level BPE, shared de+en vocab ~8k |
| Optimizer / scheduler / loss / metrics | Hand-written for study; real pipeline uses torch + sacrebleu |
| DataLoader | torch's (`dataset.py`); `data.py` stays study-only |
| Model changes | Add √d_model embedding scale and Xavier init. No weight tying. |

Weight tying was rejected: it interacts with the existing `_zero_padding_grad`
hook on the embedding, which would then also affect the output layer's pad row.
Not worth the subtlety here.

## File layout

### Study-only (from scratch, tested against the library it mirrors)

| File | Contents | Test asserts |
|---|---|---|
| `optim.py` | Adam, AdamW | matches `torch.optim.AdamW` step-for-step to ~1e-6 |
| `schedule.py` | Noam warmup-decay | matches a `LambdaLR` with the same formula |
| `loss.py` | CE + label smoothing + pad mask | matches `nn.CrossEntropyLoss` |
| `metrics.py` | BLEU, perplexity | BLEU matches `sacrebleu` on fixed pairs; perplexity matches `exp(mean CE)` |

These are never imported by `train.py`. Their value is the comparison test: each
one proves the hand-written version reproduces the library's numbers.

### Real pipeline

| File | Contents |
|---|---|
| `toy_data.py` | synthetic reverse-digit pairs, no download |
| `seq2seq_data.py` | Multi30k fetch, BPE training, `PairDataset`, pad-collate |
| `train.py` | the loop: `AdamW` + `LambdaLR` + `nn.CrossEntropyLoss` |
| `generate.py` | greedy decode (draft) |
| `checkpoint.py` | save/load model + optimizer + scheduler + step (draft) |
| `transformer.py` | **stays from scratch** — two fixes below |

## Data flow

```
toy_data / seq2seq_data
   -> (src_ids, tgt_ids), variable length
      -> collate: pad to batch max
                  tgt_in  = tgt[:-1]
                  tgt_out = tgt[1:]
         -> Transformer(src, tgt_in) -> logits (B, T, V)
            -> CrossEntropyLoss(logits.view(-1, V), tgt_out.view(-1))
```

### Special tokens

| Token | ID |
|---|---|
| `<pad>` | 0 |
| `<unk>` | 1 |
| `<bos>` | 2 |
| `<eos>` | 3 |

Targets are stored as `<bos> ... <eos>`. The shift into `tgt_in` / `tgt_out`
happens in collate. This is teacher forcing: the decoder sees the correct prefix
and predicts the next token. Getting this shift wrong trains the model to copy
its own input, which is the most common silent bug in seq2seq — hence the
overfit gate below.

`padding_idx=0` is used in two places, and both are required:

- the model zeroes the gradient of that embedding row
- the loss ignores those target positions via `ignore_index=0`

## Model changes (`transformer.py`)

1. **Embedding scale.** Multiply embeddings by `sqrt(d_model)` before adding
   positional encoding. PE values sit in `[-1, 1]` while embeddings start at
   roughly unit normal, so without the scale the positional signal is
   proportionally too loud early in training.
2. **Xavier-uniform init** on projection weights.

Both get tests.

## Training configuration

| Setting | Value |
|---|---|
| Optimizer | `torch.optim.AdamW`, betas `(0.9, 0.98)`, eps `1e-9` |
| Weight decay | applied to weights only, not biases or LayerNorm params |
| Schedule | Noam: linear warmup then inverse-sqrt decay |
| Warmup steps | 400 for the toy task, 4000 for Multi30k — the toy task converges in far fewer steps than a 4000-step warmup would allow |
| Loss | `nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)` |
| Gradient clipping | `clip_grad_norm_(params, 1.0)` |
| Device | MPS, with CPU fallback |

## Verification order

1. Unit tests for the four study modules against their library equivalents.
2. **Overfit a single toy batch to near-zero loss.** This is the gate. It
   catches shift bugs, mask bugs, and loss bugs at once. If a single batch will
   not overfit, nothing downstream is worth running.
3. Train the toy task to convergence. Check greedy decode actually reverses the
   digit sequence.
4. Train on Multi30k. Report validation perplexity and BLEU.

## Dependencies

One new install: `sacrebleu` (pure Python, small). Everything else is present —
torch 2.11, `tokenizers`, `transformers`, `datasets`, `tqdm`.

## Out of scope

- Mixed precision and multi-GPU
- Beam search (greedy only; beam is a later addition to `generate.py`)
- Weight tying
- Any change to `data.py`, which stays a study-only reference
