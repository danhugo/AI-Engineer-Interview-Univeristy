"""Seq2seq training loop.

Real pipeline: uses torch's AdamW, LambdaLR, and CrossEntropyLoss. The
from-scratch versions in optim.py / schedule.py / loss.py are study material
and are deliberately not imported here.

Usage:
    python train.py --task toy                  # synthetic, ~1 minute
    python train.py --task toy --overfit        # the correctness gate
    python train.py --task multi30k             # real translation

The overfit gate
----------------
--overfit trains on a single batch and expects the loss to reach ~0. It is the
highest-value test in the file. A model that cannot memorize one batch has a
bug, not a tuning problem, and the three usual causes all show up here:

  - target shift wrong  -> loss plateaus well above zero
  - causal mask wrong   -> loss drops suspiciously fast, generation is garbage
  - pad handling wrong  -> loss drops but decode emits <pad>

Run it before every real training run. It takes seconds.
"""
import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from checkpoint import save_checkpoint
from collate import seq2seq_collate
from generate import greedy_decode
from metrics import corpus_bleu, perplexity
from schedule import noam_lambda
from transformer import Transformer

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay weight matrices, not biases or LayerNorm parameters.

    Decay shrinks parameters toward zero, which limits how large a single
    connection grows. Biases and norm gains are offsets — shrinking them just
    fights what the layer learned.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_model(vocab_size: int, cfg: argparse.Namespace, device) -> Transformer:
    model = Transformer(
        vocab_size=vocab_size,
        padding_idx=PAD_ID,
        num_encoder_layers=cfg.layers,
        num_decoder_layers=cfg.layers,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_heads=cfg.heads,
        max_seq_len=cfg.max_len,
        drop_out=cfg.dropout,
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    """Validation loss and perplexity.

    Loss is accumulated weighted by real token count, not batch count, so
    batches with different amounts of padding are compared fairly.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        n_tokens = int((tgt_out != PAD_ID).sum())
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": mean_loss, "ppl": perplexity(mean_loss)}


def run_overfit_gate(model, batch, cfg, device) -> bool:
    """Memorize one batch. Returns True if loss reached near zero.

    Two things are deliberately off:

    - Label smoothing. It puts a floor under the loss, since you cannot reach
      0 when the target is not one-hot.
    - Dropout. It injects noise on every forward pass specifically to prevent
      memorization, which is the exact thing being measured here. With dropout
      at 0.1 this gate stalls around 0.7 and reports a bug that does not exist.

    Both are set by the caller before the model is built.
    """
    print("\n=== overfit gate: one batch, expecting loss -> 0 ===")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98))

    src = batch["src"].to(device)
    tgt_in = batch["tgt_in"].to(device)
    tgt_out = batch["tgt_out"].to(device)

    model.train()
    loss_value = float("inf")
    for step in range(1, cfg.overfit_steps + 1):
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        optimizer.step()
        loss_value = loss.item()
        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss_value:.4f}")
        if loss_value < 0.01:
            break

    passed = loss_value < 0.05
    print(f"  final loss {loss_value:.4f} -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  a model that cannot memorize one batch has a bug.")
        print("  check: target shift in collate, causal mask, ignore_index")
    return passed


def train(cfg: argparse.Namespace) -> None:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    print(f"device: {device}")

    # ---- data -------------------------------------------------------------
    tokenizer = None
    if cfg.task == "toy":
        from toy_data import TOY_VOCAB_SIZE, ReverseDigitsDataset

        train_set = ReverseDigitsDataset(cfg.train_samples, seed=cfg.seed)
        val_set = ReverseDigitsDataset(500, seed=cfg.seed + 1)
        vocab_size = TOY_VOCAB_SIZE
    else:
        from seq2seq_data import build_multi30k

        splits, tokenizer = build_multi30k(cfg.vocab_size, cfg.max_len)
        train_set, val_set = splits["train"], splits["val"]
        vocab_size = tokenizer.get_vocab_size()

    print(f"train {len(train_set)}  val {len(val_set)}  vocab {vocab_size}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=seq2seq_collate,
        num_workers=0,  # MPS + workers is unstable; data is in RAM anyway
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=seq2seq_collate
    )

    # ---- model ------------------------------------------------------------
    if cfg.overfit:
        # dropout exists to stop memorization; the gate measures memorization
        cfg.dropout = 0.0
    model = build_model(vocab_size, cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")

    if cfg.overfit:
        batch = next(iter(train_loader))
        ok = run_overfit_gate(model, batch, cfg, device)
        raise SystemExit(0 if ok else 1)

    # ---- optimizer, schedule, loss ----------------------------------------
    # lr=1.0 because the Noam lambda produces the actual rate as a multiplier
    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, noam_lambda(cfg.d_model, cfg.warmup)
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID, label_smoothing=cfg.label_smoothing
    )
    # perplexity must come from an unsmoothed loss to be comparable
    eval_criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # ---- loop -------------------------------------------------------------
    step = 0
    best_val = float("inf")
    ckpt_dir = Path(cfg.out) / cfg.task
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0

        for batch in train_loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            logits = model(src, tgt_in)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # clip before step: one bad batch can otherwise blow up the
            # weights and poison Adam's second moment for a long time
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            scheduler.step()

            step += 1
            running += loss.item()
            seen += 1

            if step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {step:6d}  "
                    f"loss {running / seen:.4f}  lr {lr:.2e}  "
                    f"{time.time() - start:.0f}s"
                )
                running, seen = 0.0, 0

        val = evaluate(model, val_loader, eval_criterion, device)
        print(
            f"-- epoch {epoch}: val loss {val['loss']:.4f}  ppl {val['ppl']:.2f}"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler,
                step=step, epoch=epoch, config=vars(cfg), best_val_loss=best_val,
            )
            print(f"   saved {ckpt_dir / 'best.pt'}")

        if cfg.task == "toy":
            show_toy_samples(model, val_set, device)

    # ---- final quality ----------------------------------------------------
    if cfg.task == "multi30k":
        report_bleu(model, val_set, tokenizer, device, cfg)

    print(f"\ndone in {time.time() - start:.0f}s  best val loss {best_val:.4f}")


@torch.no_grad()
def show_toy_samples(model, dataset, device, n: int = 3) -> None:
    """Decode a few toy examples. Loss can look fine while output is garbage."""
    from collate import pad_to_max
    from toy_data import decode_toy

    pairs = [dataset[i] for i in range(n)]
    src = pad_to_max([p[0] for p in pairs]).to(device)
    outputs = greedy_decode(model, src, max_len=16, device=device)

    print("   samples:")
    for (s, t), out in zip(pairs, outputs):
        want = decode_toy(t)
        got = decode_toy(out)
        mark = "ok " if want == got else "BAD"
        print(f"     {mark} {decode_toy(s)} -> want {want}  got {got}")


@torch.no_grad()
def report_bleu(model, dataset, tokenizer, device, cfg, limit: int = 500) -> None:
    """Corpus BLEU on a slice of validation."""
    from collate import pad_to_max

    model.eval()
    candidates, references = [], []

    for start in range(0, min(limit, len(dataset)), cfg.batch_size):
        pairs = [dataset[i] for i in range(start, min(start + cfg.batch_size, limit))]
        src = pad_to_max([p[0] for p in pairs]).to(device)
        outputs = greedy_decode(model, src, max_len=cfg.max_len, device=device)

        for (_, tgt), out in zip(pairs, outputs):
            gold = [t for t in tgt.tolist() if t not in (PAD_ID, BOS_ID, EOS_ID)]
            candidates.append(tokenizer.decode(out).split())
            references.append(tokenizer.decode(gold).split())

    print(f"\nBLEU on {len(candidates)} val sentences: "
          f"{corpus_bleu(candidates, references):.2f}")
    for i in range(min(3, len(candidates))):
        print(f"  got  {' '.join(candidates[i])}")
        print(f"  want {' '.join(references[i])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["toy", "multi30k"], default="toy")
    p.add_argument("--overfit", action="store_true", help="run the one-batch gate")
    p.add_argument("--overfit-steps", type=int, default=400)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-samples", type=int, default=2000)

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=8000)

    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--log-every", type=int, default=50)

    cfg = p.parse_args()
    if cfg.task == "multi30k":
        # real data needs a longer warmup and more capacity than the toy task
        if "--warmup" not in __import__("sys").argv:
            cfg.warmup = 4000
    return cfg


if __name__ == "__main__":
    train(parse_args())
