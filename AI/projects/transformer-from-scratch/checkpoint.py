"""Save and restore training state. DRAFT.

A checkpoint must hold everything needed to resume as if nothing happened. The
model weights alone are not enough:

  - optimizer: Adam's m and v took thousands of steps to build up. Restart
    without them and the first steps after resume are as unstable as step 1.
  - scheduler: the step count decides the learning rate. Lose it and you
    restart warmup on a converged model.
  - step / epoch: for logging and for knowing where the schedule is.

Saving only the weights is the most common resume bug, and it looks fine until
you notice the loss spikes at every restart.
"""
from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    config: dict,
    best_val_loss: float = float("inf"),
) -> None:
    """Write a resumable checkpoint.

    config is stored so the model can be rebuilt with the right shapes before
    load_state_dict is called — otherwise resuming means hand-matching
    d_model, layer counts, and vocab size from memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu"
) -> dict:
    """Restore into existing objects and return the bookkeeping fields.

    optimizer and scheduler are optional so the same function works for
    inference, where only the weights matter.
    """
    # weights_only=False because the payload holds a config dict, not just
    # tensors. Only load checkpoints you produced.
    state = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])

    return {
        "step": state["step"],
        "epoch": state["epoch"],
        "config": state["config"],
        "best_val_loss": state.get("best_val_loss", float("inf")),
    }
