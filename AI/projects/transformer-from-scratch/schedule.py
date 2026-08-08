"""Learning rate schedules from scratch. STUDY ONLY.

The real pipeline uses torch.optim.lr_scheduler.LambdaLR with the same formula.
test_schedule.py asserts they match.

Why warmup is not optional
--------------------------
Adam divides by sqrt(v), the running mean of squared gradients. At step 1, v is
built from a single sample, so that estimate is noise. Dividing by the square
root of noise produces huge, badly-aimed steps at exactly the moment the model
is most fragile.

Warmup starts the learning rate near zero and ramps it up. By the time the rate
is high, v has seen enough gradients to be a real estimate. Skip warmup and a
transformer will often diverge in the first few hundred steps, or land in a bad
region it never leaves.

Noam schedule (Attention Is All You Need, section 5.3)
------------------------------------------------------
    lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

Two branches that cross at step == warmup:

    step < warmup:  step * warmup^-1.5   grows linearly
    step > warmup:  step^-0.5            decays as inverse sqrt

The d_model^-0.5 factor makes the peak rate smaller for wider models, which is
why the same schedule works across model sizes without retuning.
"""
import math
from typing import Callable


def noam_lambda(d_model: int, warmup_steps: int) -> Callable[[int], float]:
    """Return a step -> multiplier function for the Noam schedule.

    The returned callable is what LambdaLR expects: it multiplies the
    optimizer's base lr. So set the optimizer's lr to 1.0 and let this produce
    the actual rate.

    Args:
        d_model: model width. Wider model -> smaller peak rate.
        warmup_steps: where the two branches cross, i.e. the peak.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")

    def fn(step: int) -> float:
        # LambdaLR calls with step=0 first; step^-0.5 is undefined there, so
        # clamp to 1. The lost step is irrelevant against a warmup of hundreds.
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    return fn


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """The Noam rate at a given step, as an absolute number.

    Same formula as noam_lambda, exposed directly so it can be plotted or
    checked without constructing an optimizer.
    """
    return noam_lambda(d_model, warmup_steps)(step)


class NoamScheduler:
    """Applies the Noam schedule to an optimizer, from scratch.

    Mirrors torch's LambdaLR: call step() after each optimizer.step(), and it
    overwrites param_group["lr"] for the next update.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.fn = noam_lambda(d_model, warmup_steps)
        self._step = 0
        # torch stores the starting lr as base_lr and multiplies it by the
        # lambda. Capture it once so repeated steps do not compound.
        self.base_lrs = [g["lr"] for g in self._groups()]
        self.step()  # apply the step-0 rate immediately, like LambdaLR does

    def _groups(self):
        # supports both torch optimizers and the from-scratch ones in optim.py,
        # which expose a flat .lr instead of param_groups
        if hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups
        return [{"lr": self.optimizer.lr}]

    def get_last_lr(self) -> list[float]:
        return [base * self.fn(self._step) for base in self.base_lrs]

    def step(self) -> None:
        mult = self.fn(self._step)
        if hasattr(self.optimizer, "param_groups"):
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base * mult
        else:
            self.optimizer.lr = self.base_lrs[0] * mult
        self._step += 1
