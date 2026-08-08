"""Adam and AdamW from scratch. STUDY ONLY.

The real pipeline (train.py) uses torch.optim.AdamW. This file exists to show
what that call is doing. test_optim.py asserts both match torch step-for-step.

Intuition
---------
Plain SGD uses one learning rate for every parameter. That is a bad deal when
gradients have wildly different scales: a rate small enough for the loud
parameters is far too small for the quiet ones.

Adam gives each parameter its own effective rate by tracking two running
averages of its gradient:

    m  first moment,  the mean       -> which way it has been going
    v  second moment, the mean square -> how big the steps have been

The update divides by sqrt(v), so a parameter with consistently large gradients
gets a smaller step and a parameter with tiny gradients gets a larger one.

    m_t = b1 * m_{t-1} + (1 - b1) * g
    v_t = b2 * v_{t-1} + (1 - b2) * g^2
    step = lr * m_hat / (sqrt(v_hat) + eps)

Bias correction
---------------
m and v start at zero, so early on they are biased toward zero — at t=1 with
b1=0.9, m is only 10% of the true gradient. Dividing by (1 - b1^t) undoes that.
The correction matters most in the first steps and fades to nothing later.

Adam vs AdamW
-------------
Adam implements weight decay by adding wd * param to the gradient. That decay
then flows through the m/v machinery and gets rescaled by sqrt(v), so
parameters with small gradients get decayed much harder than intended.

AdamW decouples it: the decay is applied straight to the parameter, outside the
adaptive step. This is why AdamW is the default for transformers.
"""
import math

import torch
from torch import Tensor


class Adam:
    """Adam with L2 regularization folded into the gradient (the original).

    Args:
        params: iterable of tensors with requires_grad=True
        lr: learning rate
        betas: (b1, b2) decay rates for the first and second moment
        eps: added to the denominator for numerical stability
        weight_decay: L2 penalty, added to the gradient (coupled)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = [p for p in params]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # step count is global, not per-parameter: bias correction depends on
        # how many updates have happened, which is the same for all of them.
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        # computed once per step, not per parameter
        bias_c1 = 1 - self.b1 ** self.t
        bias_c2 = 1 - self.b2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            if self.weight_decay != 0:
                # coupled: decay enters through the gradient, so it will be
                # rescaled by sqrt(v) below. This is the part AdamW fixes.
                g = g.add(p, alpha=self.weight_decay)

            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            m_hat = self.m[i] / bias_c1
            v_hat = self.v[i] / bias_c2

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


class AdamW(Adam):
    """Adam with decoupled weight decay.

    Only one line differs from Adam: the decay is applied directly to the
    parameter and never touches m or v.
    """

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bias_c1 = 1 - self.b1 ** self.t
        bias_c2 = 1 - self.b2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            if self.weight_decay != 0:
                # decoupled: shrink the parameter directly, before the adaptive
                # step. torch scales this by lr, so we match that.
                p.mul_(1 - self.lr * self.weight_decay)

            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            m_hat = self.m[i] / bias_c1
            v_hat = self.v[i] / bias_c2

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


def param_groups_with_decay(model, weight_decay: float) -> list[dict]:
    """Split parameters into decay and no-decay groups.

    Weight decay pulls parameters toward zero. That is sensible for weight
    matrices, where it limits how large any single connection can grow. It is
    wrong for biases and LayerNorm gain/shift, which are offsets — shrinking
    them toward zero just fights whatever the layer learned.

    Rule of thumb: decay anything with 2+ dimensions, skip 1D parameters.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
