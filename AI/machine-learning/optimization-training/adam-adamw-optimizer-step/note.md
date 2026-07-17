# Adam / AdamW Optimizer Step — Interview Knowledge Sheet

## Intuition

SGD uses the current gradient directly.

Adam keeps two running summaries:

- **First moment**: moving average of gradients, like momentum.
- **Second moment**: moving average of squared gradients, used to scale each parameter's step.

So Adam asks:

```
Which way have gradients been pointing?
How large have gradients usually been for this parameter?
```

Then it takes an adaptive step.

---

## 1. Adam State

At step `t`, with gradient `g_t`:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

`m` tracks direction.

`v` tracks squared gradient scale.

Typical defaults:

```python
betas = (0.9, 0.999)
eps = 1e-8
```

---

## 2. Bias Correction

At the beginning, `m` and `v` start at zero.

That makes early moving averages biased toward zero.

Adam corrects this:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

---

## 3. Adam Update

The Adam step is:

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Intuition:

- If gradients consistently point one way, `m_hat` grows.
- If a parameter has large historical gradients, `sqrt(v_hat)` shrinks its effective step.
- `eps` prevents division by zero.

---

## 4. AdamW

AdamW means Adam with **decoupled weight decay**.

Adam's adaptive update and weight decay are separate:

$$
\theta \leftarrow \theta - \eta \lambda \theta
$$

then:

$$
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}
$$

This differs from adding an L2 penalty into Adam's gradient. For adaptive optimizers, L2 regularization and weight decay are not equivalent.

---

## 5. PyTorch Pattern

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

For modern deep learning, use `AdamW` when you want Adam-style training with weight decay.

---

## 6. Interview Gotchas

- Adam is not just momentum; it also divides by a running RMS-like gradient scale.
- Bias correction matters most in early steps.
- `eps` is for numerical stability.
- AdamW decouples weight decay from the gradient moments.
- Do not call `optimizer.step()` before `loss.backward()`.

---

## References

- Adam paper: https://arxiv.org/abs/1412.6980
- Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
- PyTorch `Adam`: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
- PyTorch `AdamW`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
