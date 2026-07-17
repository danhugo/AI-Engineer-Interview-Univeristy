# Weight Decay vs L2 Regularization — Interview Knowledge Sheet

## Intuition

Both methods discourage large weights, but they enter training in different places.

L2 regularization changes the loss:

$$
J(w) = L(w) + \frac{\lambda}{2}\|w\|_2^2
$$

Weight decay changes the parameter update by shrinking weights:

$$
w \leftarrow (1 - \eta\lambda)w - \eta\nabla L(w)
$$

For plain SGD, these are equivalent up to convention. For adaptive optimizers like Adam, they are not generally equivalent.

---

## 1. L2 Regularization

L2 adds a penalty to the objective:

$$
\frac{\lambda}{2}\sum_j w_j^2
$$

Its gradient is:

$$
\lambda w
$$

So an SGD update becomes:

$$
w_{t+1} = w_t - \eta(\nabla L(w_t) + \lambda w_t)
$$

Rearrange it:

$$
w_{t+1} = (1 - \eta\lambda)w_t - \eta\nabla L(w_t)
$$

That looks exactly like weight decay.

---

## 2. Weight Decay

Weight decay says:

```
first shrink the weight, then apply the loss gradient update
```

For SGD:

```python
w = w * (1 - lr * weight_decay) - lr * grad
```

This directly biases optimization toward smaller weights. It is often described as multiplying weights by a number slightly below `1` on every step.

---

## 3. Why They Match for SGD

With ordinary SGD, every part of the gradient is scaled by the same learning rate `eta`.

L2:

$$
w \leftarrow w - \eta(\nabla L + \lambda w)
$$

Weight decay:

$$
w \leftarrow w - \eta\nabla L - \eta\lambda w
$$

Same update.

This equivalence is easy to break once the optimizer transforms gradients.

---

## 4. Why They Differ for Adam

Adam rescales gradient coordinates using moving averages of first and second moments. If L2 is added to the gradient, the penalty term is also fed into Adam's adaptive machinery:

```python
adam_grad = grad + weight_decay * w
```

That means the shrinkage can depend on Adam's per-parameter scaling.

Decoupled weight decay, used by AdamW, keeps decay outside the adaptive gradient update:

```python
w = w * (1 - lr * weight_decay)
w = adam_update_using_only_loss_gradient(w, grad)
```

The Loshchilov and Hutter AdamW paper highlights this distinction: L2 and weight decay are equivalent for standard SGD, but not for adaptive gradient algorithms.

---

## 5. Bias and Normalization Parameters

In deep learning, weight decay is usually applied to matrix/kernel weights, not to:

- bias terms
- batch norm scale and shift
- layer norm scale and shift

Those parameters do not represent the same kind of model capacity as large feature weights.

---

## 6. Interview Gotchas

- L2 regularization is a loss penalty.
- Weight decay is an optimizer update rule.
- They are equivalent for plain SGD under common scaling conventions.
- They differ for adaptive optimizers unless weight decay is decoupled.
- AdamW is Adam with decoupled weight decay.
- Check whether a framework's `weight_decay` means coupled L2 or decoupled decay.
- Bias and normalization parameters are commonly excluded.

---

## References

- Loshchilov and Hutter, "Decoupled Weight Decay Regularization": https://arxiv.org/abs/1711.05101
- PyTorch `AdamW`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- PyTorch optimizer parameter groups: https://pytorch.org/docs/stable/optim.html
