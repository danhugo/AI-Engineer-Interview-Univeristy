# Batch Normalization Forward Pass — Interview Knowledge Sheet

## Intuition

Batch normalization normalizes layer activations using mini-batch statistics, then gives the model a learned scale and shift.

The shape is:

```
normalize -> scale -> shift
```

It helps keep intermediate activations in a range that is easier to optimize, while `gamma` and `beta` let the layer recover any useful scale and offset.

---

## 1. Training Forward Pass

For a mini-batch `X` with shape `(batch, features)`, compute each feature's batch mean:

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i
$$

and variance:

$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
$$

Normalize:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Then scale and shift:

$$
y_i = \gamma \hat{x}_i + \beta
$$

---

## 2. Why `epsilon` Exists

`epsilon` is a small positive number added inside the square root:

$$
\sqrt{\sigma_B^2 + \epsilon}
$$

It prevents division by zero and improves numerical stability when a feature has tiny variance.

---

## 3. Gamma and Beta

If normalization always forced zero mean and unit variance, it could remove useful representations.

Batch norm adds learnable parameters:

- `gamma`: learned scale
- `beta`: learned shift

This lets the layer represent the identity transform when useful.

---

## 4. Running Statistics

Training uses mini-batch statistics and updates running estimates:

```python
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var = momentum * running_var + (1 - momentum) * batch_var
```

Inference uses running statistics instead of the current batch:

$$
y = \gamma \frac{x - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}} + \beta
$$

This makes inference deterministic and independent of which examples happen to be batched together.

---

## 5. Axis Details

For dense inputs shaped `(N, D)`, normalize over `N` separately for each feature.

For convolutional activations shaped `(N, C, H, W)`, batch norm usually normalizes each channel over `N`, `H`, and `W`.

Getting the axes wrong is a common implementation bug.

---

## 6. Interview Gotchas

- Batch norm has different train and eval behavior.
- Training uses mini-batch mean and variance.
- Inference uses running mean and variance.
- `gamma` and `beta` are learnable.
- `epsilon` is for numerical stability.
- Batch norm can allow higher learning rates and often has a regularizing effect.
- Small batch sizes can make batch statistics noisy.

---

## References

- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift": https://arxiv.org/abs/1502.03167
- PyTorch `BatchNorm1d`: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- TensorFlow `BatchNormalization`: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
