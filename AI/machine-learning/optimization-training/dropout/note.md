# Dropout — Interview Knowledge Sheet

## Intuition

Dropout randomly turns off some activations during training.

Instead of letting every neuron rely on the same neighbors every time, dropout forces the network to learn useful features under missing information:

```
training: some units are hidden
inference: all units are used
```

This reduces overfitting by making co-adaptation harder.

---

## 1. Forward Pass

For activation vector `x`, sample a binary mask:

$$
m_i \sim \text{Bernoulli}(q)
$$

where:

$$
q = 1 - p
$$

`p` is the drop probability and `q` is the keep probability.

With inverted dropout, the training output is:

$$
y = \frac{m \odot x}{q}
$$

The scale factor keeps the expected activation unchanged:

$$
E[y_i] = x_i
$$

---

## 2. Train vs Eval

Dropout is active only during training.

| Mode | Behavior |
|------|----------|
| train | sample mask, zero some activations, scale survivors |
| eval | return activations unchanged |

This is why forgetting `model.eval()` can make validation and inference noisy.

---

## 3. Why the Scaling Exists

Without scaling, if `p = 0.5`, the average activation would be cut roughly in half during training.

Inverted dropout fixes that during training:

$$
\frac{E[m_i x_i]}{q} = \frac{q x_i}{q} = x_i
$$

So inference does not need a special multiplication.

---

## 4. Where Dropout Is Used

Common placements:

- after dense layer activations
- inside MLP blocks
- after embeddings or attention outputs in transformers

Less common or more delicate placements:

- very early convolutional layers
- directly before batch normalization
- recurrent connections without specialized variants

---

## 5. Choosing `p`

`p` is a hyperparameter.

| Drop probability | Effect |
|------------------|--------|
| `0.0` | no dropout |
| `0.1` to `0.3` | light regularization |
| `0.5` | strong regularization |
| too high | underfitting |

The original dropout paper often describes hidden units being retained with probability `0.5`, which means `p = 0.5` in many modern APIs.

---

## 6. Interview Gotchas

- Dropout is a training-time regularizer.
- Modern libraries commonly use inverted dropout.
- In eval mode, dropout should be disabled.
- Dropout changes activations, not weights directly.
- The mask is random, so train-mode outputs are stochastic.
- `p` usually means drop probability in PyTorch.

---

## References

- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting": https://www.jmlr.org/papers/v15/srivastava14a.html
- PyTorch `Dropout`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- TensorFlow `Dropout`: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
