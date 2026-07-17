# Early Stopping with Validation Loss - Interview Knowledge Sheet

## Intuition

Training loss usually keeps going down as the model memorizes the training set.

Validation loss tells us whether that extra training still helps unseen data.

Early stopping says:

> Keep training while validation loss improves. Stop after it has not improved for a while.

It is a regularization method because it limits how far optimization can adapt to the training set.

---

## 1. Train vs Validation Loss

Training loss measures fit on examples the optimizer sees.

Validation loss measures fit on held-out examples.

A common pattern:

| Phase | Training loss | Validation loss | Meaning |
|-------|---------------|-----------------|---------|
| Early training | down | down | learning useful structure |
| Around best epoch | down | lowest | best generalization found so far |
| Overfitting | down | up or flat | model is fitting training-specific noise |

The best model is often not the final epoch. It is the epoch with the best validation metric.

---

## 2. Basic Rule

For validation loss, lower is better.

Track the best value so far:

$$
L_\text{best} = \min_{t \leq T} L_\text{val}^{(t)}
$$

An epoch improves if:

$$
L_\text{val}^{(t)} < L_\text{best} - \text{min\_delta}
$$

If there are too many epochs without improvement, stop.

---

## 3. Patience

`patience` is the number of unimproved epochs allowed before stopping.

Example with `patience = 2`:

```
val_loss = [0.90, 0.72, 0.70, 0.71, 0.73]
best epoch = 2
epochs without improvement = 2
stop after epoch 4
```

Patience prevents stopping because of one noisy validation measurement.

---

## 4. min_delta

`min_delta` ignores tiny changes.

If `min_delta = 0.01`, then `0.700 -> 0.695` is not counted as an improvement because the gain is only `0.005`.

Use it when validation loss jitters by small amounts.

---

## 5. Restore Best Weights

Stopping decides when to end training.

Restoring decides which weights to keep.

Usually, you want the weights from the best validation epoch:

```
best_epoch = argmin(validation_loss)
final_weights = checkpoint_from(best_epoch)
```

Without restoration, the final weights may come from later overfitted epochs.

---

## 6. Validation Set Discipline

Early stopping uses the validation set to make training decisions.

That means the validation set is no longer a completely untouched estimate.

Use a separate test set once at the end for the final unbiased evaluation.

---

## Interview Gotchas

- Early stopping is based on validation loss, not training loss.
- It is a form of regularization.
- `patience` handles noisy validation curves.
- `min_delta` defines how large an improvement must be.
- Save or restore the best weights, not just the last weights.
- Keep a separate test set for final evaluation.

---

## References

- Keras `EarlyStopping`: https://keras.io/api/callbacks/early_stopping/
- Prechelt, "Early Stopping - But When?": https://doi.org/10.1007/3-540-49430-8_3
