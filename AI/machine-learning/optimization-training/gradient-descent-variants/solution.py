"""Reference solutions for Gradient Descent Variants."""

import numpy as np


def np_mse_gradient(X, y, theta):
    """Return the MSE gradient for a batch."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    pred = X @ theta
    return (2 / X.shape[0]) * X.T @ (pred - y)


def np_batch_gradient_step(X, y, theta, lr):
    """Return theta after one full-batch gradient descent step."""
    theta = np.asarray(theta, dtype=float)
    return theta - lr * np_mse_gradient(X, y, theta)


def np_make_minibatches(X, y, batch_size, shuffle=False, seed=0):
    """Return a list of (X_batch, y_batch) pairs."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    indices = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
    batches = []
    for start in range(0, X.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        batches.append((X[batch_idx], y[batch_idx]))
    return batches


def np_minibatch_epoch(X, y, theta, lr, batch_size, shuffle=False, seed=0):
    """Run one epoch of mini-batch gradient descent and return theta."""
    theta = np.asarray(theta, dtype=float)
    for xb, yb in np_make_minibatches(X, y, batch_size, shuffle, seed):
        theta = np_batch_gradient_step(xb, yb, theta, lr)
    return theta


def np_sgd_epoch(X, y, theta, lr, shuffle=False, seed=0):
    """Run one epoch of one-sample SGD and return theta."""
    return np_minibatch_epoch(X, y, theta, lr, batch_size=1, shuffle=shuffle, seed=seed)


def torch_minibatch_train_one_epoch(model, loader, optimizer, loss_fn):
    """Train a PyTorch model for one epoch and return mean loss."""
    total_loss = 0.0
    total_examples = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        batch_size = xb.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
    return total_loss / total_examples


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


X_REG = np.array([[1.0, 0.0],
                  [1.0, 1.0],
                  [1.0, 2.0],
                  [1.0, 3.0]])
y_REG = np.array([1.0, 3.0, 5.0, 7.0])


def test_numpy():
    theta0 = np.array([0.0, 0.0])
    grad = np_mse_gradient(X_REG, y_REG, theta0)
    check(np.allclose(grad, [-8.0, -17.0]), f"batch gradient wrong: {grad}")
    print("PASS  np_mse_gradient")

    theta1 = np_batch_gradient_step(X_REG, y_REG, theta0, lr=0.01)
    check(np.allclose(theta1, [0.08, 0.17]), f"batch step wrong: {theta1}")
    print("PASS  np_batch_gradient_step")

    batches = np_make_minibatches(X_REG, y_REG, batch_size=2, shuffle=False)
    check(len(batches) == 2, f"batch count wrong: {len(batches)}")
    check(np.allclose(batches[0][0], X_REG[:2]), "first batch should keep order")
    print("PASS  np_make_minibatches")

    theta_mb = np_minibatch_epoch(X_REG, y_REG, theta0, lr=0.01, batch_size=2)
    check(np.allclose(theta_mb, [0.1577, 0.3341]), f"minibatch epoch wrong: {theta_mb}")
    print("PASS  np_minibatch_epoch")

    theta_sgd = np_sgd_epoch(X_REG, y_REG, theta0, lr=0.01)
    check(np.allclose(theta_sgd, [0.29701264, 0.61581392]), f"sgd epoch wrong: {theta_sgd}")
    print("PASS  np_sgd_epoch")


def test_torch():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch ({e})")
        return

    torch.manual_seed(0)
    X = torch.tensor(X_REG, dtype=torch.float32)
    y = torch.tensor(y_REG, dtype=torch.float32).view(-1, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    before = float(loss_fn(model(X), y).detach())
    mean_loss = torch_minibatch_train_one_epoch(model, loader, optimizer, loss_fn)
    after = float(loss_fn(model(X), y).detach())
    check(mean_loss > 0.0, "mean loss should be positive")
    check(after < before, f"training should reduce loss: before={before}, after={after}")
    print("PASS  torch_minibatch_train_one_epoch")


if __name__ == "__main__":
    test_numpy()
    test_torch()
    print("All tests passed.")
