"""Reference solutions for Gradient Checkpointing."""


def make_counted_block(width):
    """Return (block, counter), where block is a small torch.nn.Module."""
    import torch

    counter = {"calls": 0}

    class CountedBlock(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.ReLU(),
                torch.nn.Linear(width, width),
            )

        def forward(self, x):
            counter["calls"] += 1
            return self.net(x)

    return CountedBlock(width), counter


def run_without_checkpoint(block, x):
    """Return block(x) without checkpointing."""
    return block(x)


def run_with_checkpoint(block, x):
    """Return block(x) using torch.utils.checkpoint.checkpoint."""
    from torch.utils.checkpoint import checkpoint
    return checkpoint(block, x, use_reentrant=False)


def input_gradient(block, x, use_checkpoint):
    """Return the gradient of mean(block(x) ** 2) with respect to x."""
    x = x.clone().detach().requires_grad_(True)
    out = run_with_checkpoint(block, x) if use_checkpoint else run_without_checkpoint(block, x)
    loss = out.pow(2).mean()
    loss.backward()
    return x.grad


def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL  {msg}")


def run_tests():
    try:
        import torch
    except Exception as e:
        print(f"SKIP  PyTorch tests ({e})")
        return

    torch.manual_seed(7)
    block, counter = make_counted_block(width=4)
    x = torch.randn(3, 4, requires_grad=True)

    counter["calls"] = 0
    y = run_without_checkpoint(block, x)
    y.pow(2).mean().backward()
    check(counter["calls"] == 1, f"normal forward calls wrong: {counter['calls']}")
    print("PASS  no-checkpoint call count")

    block.zero_grad(set_to_none=True)
    x.grad = None
    counter["calls"] = 0
    y = run_with_checkpoint(block, x)
    y.pow(2).mean().backward()
    check(counter["calls"] == 2, f"checkpoint should recompute during backward: {counter['calls']}")
    print("PASS  checkpoint recomputes")

    torch.manual_seed(7)
    block_a, _ = make_counted_block(width=4)
    torch.manual_seed(7)
    block_b, _ = make_counted_block(width=4)
    x0 = torch.randn(3, 4)
    grad_a = input_gradient(block_a, x0, use_checkpoint=False)
    grad_b = input_gradient(block_b, x0, use_checkpoint=True)
    check(torch.allclose(grad_a, grad_b, atol=1e-6), "checkpoint gradient should match")
    print("PASS  gradient equivalence")


if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
