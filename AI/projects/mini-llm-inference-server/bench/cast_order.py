"""Micro-benchmark: does .cuda().to(dtype) vs .to(dtype).cuda() matter?

Run on the GPU box: python bench/cast_order.py
"""

import time
import torch
from torch import nn


def make_model():
    # Roughly Qwen3-0.6B-sized: a stack of linears, fp32 by default.
    return nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(20)])


def bench(fn, label, n=10):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        m = fn()
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n
    print(f"{label:30s} {elapsed * 1000:8.2f} ms/iter")
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available()

    cpu_model = make_model()  # fp32 on CPU, ~270MB

    bench(lambda: make_model().cuda().to(torch.bfloat16), "cuda() then to(bf16)")
    bench(lambda: make_model().to(torch.bfloat16).cuda(), "to(bf16) then cuda()")
