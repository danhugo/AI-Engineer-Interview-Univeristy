# Setup

## Why this file exists

Local git is **the source of truth**; the box is disposable and synced with
`rsync`. This file records the exact env so the box can be rebuilt from scratch.

The box does have git (2.34.1) and can reach GitHub — we just don't use it
there. rsync keeps the edit-test loop fast: no commit needed to try something.

## The box (Elice on-demand)

| | |
|---|---|
| Host | `central-01.tcp.tunnel.elice.io` port **34224**, user `elicer` |
| Key | `~/elice-cloud-ondemand-73195e20-1640-4600-b8ee-953d2ca6b2c3.pem` (chmod 600) |
| GPU | **2× A100 80GB PCIe**, compute capability **8.0** (Ampere) |
| Driver | 535.216.03 → CUDA **12.2** ceiling |
| CPU / RAM | 32 cores / 384 GB |
| Disk | 128 GB (16 GB used by the model) |
| Python | 3.10.14, venv at `~/mini-llm/.venv` |
| Remote project dir | `~/mini-llm` |

There is a second instance on port **45664** with only 1 GPU. Use 34224.

Ports change when the instance restarts. Override with `ELICE_PORT=... ./sync.sh push`.

## Installed

```
torch          2.6.0+cu124     # cu124, not cu128 — driver caps at 12.2
transformers   5.16.1
flash-attn     2.7.4.post1     # see the ABI trap below
accelerate, huggingface_hub, ninja, einops
```

Model: `Qwen/Qwen3-8B` in `~/mini-llm/hf` (16 GB), via `HF_HOME=~/mini-llm/hf`.

## The flash-attn ABI trap

**Do not `pip install flash-attn`** — it builds from source and takes over an hour.
Install a prebuilt wheel from the [GitHub releases](https://github.com/Dao-AILab/flash-attention/releases).

The wheel must match torch's C++ ABI. `torch 2.6.0+cu124` reports:

```python
torch._C._GLIBCXX_USE_CXX11_ABI  # False  -> want a cxx11abiFALSE wheel
```

**But the 2.8.x `cxx11abiFALSE` wheels are mis-built** — they still require the
new ABI and fail at import with:

```
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_string...
```

`__cxx11...basic_string` in the symbol = new ABI. Old ABI mangles as `...ESs`.

**2.7.4.post1 `cxx11abiFALSE` is correct.** This is the working install:

```bash
~/mini-llm/.venv/bin/pip install --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

To check any wheel's ABI without installing it, look at the undefined symbol:

```bash
nm --dynamic --undefined-only flash_attn_2_cuda*.so | grep 3c105ErrorC
# ...SourceLocationESs        -> OLD abi (matches torch 2.6+cu124)
# ...__cxx1112basic_string    -> NEW abi (won't link)
```

## Rebuild from scratch

```bash
python3 -m venv ~/mini-llm/.venv
~/mini-llm/.venv/bin/pip install --upgrade pip setuptools wheel
~/mini-llm/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/mini-llm/.venv/bin/pip install transformers accelerate huggingface_hub ninja einops
# flash-attn: use the exact wheel URL above
HF_HOME=~/mini-llm/hf ~/mini-llm/.venv/bin/hf download Qwen/Qwen3-8B
```

## Daily use

```bash
./sync.sh push              # local -> box (mirrors: --delete removes box-only files)
./sync.sh py run.py         # push, then run run.py on the box
./sync.sh run pytest -q     # push, then run tests
./sync.sh gpu               # nvidia-smi
./sync.sh shell             # interactive ssh
./sync.sh diff              # preview what pull would change
./sync.sh pull              # box -> local (never deletes; --update keeps newer local files)
```

**Direction matters.** `push` mirrors, so anything edited only on the box is
destroyed. `pull` is guarded with `--update` so an older copy on the box cannot
clobber a file you just edited locally — a mistake that already cost one edit
before the guard existed. Run `diff` first when unsure.

## Verified working

```
flash_attn 2.7.4.post1
flash out: (2, 512, 32, 128) torch.bfloat16     # GQA 32q/8kv, head_dim 128
max abs diff vs SDPA: 0.00195                   # bf16 noise, correct
GPUs: 2 | NVIDIA A100 80GB PCIe
```
