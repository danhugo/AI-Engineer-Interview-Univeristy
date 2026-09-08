"""Drill into the layer where our hidden states jump away from HuggingFace.

Run: ./sync.sh py diag_layer16.py

diag_layers.py shows a ~2000x jump at layer 16. This narrows it to a sublayer
and prints the exact (position, dim) plus both models' values there.
"""

import torch

from run import build_long_ids, load_models, path

SEQ_LEN = 2048
LAYER = 16
SUBLAYERS = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")


def tensor_of(out):
    t = (out[0] if isinstance(out, tuple) else out).detach().float()
    return t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t


def capture(model, layer, ids, positions=None):
    """Grab the layer's input and each sublayer's output."""
    got = {}
    handles = [layer.register_forward_pre_hook(
        lambda _m, inp: got.__setitem__("layer_in", tensor_of(inp[0]))
    )]
    for name in SUBLAYERS:
        mod = getattr(layer, name)
        handles.append(mod.register_forward_hook(
            lambda _m, _i, out, n=name: got.__setitem__(n, tensor_of(out))
        ))
    # Inside the MLP: which projection introduces the gap?
    for name in ("gate_proj", "up_proj", "down_proj"):
        handles.append(getattr(layer.mlp, name).register_forward_hook(
            lambda _m, _i, out, n=name: got.__setitem__("mlp." + n, tensor_of(out))
        ))
    handles.append(layer.register_forward_hook(
        lambda _m, _i, out: got.__setitem__("layer_out", tensor_of(out))
    ))
    try:
        with torch.inference_mode():
            model(ids) if positions is None else model(ids, positions)
    finally:
        for h in handles:
            h.remove()
    return got


def main():
    tokenizer, hf, ours = load_models(path)
    ids = build_long_ids(tokenizer, SEQ_LEN)
    positions = torch.arange(ids.shape[1], device=ids.device)

    ref = capture(hf, hf.model.layers[LAYER], ids)
    our = capture(ours, ours.model.layers[LAYER], ids, positions)

    print(f"layer {LAYER}, seq_len {SEQ_LEN}\n")
    print(f"{'stage':>28}  {'max|diff|':>12}")
    stages = ("layer_in", "input_layernorm", "self_attn", "post_attention_layernorm",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp", "layer_out")
    for name in stages:
        if name not in ref or name not in our:
            print(f"{name:>28}  (not captured)")
            continue
        r_, o_ = ref[name], our[name]
        print(f"{name:>28}  {(r_ - o_).abs().max().item():12.4f}"
              f"   |HF|max {r_.abs().max().item():11.2f}   |ours|max {o_.abs().max().item():11.2f}")

    # Pinpoint the worst element in this layer's output.
    r, o = ref["layer_out"], our["layer_out"]
    d = (r - o).abs()
    flat = d.argmax().item()
    pos, dim = flat // d.shape[-1], flat % d.shape[-1]
    print(f"\nworst element: position {pos}, hidden dim {dim}")
    print(f"  HF:   {r[pos, dim].item():12.3f}")
    print(f"  ours: {o[pos, dim].item():12.3f}")
    print(f"  token id at that position: {ids[0, pos].item()}"
          f"  ({tokenizer.decode([ids[0, pos].item()])!r})")

    # Is the gap concentrated at one position, or spread out?
    per_pos = d.amax(dim=-1)
    top = per_pos.topk(min(8, per_pos.numel()))
    print(f"\nworst 8 positions: {top.indices.tolist()}")
    print(f"  their max diffs: {[round(v, 1) for v in top.values.tolist()]}")
    print(f"  median over all positions: {per_pos.median().item():.4f}")

    # And how large are the raw activations there? Some models grow "massive
    # activations" at a specific layer/dim; if only one side does, that's our gap.
    print(f"\nHF   |act| max at that dim: {r[:, dim].abs().max().item():12.3f}")
    print(f"ours |act| max at that dim: {o[:, dim].abs().max().item():12.3f}")


if __name__ == "__main__":
    main()
