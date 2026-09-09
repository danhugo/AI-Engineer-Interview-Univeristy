"""Where do our hidden states start diverging from HuggingFace?

Run: ./sync.sh py diag_layers.py

Reads per-layer output from both models on the same input and reports the gap.
The shape of the curve is the diagnosis:

  smooth geometric growth  -> bf16 rounding accumulating. Expected, harmless.
  a jump at one layer      -> a bug in that layer.
  divergence at layer 0    -> embedding or weight-loading problem.

Useful again in later stages: a paged-KV or TP bug shows up here the same way.
"""

import torch

from common import build_long_ids, load_models, path

SEQ_LEN = 2048


def collect_layer_outputs(model, layers, ids, positions=None):
    """Run the model, returning each decoder layer's output hidden states."""
    caught = {}

    def hook(idx):
        def fn(_mod, _inp, out):
            # HF layers return a tuple; ours returns a bare tensor.
            caught[idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            if positions is None:
                model(ids)
            else:
                model(ids, positions)
    finally:
        for h in handles:
            h.remove()
    return [caught[i] for i in range(len(layers))]


def main():
    tokenizer, hf, ours = load_models(path)
    ids = build_long_ids(tokenizer, SEQ_LEN)
    positions = torch.arange(ids.shape[1], device=ids.device)

    ref = collect_layer_outputs(hf, hf.model.layers, ids)
    our = collect_layer_outputs(ours, ours.model.layers, ids, positions)

    print(f"seq_len {SEQ_LEN}, {len(ref)} layers\n")
    print(f"{'layer':>5}  {'max|diff|':>10}  {'rms scale':>10}  {'relative':>9}  {'growth':>7}")
    prev = None
    for i, (r, o) in enumerate(zip(ref, our)):
        d = (r - o).abs().max().item()
        scale = r.pow(2).mean().sqrt().item()  # RMS of the hidden state
        rel = d / scale
        growth = f"{d / prev:6.2f}x" if prev else "     --"
        print(f"{i:5d}  {d:10.4f}  {scale:10.3f}  {rel:9.2%}  {growth:>7}")
        prev = d if d > 0 else None

    # Embeddings should be bit-identical: same weights, same lookup.
    e_ref = hf.model.embed_tokens(ids).float()
    e_our = ours.model.embed_tokens(ids).float()
    print(f"\nembedding max|diff|: {(e_ref - e_our).abs().max().item():.3e}  (expect exactly 0)")


if __name__ == "__main__":
    main()
