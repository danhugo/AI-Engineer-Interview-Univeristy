"""Tests for the data pipeline: collate shift, toy data, checkpointing.

The collate tests matter most. A wrong target shift is the classic silent
seq2seq bug — loss looks healthy while the model learns to echo its input.

Run: python -m pytest test_pipeline.py -v
"""
import torch
import pytest

from collate import pad_to_max, seq2seq_collate
from toy_data import (
    BOS_ID, EOS_ID, PAD_ID, DIGIT_OFFSET,
    ReverseDigitsDataset, decode_toy,
)


class TestPadding:
    def test_pads_to_longest(self):
        out = pad_to_max([torch.tensor([1, 2, 3]), torch.tensor([4])])
        assert out.shape == (2, 3)
        assert out[1].tolist() == [4, PAD_ID, PAD_ID]

    def test_no_padding_when_equal_length(self):
        out = pad_to_max([torch.tensor([1, 2]), torch.tensor([3, 4])])
        assert (out != PAD_ID).all()


class TestCollateShift:
    def test_tgt_in_and_out_are_offset_by_one(self):
        # <bos> 7 8 9 <eos>
        tgt = torch.tensor([BOS_ID, 7, 8, 9, EOS_ID])
        batch = seq2seq_collate([(torch.tensor([1, 2]), tgt)])

        assert batch["tgt_in"][0].tolist() == [BOS_ID, 7, 8, 9]
        assert batch["tgt_out"][0].tolist() == [7, 8, 9, EOS_ID]

    def test_decoder_input_starts_with_bos(self):
        tgt = torch.tensor([BOS_ID, 5, EOS_ID])
        batch = seq2seq_collate([(torch.tensor([1]), tgt)])
        assert batch["tgt_in"][0, 0].item() == BOS_ID

    def test_decoder_target_never_contains_bos(self):
        """If <bos> appears in tgt_out the model is being taught to emit it."""
        tgt = torch.tensor([BOS_ID, 5, 6, EOS_ID])
        batch = seq2seq_collate([(torch.tensor([1]), tgt)])
        assert BOS_ID not in batch["tgt_out"][0].tolist()

    def test_target_ends_with_eos(self):
        """Without <eos> in tgt_out the model never learns to stop."""
        tgt = torch.tensor([BOS_ID, 5, 6, EOS_ID])
        batch = seq2seq_collate([(torch.tensor([1]), tgt)])
        assert batch["tgt_out"][0, -1].item() == EOS_ID

    def test_shapes_agree(self):
        pairs = [
            (torch.tensor([1, 2, 3]), torch.tensor([BOS_ID, 4, 5, EOS_ID])),
            (torch.tensor([1]), torch.tensor([BOS_ID, 9, EOS_ID])),
        ]
        batch = seq2seq_collate(pairs)
        assert batch["tgt_in"].shape == batch["tgt_out"].shape
        assert batch["src"].shape[0] == 2

    def test_shift_is_not_identity(self):
        """Guards against the copy bug: input and target must differ."""
        tgt = torch.tensor([BOS_ID, 1, 2, 3, EOS_ID])
        batch = seq2seq_collate([(torch.tensor([1]), tgt)])
        assert not torch.equal(batch["tgt_in"], batch["tgt_out"])

    def test_ragged_batch_pads_both_sides(self):
        pairs = [
            (torch.tensor([1, 2, 3, 4]), torch.tensor([BOS_ID, 5, 6, 7, EOS_ID])),
            (torch.tensor([1]), torch.tensor([BOS_ID, 9, EOS_ID])),
        ]
        batch = seq2seq_collate(pairs)
        assert batch["src"][1].tolist() == [1, PAD_ID, PAD_ID, PAD_ID]
        # short target is padded on the right in both views
        assert batch["tgt_out"][1, -1].item() == PAD_ID


class TestToyData:
    def test_target_is_reverse_of_source(self):
        ds = ReverseDigitsDataset(n_samples=20, seed=0)
        for src, tgt in ds:
            body = tgt[1:-1]  # strip bos/eos
            assert torch.equal(body, src.flip(0))

    def test_target_is_wrapped_in_bos_eos(self):
        ds = ReverseDigitsDataset(n_samples=5, seed=0)
        for _, tgt in ds:
            assert tgt[0].item() == BOS_ID
            assert tgt[-1].item() == EOS_ID

    def test_digits_are_offset_past_specials(self):
        """Digit tokens must not collide with pad/unk/bos/eos."""
        ds = ReverseDigitsDataset(n_samples=20, seed=0)
        for src, _ in ds:
            assert (src >= DIGIT_OFFSET).all()

    def test_is_deterministic(self):
        a = ReverseDigitsDataset(n_samples=10, seed=7)
        b = ReverseDigitsDataset(n_samples=10, seed=7)
        for (s1, t1), (s2, t2) in zip(a, b):
            assert torch.equal(s1, s2) and torch.equal(t1, t2)

    def test_different_seeds_differ(self):
        a = ReverseDigitsDataset(n_samples=10, seed=1)
        b = ReverseDigitsDataset(n_samples=10, seed=2)
        assert not all(torch.equal(x[0], y[0]) for x, y in zip(a, b))

    def test_decode_roundtrip(self):
        ds = ReverseDigitsDataset(n_samples=5, seed=0)
        src, tgt = ds[0]
        assert decode_toy(tgt) == decode_toy(src)[::-1]

    def test_decode_stops_at_eos(self):
        ids = [BOS_ID, DIGIT_OFFSET + 1, EOS_ID, DIGIT_OFFSET + 9]
        assert decode_toy(ids) == "1"

    def test_rejects_bad_lengths(self):
        with pytest.raises(ValueError):
            ReverseDigitsDataset(n_samples=1, min_len=5, max_len=2)


class TestModelIntegration:
    def test_forward_produces_expected_logit_shape(self):
        from transformer import Transformer
        from toy_data import TOY_VOCAB_SIZE

        model = Transformer(
            vocab_size=TOY_VOCAB_SIZE, padding_idx=PAD_ID,
            num_encoder_layers=1, num_decoder_layers=1,
            d_model=32, d_ff=64, num_heads=2, max_seq_len=32, drop_out=0.0,
        )
        ds = ReverseDigitsDataset(n_samples=4, seed=0)
        batch = seq2seq_collate([ds[i] for i in range(4)])
        logits = model(batch["src"], batch["tgt_in"])

        assert logits.shape == (
            4, batch["tgt_in"].shape[1], TOY_VOCAB_SIZE
        )

    def test_embedding_scale_is_applied(self):
        from transformer import Transformer
        import math

        model = Transformer(
            vocab_size=14, padding_idx=PAD_ID,
            num_encoder_layers=1, num_decoder_layers=1,
            d_model=64, d_ff=64, num_heads=2, max_seq_len=16, drop_out=0.0,
        )
        assert math.isclose(model.embed_scale, math.sqrt(64))

    def test_causal_mask_blocks_future(self):
        """Changing a later target token must not change an earlier output."""
        from transformer import Transformer

        torch.manual_seed(0)
        model = Transformer(
            vocab_size=14, padding_idx=PAD_ID,
            num_encoder_layers=1, num_decoder_layers=1,
            d_model=32, d_ff=64, num_heads=2, max_seq_len=16, drop_out=0.0,
        ).eval()

        src = torch.tensor([[4, 5, 6]])
        tgt_a = torch.tensor([[BOS_ID, 7, 8, 9]])
        tgt_b = torch.tensor([[BOS_ID, 7, 8, 11]])  # only last position differs

        with torch.no_grad():
            out_a = model(src, tgt_a)
            out_b = model(src, tgt_b)

        # positions 0..2 cannot see position 3, so they must be identical
        assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)
        # position 3 did change
        assert not torch.allclose(out_a[:, 3], out_b[:, 3], atol=1e-5)


class TestCheckpoint:
    def test_roundtrip_restores_weights_and_step(self, tmp_path):
        from checkpoint import load_checkpoint, save_checkpoint
        from transformer import Transformer

        def make():
            return Transformer(
                vocab_size=14, padding_idx=PAD_ID,
                num_encoder_layers=1, num_decoder_layers=1,
                d_model=32, d_ff=64, num_heads=2, max_seq_len=16, drop_out=0.0,
            )

        model = make()
        opt = torch.optim.AdamW(model.parameters(), lr=1.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, sched, step=123, epoch=4, config={"d_model": 32})

        restored = make()
        meta = load_checkpoint(path, restored)

        assert meta["step"] == 123
        assert meta["epoch"] == 4
        assert meta["config"]["d_model"] == 32
        for a, b in zip(model.parameters(), restored.parameters()):
            assert torch.equal(a, b)

    def test_optimizer_state_survives(self, tmp_path):
        """Adam's m and v must come back, or resume restarts unstable."""
        from checkpoint import load_checkpoint, save_checkpoint
        from transformer import Transformer

        def make():
            return Transformer(
                vocab_size=14, padding_idx=PAD_ID,
                num_encoder_layers=1, num_decoder_layers=1,
                d_model=32, d_ff=64, num_heads=2, max_seq_len=16, drop_out=0.0,
            )

        model = make()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

        # take a real step so optimizer state is non-empty
        model(torch.tensor([[4, 5]]), torch.tensor([[BOS_ID, 6]])).sum().backward()
        opt.step()

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, opt, sched, step=1, epoch=1, config={})

        restored = make()
        new_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt, lambda s: 1.0)
        load_checkpoint(path, restored, new_opt, new_sched)

        assert len(new_opt.state_dict()["state"]) > 0
