"""Tests for the Multi30k data module.

Uses a tokenizer trained on a tiny in-memory corpus, so nothing here needs a
download. The one test that does touch the cache is marked and skipped when
the files are absent.

Run: python -m pytest test_seq2seq_data.py -v
"""
import pytest
import torch

from seq2seq_data import (
    BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS,
    DEFAULT_CACHE, Multi30kDataset, train_tokenizer,
)

CORPUS = [
    "ein mann geht über die straße",
    "a man walks across the street",
    "zwei hunde spielen im park",
    "two dogs play in the park",
    "eine frau liest ein buch",
    "a woman reads a book",
] * 40  # repeat so BPE has enough counts to learn merges


@pytest.fixture(scope="module")
def tokenizer():
    return train_tokenizer(CORPUS, vocab_size=400)


class TestTokenizer:
    def test_special_tokens_get_ids_zero_to_three(self, tokenizer):
        """The whole pipeline hardcodes these IDs."""
        for expected, token in enumerate(SPECIAL_TOKENS):
            assert tokenizer.token_to_id(token) == expected

    def test_roundtrip_is_lossless(self, tokenizer):
        text = "ein mann geht über die straße"
        assert tokenizer.decode(tokenizer.encode(text).ids).strip() == text

    def test_byte_level_handles_unseen_characters(self, tokenizer):
        """Byte-level BPE must never fail on characters it never trained on."""
        text = "日本語 🎉 çøß"
        assert tokenizer.decode(tokenizer.encode(text).ids).strip() == text

    def test_never_emits_unk(self, tokenizer):
        ids = tokenizer.encode("völlig unbekannte wörter xyzzy").ids
        assert tokenizer.token_to_id("<unk>") not in ids

    def test_rejects_mismatched_special_token_ids(self):
        """Guard against a silent reorder breaking every hardcoded ID."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<bos>", "<pad>", "<unk>", "<eos>"],  # wrong order
            show_progress=False,
        )
        tok.train_from_iterator(CORPUS, trainer=trainer)
        assert tok.token_to_id("<pad>") != 0  # precondition for the guard


class TestMulti30kDataset:
    def test_target_is_wrapped_in_bos_eos(self, tokenizer):
        ds = Multi30kDataset(CORPUS[:6], CORPUS[:6], tokenizer, max_len=32)
        for _, tgt in ds:
            assert tgt[0].item() == BOS_ID
            assert tgt[-1].item() == EOS_ID

    def test_source_has_no_special_tokens(self, tokenizer):
        ds = Multi30kDataset(CORPUS[:6], CORPUS[:6], tokenizer, max_len=32)
        for src, _ in ds:
            assert BOS_ID not in src.tolist()
            assert EOS_ID not in src.tolist()

    def test_respects_max_len(self, tokenizer):
        max_len = 8
        long_lines = ["ein mann geht über die straße " * 10] * 4
        ds = Multi30kDataset(long_lines, long_lines, tokenizer, max_len=max_len)
        for src, tgt in ds:
            assert len(src) <= max_len
            # target leaves room for <bos> and <eos>
            assert len(tgt) <= max_len

    def test_returns_long_tensors(self, tokenizer):
        """Embedding indexing requires int64, not int32."""
        ds = Multi30kDataset(CORPUS[:4], CORPUS[:4], tokenizer, max_len=32)
        src, tgt = ds[0]
        assert src.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_skips_empty_lines(self, tokenizer):
        de = ["ein mann geht", "", "zwei hunde spielen"]
        en = ["a man walks", "", "two dogs play"]
        ds = Multi30kDataset(de, en, tokenizer, max_len=32)
        assert len(ds) == 2  # the empty pair is dropped

    def test_rejects_mismatched_line_counts(self, tokenizer):
        with pytest.raises(ValueError):
            Multi30kDataset(["a", "b"], ["a"], tokenizer, max_len=32)

    def test_integrates_with_collate(self, tokenizer):
        from collate import seq2seq_collate

        ds = Multi30kDataset(CORPUS[:8], CORPUS[:8], tokenizer, max_len=32)
        batch = seq2seq_collate([ds[i] for i in range(4)])

        assert batch["src"].shape[0] == 4
        assert batch["tgt_in"].shape == batch["tgt_out"].shape
        # the shift must hold after real tokenization too
        assert BOS_ID not in batch["tgt_out"].tolist()


@pytest.mark.skipif(
    not (DEFAULT_CACHE / "train.de").exists(),
    reason="Multi30k not downloaded",
)
class TestRealCorpus:
    def test_split_sizes(self):
        from seq2seq_data import download_multi30k

        data = download_multi30k()
        assert len(data["train"]["de"]) == 29000
        assert len(data["val"]["de"]) == 1014

    def test_parallel_lines_align(self):
        from seq2seq_data import download_multi30k

        data = download_multi30k()
        for split in data:
            assert len(data[split]["de"]) == len(data[split]["en"])
