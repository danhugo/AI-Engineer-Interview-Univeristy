"""Multi30k de->en: download, tokenize, serve batches.

Uses the HuggingFace `tokenizers` library rather than the from-scratch BPE.
The README explains the algorithm; this file just uses a solid implementation.

Shared vocabulary
-----------------
One byte-level BPE trained over German and English together. Two reasons:

  - one embedding matrix instead of two, which matters when the corpus is only
    29k sentences and every row needs gradient
  - German and English share a lot of surface form (names, numbers, borrowed
    words, punctuation), so shared merges are not wasted

Byte-level means no <unk> can ever be produced, though <unk> stays in the
vocabulary as a declared special token.
"""
import gzip
import os
import urllib.request
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

RAW_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
SPLITS = {"train": "train", "val": "val", "test": "test_2016_flickr"}
DEFAULT_CACHE = Path(__file__).parent / ".cache" / "multi30k"


def download_multi30k(cache_dir: Path = DEFAULT_CACHE) -> dict[str, dict[str, list[str]]]:
    """Fetch the raw sentence files, one per split per language.

    Files are small (a few MB total) and cached, so this is a one-time cost.
    Returns {split: {"de": [...], "en": [...]}} with lines already stripped.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, list[str]]] = {}

    for split, stem in SPLITS.items():
        data[split] = {}
        for lang in ("de", "en"):
            local = cache_dir / f"{stem}.{lang}"
            if not local.exists():
                url = f"{RAW_URL}/{stem}.{lang}.gz"
                print(f"downloading {url}")
                with urllib.request.urlopen(url, timeout=60) as response:
                    raw = gzip.decompress(response.read())
                local.write_bytes(raw)
            lines = local.read_text(encoding="utf-8").splitlines()
            data[split][lang] = [line.strip() for line in lines]

        n_de, n_en = len(data[split]["de"]), len(data[split]["en"])
        if n_de != n_en:
            raise ValueError(f"{split}: {n_de} de lines vs {n_en} en lines")

    return data


def train_tokenizer(
    corpus: list[str], vocab_size: int = 8000, save_path: Path | None = None
):
    """Train a byte-level BPE over the combined de+en text.

    Args:
        corpus: every training sentence from both languages
        vocab_size: total, including the 256 byte tokens and 4 specials
        save_path: where to write tokenizer.json, if given
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # ByteLevel with add_prefix_space so a leading word is tokenized the same
    # whether or not it starts the sentence
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,  # order fixes the IDs at 0,1,2,3
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # the whole pipeline assumes these IDs
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            raise ValueError(f"{token} got id {actual}, expected {expected_id}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
    return tokenizer


class Multi30kDataset(Dataset):
    """Encoded (src, tgt) pairs for one split.

    src is the German sentence with no special tokens. tgt is the English
    sentence wrapped as <bos> ... <eos>. collate does the shift.

    Sentences longer than max_len are truncated rather than dropped, so the
    split size stays predictable.
    """

    def __init__(
        self,
        de_lines: list[str],
        en_lines: list[str],
        tokenizer,
        max_len: int = 64,
    ):
        if len(de_lines) != len(en_lines):
            raise ValueError("de and en must have the same number of lines")
        self.max_len = max_len

        # encode_batch is much faster than encoding one at a time
        de_encoded = tokenizer.encode_batch(de_lines)
        en_encoded = tokenizer.encode_batch(en_lines)

        self.pairs: list[tuple[Tensor, Tensor]] = []
        for de, en in zip(de_encoded, en_encoded):
            # leave room for <bos> and <eos> on the target
            src = torch.tensor(de.ids[:max_len], dtype=torch.long)
            body = en.ids[: max_len - 2]
            tgt = torch.tensor([BOS_ID] + body + [EOS_ID], dtype=torch.long)
            if len(src) == 0 or len(body) == 0:
                continue  # skip empty lines rather than emit a degenerate pair
            self.pairs.append((src, tgt))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.pairs[i]


def build_multi30k(
    vocab_size: int = 8000,
    max_len: int = 64,
    cache_dir: Path = DEFAULT_CACHE,
):
    """Download, train the tokenizer, and build all three splits.

    The tokenizer is trained on the training split only. Training it on val or
    test would leak information about those sentences into the vocabulary.
    """
    data = download_multi30k(cache_dir)

    tokenizer_path = cache_dir / f"bpe-{vocab_size}.json"
    if tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        corpus = data["train"]["de"] + data["train"]["en"]
        tokenizer = train_tokenizer(corpus, vocab_size, tokenizer_path)

    splits = {
        name: Multi30kDataset(
            data[name]["de"], data[name]["en"], tokenizer, max_len
        )
        for name in SPLITS
    }
    return splits, tokenizer
