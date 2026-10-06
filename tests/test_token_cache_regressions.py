import json
from pathlib import Path

import numpy as np
import pytest

from src.data.tokenize_pack import (
    pack_and_tokenize_to_sharded_cache, token_cache_is_complete,
)


class Tokenizer:
    vocab_size = 65536
    eos_token_id = 2
    pad_token_id = 2  # EOS and padding must be distinguished by length.
    def __len__(self):
        return 65537
    def get_vocab(self):
        return {"added": 65536, "eos": 2}
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[65536 if c == "z" else ord(c) for c in x] for x in texts]}


def build(tmp_path, items, **kwargs):
    return pack_and_tokenize_to_sharded_cache(
        iter(items), Tokenizer(), seq_len=4, cache_dir=str(tmp_path),
        show_progress=False, batch_size=1, chunks_per_shard=1, **kwargs,
    )


def values(root, manifest, key, dtype):
    return np.concatenate([np.fromfile(root / s[key], dtype=dtype) for s in manifest["shards"]])


def test_added_vocab_and_mixed_sources_padding(tmp_path):
    m = build(tmp_path, [("z", "tr"), ("a", "en"), ("b", "tr")])
    assert m["dtype"] == "uint32"
    assert values(tmp_path, m, "path", np.uint32).tolist() == [65536, 2, 97, 2, 98, 2, 2, 2]
    assert values(tmp_path, m, "token_source_ids_path", np.uint8).tolist() == [1, 1, 2, 2, 1, 1, 0, 0]
    assert values(tmp_path, m, "valid_lengths_path", np.uint32).tolist() == [4, 2]
    assert m["total_tokens"] == 6
    assert m["source_token_counts"] == {"tr": 4, "en": 2}


def test_interrupted_cache_resumes_after_empty_document(tmp_path):
    def failing():
        yield ("", "tr")
        yield ("a", "tr")
        raise RuntimeError("source failed")
    with pytest.raises(RuntimeError, match="source failed"):
        build(tmp_path, failing())
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert not saved["complete"]
    assert saved["texts_processed"] == 2
    assert saved["pending_buffer"] == [97, 2]
    m = build(tmp_path, [("", "tr"), ("a", "tr"), ("bc", "en")])
    expected = build(tmp_path / "baseline", [("", "tr"), ("a", "tr"), ("bc", "en")])
    assert values(tmp_path, m, "path", np.uint32).tolist() == values(tmp_path / "baseline", expected, "path", np.uint32).tolist()
    assert m["total_tokens"] == 5


def test_resume_at_document_boundary_even_when_document_spans_shards(tmp_path):
    def failing():
        yield ("abcdefghij", "tr")
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        build(tmp_path, failing())
    m = build(tmp_path, [("abcdefghij", "tr"), ("k", "en")])
    assert m["texts_processed"] == 2
    assert values(tmp_path, m, "path", np.uint32).tolist() == list(map(ord, "abcdefghij")) + [2, ord("k"), 2, 2, 2, 2]


def test_identity_and_integrity_enforced_before_reuse(tmp_path):
    m = build(tmp_path, ["abcd"], cache_identity={"data_revision": "one"})
    with pytest.raises(ValueError, match="identity"):
        build(tmp_path, [], cache_identity={"data_revision": "two"})
    assert token_cache_is_complete(str(tmp_path), 4, {"data_revision": "one"})
    path = tmp_path / m["shards"][0]["path"]
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    assert not token_cache_is_complete(str(tmp_path), 4)
    with pytest.raises(ValueError, match="integrity"):
        build(tmp_path, [], cache_identity={"data_revision": "one"})


def test_token_budget_and_short_tail_are_exact(tmp_path):
    m = build(tmp_path, ["abc", "defghij", "unused"], max_tokens=5)
    assert m["total_tokens"] == 5
    assert m["texts_processed"] == 2
    assert values(tmp_path, m, "valid_lengths_path", np.uint32).tolist() == [4, 1]
    assert values(tmp_path, m, "path", np.uint32).tolist() == [97, 98, 99, 2, 100, 2, 2, 2]


def test_padding_labels_preserve_real_eos(tmp_path):
    pytest.importorskip("torch")
    from src.data.dataloader import ShardedMemmapPackedDataset
    build(tmp_path, [("a", "tr")])
    item = ShardedMemmapPackedDataset(str(tmp_path / "manifest.json"))[0]
    assert item["labels"].tolist() == [97, 2, -100, -100]
    assert item["attention_mask"].tolist() == [1, 1, 0, 0]
    assert item["token_source_ids"].tolist() == [1, 1, 0, 0]


def test_in_memory_tail_and_chunk_limit():
    pytest.importorskip("torch")
    from src.data.tokenize_pack import pack_and_tokenize
    from src.data.dataloader import PackedDataset
    class Encoder(Tokenizer):
        def encode(self, text, **kwargs):
            return [ord(x) for x in text]
    chunks = pack_and_tokenize(iter(["abcd"]), Encoder(), seq_len=4, show_progress=False)
    assert len(chunks) == 2
    assert PackedDataset(chunks)[1]["labels"].tolist() == [2, -100, -100, -100]
    limited = pack_and_tokenize(iter(["abcdefghijk"]), Encoder(), seq_len=4,
                                max_chunks=1, show_progress=False)
    assert len(limited) == 1


def test_equivalent_saved_tokenizer_paths_share_cache_identity(tmp_path):
    from src.data.tokenize_pack import tokenization_fingerprint
    first, saved = Tokenizer(), Tokenizer()
    first.name_or_path = "upstream/model"
    saved.name_or_path = "/output/checkpoint-100"
    assert tokenization_fingerprint(first) == tokenization_fingerprint(saved)
    build(tmp_path, ["abc"])
    reused = pack_and_tokenize_to_sharded_cache(
        iter([]), saved, seq_len=4, cache_dir=str(tmp_path), show_progress=False,
    )
    assert reused["complete"]


def test_tokenizer_fingerprint_covers_content_and_special_tokens():
    from src.data.tokenize_pack import tokenization_fingerprint
    first, changed = Tokenizer(), Tokenizer()
    changed.get_vocab = lambda: {"different": 65536, "eos": 2}
    assert tokenization_fingerprint(first) != tokenization_fingerprint(changed)
    changed = Tokenizer()
    changed.eos_token_id = 3
    assert tokenization_fingerprint(first) != tokenization_fingerprint(changed)
