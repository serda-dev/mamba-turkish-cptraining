"""CPU-only contract checks: no checkpoint downloads or GPU required."""
import math
from types import SimpleNamespace

import pytest
import torch
from jinja2 import Environment

from src.model import load
from src.eval import perplexity


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    def get_vocab(self):
        return self.vocab
    def __len__(self):
        return len(self.vocab)


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.resize_calls = []
    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=self.vocab_size)
    def resize_token_embeddings(self, size, **kwargs):
        self.resize_calls.append((size, kwargs))
        self.vocab_size = size
    def forward(self, input_ids, labels, **kwargs):
        # Distinct loss per sequence makes token weighting observable.
        return SimpleNamespace(loss=torch.tensor(float(input_ids[0, 0])))


def test_append_only_vocabulary_preserves_pretrained_ids():
    base = Tokenizer({"a": 0, "b": 1})
    load.validate_append_only_tokenizer(Tokenizer({"a": 0, "b": 1, "ğ": 2}), base)
    with pytest.raises(ValueError, match="ID remap"):
        load.validate_append_only_tokenizer(Tokenizer({"a": 1, "b": 0}), base)
    with pytest.raises(ValueError, match="shrinks"):
        load.validate_append_only_tokenizer(Tokenizer({"a": 0}), base)
    with pytest.raises(ValueError, match="unique and contiguous"):
        load.validate_append_only_tokenizer(Tokenizer({"a": 0, "b": 1, "ğ": 3}), base)


def mock_loaders(monkeypatch, model, reference):
    calls = {}
    def config_loader(name, **kwargs):
        calls["config"] = kwargs
        return SimpleNamespace(use_cache=True, use_mamba_kernels=True)
    def model_loader(name, **kwargs):
        calls["model"] = kwargs
        return model
    def tokenizer_loader(name, **kwargs):
        calls["tokenizer"] = kwargs
        return reference
    monkeypatch.setattr(load.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(load.AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(load.AutoTokenizer, "from_pretrained", tokenizer_loader)
    return calls


def test_model_revision_pins_config_weights_and_reference_tokenizer(monkeypatch):
    model = FakeModel()
    calls = mock_loaders(monkeypatch, model, Tokenizer({"a": 0, "b": 1}))
    load.load_model("base", Tokenizer({"a": 0, "b": 1, "ğ": 2}),
                    device="cpu", revision="abc123")
    assert all(calls[key]["revision"] == "abc123" for key in ("config", "model", "tokenizer"))
    assert model.resize_calls == [(3, {"mean_resizing": True})]


def test_shrink_and_remap_fail_without_silent_resize(monkeypatch):
    model = FakeModel(vocab_size=3)
    base = Tokenizer({"a": 0, "b": 1})
    mock_loaders(monkeypatch, model, base)
    with pytest.raises(ValueError, match="shrink checkpoint"):
        load.load_model("base", base, device="cpu")
    assert model.resize_calls == []
    with pytest.raises(ValueError, match="ID remap"):
        load.load_model("base", Tokenizer({"a": 1, "b": 0}), device="cpu")


def test_chat_fallback_renders_real_newlines():
    output = Environment().from_string(load.FALLBACK_JAMBA_CHAT_TEMPLATE).render(
        bos_token=None, messages=[{"role": "user", "content": "Merhaba"}], add_generation_prompt=True)
    assert output == "<|im_start|>user\nMerhaba<|im_end|>\n<|im_start|>assistant\n"
    assert "\\n" not in output


def batch(ids, labels=None, mask=None):
    output = {"input_ids": torch.tensor([ids])}
    if labels is not None:
        output["labels"] = torch.tensor([labels])
    if mask is not None:
        output["attention_mask"] = torch.tensor([mask])
    return output


def test_perplexity_uses_token_weighted_nll_and_shifted_nonpadding_count():
    model = FakeModel()
    result = perplexity.evaluate_batches(model, [batch([1, 4]), batch([3, 4, 5, 6, 0], mask=[1, 1, 1, 1, 0])], "cpu", "explicit_texts")
    assert result["total_tokens"] == 4
    assert result["mean_nll"] == 2.5
    assert result["average_perplexity"] == pytest.approx(math.exp(2.5))
    assert [r["num_tokens"] for r in result["individual_results"]] == [1, 3]


def test_masked_labels_and_overflow():
    nll, count = perplexity.compute_token_nll(FakeModel(), batch([1, 2, 3, 4], [1, 2, -100, -100]), "cpu")
    assert (nll, count) == (1.0, 1)
    assert math.isinf(perplexity._exp_nll(1000))
    assert perplexity.compute_token_nll(FakeModel(), batch([1]), "cpu") == (0.0, 0)
    with pytest.raises(ValueError, match="no predictable tokens"):
        perplexity.evaluate_batches(FakeModel(), [batch([1])], "cpu", "explicit_texts")


def test_missing_eval_corpus_fails_before_model_download(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Must not load model for missing evaluation data")
    monkeypatch.setattr(perplexity, "load_model_and_tokenizer", forbidden)
    with pytest.raises(ValueError, match="Choose exactly one"):
        perplexity.evaluate_perplexity("base")
    with pytest.raises(ValueError, match="empty"):
        perplexity.evaluate_perplexity("base", texts=[])


def test_cache_evaluator_rejects_training_cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"cache_identity": {"split": "train"}}')
    with pytest.raises(ValueError, match="Heldout cache"):
        list(perplexity._heldout_batches(str(manifest), None))


def backend_tokenizer(vocab, merges=None, normalizer=None):
    import json
    tokenizer = Tokenizer(vocab)
    state = {"model": {"type": "BPE", "vocab": vocab, "merges": merges or [], "dropout": None},
             "normalizer": normalizer, "pre_tokenizer": None, "decoder": None,
             "post_processor": None, "added_tokens": []}
    tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(state))
    return tokenizer


def test_bpe_extension_preserves_base_merges_and_normalization():
    base = backend_tokenizer({"a": 0, "b": 1}, [["a", "b"]])
    extended = backend_tokenizer({"a": 0, "b": 1, "ğ": 2}, [["a", "b"], ["b", "ğ"]])
    load.validate_append_only_tokenizer(extended, base)
    reordered = backend_tokenizer({"a": 0, "b": 1, "ğ": 2}, [["b", "ğ"], ["a", "b"]])
    with pytest.raises(ValueError, match="merge prefix"):
        load.validate_append_only_tokenizer(reordered, base)
    normalized = backend_tokenizer({"a": 0, "b": 1}, [["a", "b"]], {"type": "Lowercase"})
    with pytest.raises(ValueError, match="normalizer"):
        load.validate_append_only_tokenizer(normalized, base)


def test_existing_added_special_token_flags_cannot_change():
    import json
    base = backend_tokenizer({"a": 0, "b": 1})
    extended = backend_tokenizer({"a": 0, "b": 1, "ğ": 2})
    base_state = json.loads(base.backend_tokenizer.to_str())
    new_state = json.loads(extended.backend_tokenizer.to_str())
    base_state["added_tokens"] = [{"id": 1, "content": "b", "special": True}]
    new_state["added_tokens"] = [{"id": 1, "content": "b", "special": False}]
    base.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(base_state))
    extended.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(new_state))
    with pytest.raises(ValueError, match="added/special"):
        load.validate_append_only_tokenizer(extended, base)


def test_cache_evaluator_rejects_tokenizer_mismatch_before_loading_shards(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"cache_identity": {"split": "validation"}, "tokenizer_fingerprint": "wrong"}')
    tokenizer = Tokenizer({"a": 0, "b": 1})
    tokenizer.eos_token_id = 1
    tokenizer.pad_token_id = 1
    with pytest.raises(ValueError, match="fingerprint"):
        list(perplexity._heldout_batches(str(manifest), None, tokenizer=tokenizer))


def test_heldout_cache_evaluator_masks_padding_and_keeps_real_eos(tmp_path):
    from src.data.tokenize_pack import pack_and_tokenize_to_sharded_cache
    class SmallTokenizer(Tokenizer):
        eos_token_id = 1
        pad_token_id = 1
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[0, 1] for text in texts]}
    tokenizer = SmallTokenizer({"a": 0, "eos": 1})
    pack_and_tokenize_to_sharded_cache(
        iter([("a", "tr")]), tokenizer, seq_len=4,
        cache_dir=str(tmp_path), cache_identity={"split": "validation"},
        show_progress=False,
    )
    inputs = list(perplexity._heldout_batches(str(tmp_path / "manifest.json"), None, tokenizer))
    assert inputs[0]["labels"].tolist() == [[0, 1, 1, -100]]
    result = perplexity.evaluate_batches(FakeModel(), inputs, "cpu", "heldout_cache")
    assert result["total_tokens"] == 2
    assert result["evaluation_kind"] == "heldout_cache"
