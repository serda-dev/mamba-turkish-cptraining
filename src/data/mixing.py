"""Deterministic text-source mixing for curriculum phases."""

import random
from typing import Iterable, Iterator, Optional, Tuple


def token_balanced_mix(streams, weights, tokenizer):
    """Deterministic deficit balancing by actual encoded tokens (including EOS).

    Stop on exhaustion instead of silently changing the requested distribution.
    The ratio error is bounded by document length; cache records exact results.
    """
    if len(streams) != len(weights) or not streams or any(w <= 0 for w in weights):
        raise ValueError("Streams require matching positive token weights")
    iterators = [iter(stream) for stream in streams]
    counts = [0] * len(streams)
    while True:
        idx = min(range(len(streams)), key=lambda i: counts[i] / weights[i])
        try:
            text = next(iterators[idx])
        except StopIteration:
            return
        tokens = tokenizer.encode(text, add_special_tokens=False)
        counts[idx] += len(tokens) + 1
        yield text, idx


def weighted_mix_texts(
    turkish_texts: Iterable[str],
    english_texts: Iterable[str],
    turkish_ratio: float = 0.75,
    seed: int = 42,
    max_items: Optional[int] = None,
) -> Iterator[Tuple[str, str]]:
    """
    Mix Turkish and English streams by weighted random source selection.

    The iterator is deterministic for a fixed seed. When one source is
    exhausted, it continues with the remaining source.
    """
    rng = random.Random(seed)
    tr_iter = iter(turkish_texts)
    en_iter = iter(english_texts)
    tr_done = False
    en_done = False
    emitted = 0

    while not (tr_done and en_done):
        if max_items is not None and emitted >= max_items:
            break

        choose_tr = rng.random() < turkish_ratio
        if tr_done:
            choose_tr = False
        if en_done:
            choose_tr = True

        if choose_tr:
            try:
                yield next(tr_iter), "turkish"
                emitted += 1
            except StopIteration:
                tr_done = True
        else:
            try:
                yield next(en_iter), "english"
                emitted += 1
            except StopIteration:
                en_done = True


def observed_ratio(samples: Iterable[Tuple[str, str]]) -> float:
    tr = 0
    total = 0
    for _, source in samples:
        total += 1
        if source == "turkish":
            tr += 1
    return tr / total if total else 0.0
