"""Text tokenizers, dataset loading, and random batching.

A language model never sees raw text — it sees a sequence of integers.
A *tokenizer* is the (reversible) function that maps text -> integers and
back. The choice of tokenizer is one of the highest-leverage decisions in
building a language model: it determines the model's vocabulary size, how
many "steps" it takes to read a given amount of text, and even what kinds
of patterns are easy or hard for it to learn. This file implements two
tokenizers of increasing sophistication (character-level, then byte-pair
encoding) so their tradeoffs can be compared directly — see
configs/transformer_small.yaml (char) vs configs/transformer_bpe.yaml (BPE)
for the same architecture trained under each.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch

from story_model.corpus import sha256_text


# NumPy has some fixed allocation overhead, so the original simple Python
# loop remains quicker for short prompts.  Corpus-sized strings take the
# vectorized path, where each merge pass runs in compiled code rather than
# executing one Python loop iteration per token.
BPE_VECTORIZATION_THRESHOLD = 64 * 1024


@dataclass
class CharTokenizer:
    """The simplest possible tokenizer: one token per character.

    stoi ("string to integer") and itos ("integer to string") are inverse
    lookup tables built from whatever characters actually appear in the
    training text — so the vocabulary is exactly as large as the alphabet
    of the corpus (for Shakespeare, ~65 characters: letters, punctuation,
    whitespace). This has two big advantages for a small learning project:
    it's trivial to reason about (every token is one visible character,
    so model output is always human-readable) and it can never fail to
    encode a character it's seen before. Its downside is that it forces
    the model to spend a lot of its limited context window on very
    low-level detail — predicting "t", then "h", then "e" one character
    at a time is a much harder modeling problem than predicting the
    single token "the" would be, which is exactly the gap byte-pair
    encoding (below) is designed to close.
    """

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # sorted(set(text)) gives a deterministic, reproducible ordering
        # of the alphabet — using an unsorted set would make the token ids
        # (and therefore any trained model's embedding table) depend on
        # incidental set-iteration order, breaking reproducibility.
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[t] for t in tokens)

    def token_byte_length(self, token_id: int) -> int:
        return len(
            self.itos[token_id].encode("utf-8")
        )

    def to_dict(self) -> dict:
        # Every tokenizer's to_dict() is embedded directly into training
        # checkpoints (see train.py) so a checkpoint is fully
        # self-describing: generate.py can later reconstruct the exact
        # same encode/decode mapping without needing the original corpus
        # on disk, or needing to know in advance which tokenizer type was
        # used. The "type" field is what lets tokenizer_from_dict() below
        # dispatch to the right class.
        return {
            "type": "char",
            "version": 1,
            "stoi": self.stoi,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        tokenizer_type = data.get("type", "char")

        if tokenizer_type != "char":
            raise ValueError("tokenizer data is not character-level")

        return cls(
            # JSON object keys are always strings, so a checkpoint's
            # itos={"0": "a", "1": "b", ...} needs its keys cast back to
            # int before it's usable as a token-id lookup table again.
            stoi={
                str(character): int(token_id)
                for character, token_id in data["stoi"].items()
            },
            itos={
                int(token_id): str(character)
                for token_id, character in data["itos"].items()
            },
        )


def merge_token_pair(
    tokens: list[int],
    pair: tuple[int, int],
    new_token: int,
) -> list[int]:
    """Replace every occurrence of `pair` in `tokens` with `new_token`.

    This is the single operation byte-pair encoding (BPE) is built from —
    both training a BPE tokenizer and using it to encode new text repeat
    this merge step over and over (see ByteBPETokenizer below). Matches
    are taken left to right and non-overlapping: once two tokens are
    merged, the merged result isn't reconsidered as part of another pair
    in the same pass (e.g. merging "aa" in "aaa" produces one merged
    token plus one leftover "a", not two overlapping merges).
    """

    merged: list[int] = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            merged.append(new_token)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1

    return merged


def merge_token_pair_array(
    tokens: np.ndarray,
    pair: tuple[int, int],
    new_token: int,
) -> np.ndarray:
    """NumPy equivalent of :func:`merge_token_pair`.

    Results are bit-for-bit compatible with the original left-to-right,
    non-overlapping Python implementation.  The only special case is a pair
    whose two sides are equal: adjacent match positions overlap inside runs
    such as ``aaa``, so only alternating starts in each run are retained.
    """

    if tokens.ndim != 1:
        raise ValueError("tokens must be a one-dimensional array")

    if len(tokens) < 2:
        return tokens

    matches = tokens[:-1] == pair[0]
    matches &= tokens[1:] == pair[1]
    starts = np.flatnonzero(matches)

    if len(starts) == 0:
        return tokens

    if pair[0] == pair[1] and len(starts) > 1:
        new_run = np.empty(len(starts), dtype=np.bool_)
        new_run[0] = True
        new_run[1:] = np.diff(starts) > 1
        run_start_values = np.where(new_run, starts, 0)
        run_starts = np.maximum.accumulate(run_start_values)
        starts = starts[(starts - run_starts) % 2 == 0]

    tokens[starts] = new_token
    keep = np.ones(len(tokens), dtype=np.bool_)
    keep[starts + 1] = False
    return tokens[keep]


@dataclass
class ByteBPETokenizer:
    """UTF-8 byte tokenizer with learned byte-pair merges.

    Byte-Pair Encoding (BPE) is the tokenization scheme used by GPT-2/3/4,
    LLaMA, and most modern LLMs. It starts from the 256 possible raw UTF-8
    bytes as the base vocabulary (so, unlike CharTokenizer, it can NEVER
    encounter a character it doesn't know how to represent — worst case it
    falls back to spelling something out byte by byte), then learns to
    "merge" frequently-adjacent byte pairs into new, single tokens: e.g.
    if "t" and "h" appear next to each other very often in the training
    text, BPE learns a "th" token; if "th" and "e" are then frequently
    adjacent, it can go on to learn a "the" token. The result is a
    vocabulary that automatically captures whatever frequent substrings
    (often whole common words, for a large enough vocab_size) exist in the
    training corpus, without any language-specific rules baked in.

    Why this matters for the model: fewer, more meaningful tokens per unit
    of text means the same context window (block_size) covers more actual
    content, and the model spends its capacity predicting higher-level
    units ("the") instead of low-level spelling ("t", "h", "e"). See
    scripts/inspect_bpe.py for a tool that reports the resulting
    bytes-per-token compression ratio on this project's corpus.
    """

    # merges[i] is the i-th pair learned, in the order it was learned.
    # Order matters for both training and encoding: earlier merges were
    # more frequent (and therefore get applied first), and later merges
    # may reference token ids produced by earlier ones (e.g. merging
    # "th" + "e" into "the" requires "th" to already exist as a token).
    merges: list[tuple[int, int]]

    # Extra whole-string tokens layered ON TOP of the base byte/merge
    # vocabulary — e.g. Phase 15's "<|user|>" control markers. Unlike a
    # merge (which is only ever reachable by first tokenizing raw bytes
    # and then repeatedly combining adjacent pairs), a special token is
    # matched as a literal, atomic unit directly against the input text
    # before BPE segmentation ever runs — see encode() below. This keeps
    # control markers from ever being split apart the way ordinary BPE
    # merging might otherwise split an arbitrary substring.
    special_tokens: tuple[str, ...] = field(default_factory=tuple)

    # An internal cache mapping every token id (0-255 for raw bytes, 256+
    # for learned merges) to the literal bytes it expands to — built once
    # in __post_init__ so decode() and token_byte_length() are O(1) dict
    # lookups instead of having to walk the merge list every call.
    # field(init=False) means dataclass's auto-generated __init__ doesn't
    # accept it as a constructor argument; it's derived state, not input.
    _token_bytes: dict[int, bytes] = field(
        init=False,
        repr=False,
    )
    # Special-token text -> its assigned token id, built once alongside
    # _token_bytes for the same reason: encode() needs to look this up on
    # every call and shouldn't recompute it from scratch each time.
    _special_to_id: dict[str, int] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        normalized_merges: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()

        for rank, raw_pair in enumerate(self.merges):
            if len(raw_pair) != 2:
                raise ValueError(
                    "each BPE merge must contain two token ids"
                )

            pair = (
                int(raw_pair[0]),
                int(raw_pair[1]),
            )
            new_token = 256 + rank

            # Every merge must only reference tokens that already exist at
            # the point it's applied (raw bytes, or earlier merges) — a
            # merge referencing a not-yet-defined token id would be
            # self-contradictory (or reference something learned later),
            # which can only happen if a merges list was corrupted or
            # hand-edited incorrectly.
            if not all(
                0 <= token_id < new_token
                for token_id in pair
            ):
                raise ValueError(
                    "BPE merge references an unavailable token"
                )

            if pair in seen_pairs:
                raise ValueError("duplicate BPE merge pair")

            normalized_merges.append(pair)
            seen_pairs.add(pair)

        self.merges = normalized_merges

        # Validate and de-duplicate special tokens the same way merges
        # were above — order is preserved (it determines assigned token
        # ids below), and a duplicate would silently make two different
        # ids ambiguous for the same text.
        normalized_special_tokens: list[str] = []

        for token in self.special_tokens:
            if not isinstance(token, str):
                raise TypeError(
                    "each special token must be a string"
                )

            if not token:
                raise ValueError(
                    "special tokens cannot be empty"
                )

            if token in normalized_special_tokens:
                raise ValueError(
                    f"duplicate special token: {token!r}"
                )

            normalized_special_tokens.append(token)

        self.special_tokens = tuple(
            normalized_special_tokens
        )

        # Every raw byte value is its own one-byte token to start.
        token_bytes = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

        # Each learned merge's byte representation is just the
        # concatenation of its two parts' byte representations — this is
        # what lets decode() turn token ids straight back into the exact
        # original bytes, with no ambiguity or lossy step anywhere in the
        # pipeline.
        for rank, pair in enumerate(self.merges):
            token_bytes[256 + rank] = (
                token_bytes[pair[0]]
                + token_bytes[pair[1]]
            )

        # Special tokens are assigned ids continuing straight on from the
        # last learned merge (base_vocab_size, base_vocab_size + 1, ...),
        # so adding special tokens on top of an already-trained BPE
        # vocabulary NEVER changes the meaning of any existing token id —
        # see checkpoint.py's load_model_warm_start, which relies on
        # exactly this property to preserve a model's already-learned
        # embeddings when its vocabulary grows.
        base_vocabulary_size = 256 + len(self.merges)
        special_to_id = {
            token: base_vocabulary_size + index
            for index, token in enumerate(self.special_tokens)
        }

        for token, token_id in special_to_id.items():
            token_bytes[token_id] = token.encode("utf-8")

        self._token_bytes = token_bytes
        self._special_to_id = special_to_id

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        """Learn merges greedily: repeatedly merge the most frequent pair.

        This is the classic BPE training algorithm. Starting from raw
        UTF-8 bytes, it repeats "count every adjacent pair, merge the most
        frequent one, repeat" until either vocab_size is reached or no
        pair occurs often enough to be worth merging (min_frequency). This
        is a *greedy* algorithm — it always takes the locally-best merge
        available at each step rather than searching for a globally
        optimal vocabulary — which is standard for BPE and is what makes
        it fast enough to run over a multi-megabyte corpus (see
        scripts/build_corpus.py) in a reasonable amount of time.
        """

        if not text:
            raise ValueError("BPE training text cannot be empty")

        if vocab_size < 256:
            raise ValueError(
                "BPE vocab_size must be at least 256"
            )

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be positive"
            )

        tokens = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []

        while 256 + len(merges) < vocab_size:
            # Counter(zip(tokens, tokens[1:])) counts every adjacent pair
            # in one pass — zip(tokens, tokens[1:]) is the standard
            # Python idiom for "every consecutive pair in this sequence."
            pair_counts = Counter(
                zip(tokens, tokens[1:])
            )

            if not pair_counts:
                break

            # Prefer the most frequent pair, then the numerically
            # smallest pair so training is deterministic on ties.
            best_pair, frequency = min(
                pair_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )

            if frequency < min_frequency:
                # No remaining pair is common enough to justify spending a
                # vocabulary slot on it — merging rare pairs would waste
                # vocab_size on tokens that barely ever get reused, and
                # rare substrings are exactly what would happen if
                # training and encoding text differ (i.e. this is also a
                # safeguard against overfitting the vocabulary to
                # one-off noise in the training text).
                break

            new_token = 256 + len(merges)
            tokens = merge_token_pair(
                tokens,
                best_pair,
                new_token,
            )
            merges.append(best_pair)

        return cls(merges=merges)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.special_tokens)

    @property
    def base_vocab_size(self) -> int:
        # The vocabulary size BEFORE any special tokens are layered on
        # top — i.e. exactly what vocab_size used to mean before Phase 15.
        # This is what checkpoint.py's warm-start logic checks stays
        # fixed: special tokens may only ever be ADDED at the end, never
        # mixed into or resizing the learned byte/merge vocabulary itself.
        return 256 + len(self.merges)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return dict(self._special_to_id)

    def with_special_tokens(
        self,
        special_tokens: list[str] | tuple[str, ...],
    ) -> "ByteBPETokenizer":
        """Return a new tokenizer with additional special tokens appended.

        Already-present special tokens are left in their original
        position (and therefore keep their original id) — only genuinely
        new tokens are appended at the end. Because this returns a NEW
        ByteBPETokenizer rather than mutating self, existing references to
        the original tokenizer (and any ids it already assigned) remain
        valid and unaffected.
        """

        combined = list(self.special_tokens)

        for token in special_tokens:
            if token not in combined:
                combined.append(token)

        return ByteBPETokenizer(
            merges=list(self.merges),
            special_tokens=tuple(combined),
        )

    def _encode_bpe_segment(
        self,
        text: str,
        progress_callback: Callable[[int, int, int], None] | None = None,
        progress_offset: int = 0,
        progress_total: int | None = None,
    ) -> list[int]:
        # The original encode() logic, now applied only to the stretches
        # of text BETWEEN special tokens (see encode() below) rather than
        # unconditionally to the whole input. Encoding replays the SAME
        # merges, in the SAME order they were learned, starting again from
        # raw bytes — this is why merge order is stored and preserved:
        # applying merges out of order (or a different subset) would
        # produce a different, incompatible tokenization than what the
        # model was trained on.
        encoded_bytes = text.encode("utf-8")
        total = (
            len(self.merges)
            if progress_total is None
            else progress_total
        )

        if len(encoded_bytes) >= BPE_VECTORIZATION_THRESHOLD:
            return self._encode_bpe_segment_vectorized(
                encoded_bytes,
                progress_callback=progress_callback,
                progress_offset=progress_offset,
                progress_total=total,
            )

        tokens = list(encoded_bytes)

        for rank, pair in enumerate(self.merges):
            tokens = merge_token_pair(
                tokens,
                pair,
                256 + rank,
            )

            if progress_callback is not None:
                progress_callback(
                    progress_offset + rank + 1,
                    total,
                    len(tokens),
                )

        return tokens

    def _encode_bpe_segment_vectorized(
        self,
        encoded_bytes: bytes,
        progress_callback: Callable[[int, int, int], None] | None,
        progress_offset: int,
        progress_total: int,
    ) -> list[int]:
        """Replay learned merges with exact NumPy-vectorized passes."""

        largest_token_id = self.base_vocab_size - 1

        if largest_token_id <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif largest_token_id <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:
            dtype = np.uint64

        tokens = np.frombuffer(
            encoded_bytes,
            dtype=np.uint8,
        ).astype(dtype)

        for rank, pair in enumerate(self.merges):
            tokens = merge_token_pair_array(
                tokens,
                pair,
                256 + rank,
            )

            if progress_callback is not None:
                progress_callback(
                    progress_offset + rank + 1,
                    progress_total,
                    len(tokens),
                )

        return tokens.tolist()

    def encode(
        self,
        text: str,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> list[int]:
        """Encode text, optionally reporting completed BPE merge passes.

        The callback receives ``(completed_passes, total_passes,
        current_token_count)``.  It does not change tokenization; it only
        exposes progress during large, otherwise silent corpus encodes.
        """

        if not self.special_tokens:
            return self._encode_bpe_segment(
                text,
                progress_callback=progress_callback,
            )

        # Special tokens are matched as literal substrings BEFORE any BPE
        # segmentation runs, so "<|user|>" always becomes exactly one
        # token id, never something BPE merging might otherwise split it
        # into. Sorting longest-first (then by id, for determinism on
        # ties) ensures that if one special token's text happens to be a
        # prefix of another's (e.g. "<|turn|>" vs "<|turn|>end"), the
        # LONGER, more specific match wins — the same "maximal munch"
        # principle used by most tokenizers and lexers.
        ordered_special_tokens = sorted(
            self.special_tokens,
            key=lambda token: (
                -len(token),
                self._special_to_id[token],
            ),
        )
        pattern = re.compile(
            "|".join(
                re.escape(token)
                for token in ordered_special_tokens
            )
        )

        matches = list(pattern.finditer(text))
        segment_count = len(matches) + 1
        progress_total = len(self.merges) * segment_count
        tokens: list[int] = []
        position = 0

        # Walk the text left to right: everything BEFORE each special-
        # token match gets ordinary BPE encoding; the match itself becomes
        # its single reserved token id; then continue past the match.
        # Whatever text remains after the last match is BPE-encoded too.
        for segment_index, match in enumerate(matches):
            tokens.extend(
                self._encode_bpe_segment(
                    text[position : match.start()],
                    progress_callback=progress_callback,
                    progress_offset=(
                        segment_index * len(self.merges)
                    ),
                    progress_total=progress_total,
                )
            )
            tokens.append(
                self._special_to_id[match.group(0)]
            )
            position = match.end()

        tokens.extend(
            self._encode_bpe_segment(
                text[position:],
                progress_callback=progress_callback,
                progress_offset=(
                    len(matches) * len(self.merges)
                ),
                progress_total=progress_total,
            )
        )
        return tokens

    def decode(self, tokens: list[int]) -> str:
        pieces: list[bytes] = []

        for token_id in tokens:
            if token_id not in self._token_bytes:
                raise ValueError(
                    f"unknown BPE token id: {token_id}"
                )

            pieces.append(self._token_bytes[token_id])

        # errors="replace" swaps any invalid byte sequence for the Unicode
        # replacement character (U+FFFD) instead of raising. This matters
        # because a model can, in principle, GENERATE a token sequence
        # that doesn't correspond to any valid UTF-8 string (e.g. an
        # incomplete multi-byte character at the very end of a sample) —
        # decode() needs to degrade gracefully rather than crash the
        # generation script over what's ultimately just a slightly
        # malformed sample.
        return b"".join(pieces).decode(
            "utf-8",
            errors="replace",
        )

    def token_byte_length(self, token_id: int) -> int:
        if token_id not in self._token_bytes:
            raise ValueError(
                f"unknown BPE token id: {token_id}"
            )

        return len(self._token_bytes[token_id])

    def to_dict(self) -> dict:
        return {
            "type": "byte_bpe",
            # Bumped from 1 to 2 when special_tokens was added — this is
            # what lets from_dict() below tell an older, pre-Phase-15
            # checkpoint (implicitly version 1, no special tokens) apart
            # from a newer one, and reject anything it doesn't recognize
            # rather than silently misinterpreting unknown fields.
            "version": 2,
            "merges": [
                [left, right]
                for left, right in self.merges
            ],
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ByteBPETokenizer":
        if data.get("type") != "byte_bpe":
            raise ValueError("tokenizer data is not byte-level BPE")

        version = int(data.get("version", 1))

        if version not in {1, 2}:
            raise ValueError(
                f"unsupported BPE tokenizer version: {version}"
            )

        return cls(
            merges=[
                (int(pair[0]), int(pair[1]))
                for pair in data["merges"]
            ],
            # .get(..., ()) is what makes this backward compatible with
            # version-1 checkpoints saved before special_tokens existed —
            # they simply get an empty tuple, i.e. "no special tokens,"
            # which is exactly what those older checkpoints actually had.
            special_tokens=tuple(
                data.get("special_tokens", ())
            ),
        )


# A type alias, not a new class: anywhere the code says `Tokenizer`, it
# means "either a CharTokenizer or a ByteBPETokenizer" — both expose the
# same encode/decode/vocab_size/to_dict interface (Python doesn't enforce
# this the way a formal Protocol or ABC would, but every caller in this
# project treats the two classes interchangeably).
Tokenizer = Union[CharTokenizer, ByteBPETokenizer]


def split_text(
    text: str, train_split: float
) -> tuple[str, str]:
    """Split one blob of raw text into a training prefix and validation suffix.

    This is the "legacy" single-file splitting path (used by the original
    tiny_shakespeare configs, before the project moved to the multi-document
    corpus pipeline in corpus.py). It's a simple, deterministic prefix/suffix
    split — no shuffling — so the same text and train_split always produce
    the same split, which matters for reproducibility: a checkpoint's
    training/validation data can always be reconstructed later from just
    the config, with no separate record of which rows went where.
    """

    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    n = int(train_split * len(text))
    training_text = text[:n]
    validation_text = text[n:]

    if not training_text or not validation_text:
        raise ValueError(
            "text split must produce non-empty train and validation data"
        )

    return training_text, validation_text


def build_tokenizer(
    training_text: str,
    config: dict | None = None,
) -> Tokenizer:
    """Construct a tokenizer from config, fitting it to TRAINING text only.

    This function is always called with `training_text` — never the full
    corpus, never the validation text — and that's deliberate, not
    incidental. If the BPE merge vocabulary were learned from text that
    includes the validation split, the tokenizer itself would have
    "seen" validation data before evaluation ever ran, which is a subtle
    form of data leakage: the resulting BPE vocabulary could end up
    containing merges tuned to patterns that only occur in the supposedly
    held-out text, inflating validation performance in a way that
    wouldn't generalize to genuinely new text. Keeping tokenizer training
    and model training both scoped to the same training split is what
    makes the validation loss (see evaluate.py) a trustworthy estimate of
    how the model performs on unseen text.
    """

    config = config or {}
    tokenizer_type = config.get("type", "char")

    if tokenizer_type == "char":
        return CharTokenizer.from_text(training_text)

    if tokenizer_type == "byte_bpe":
        # special_tokens (e.g. Phase 15's character control markers) are
        # layered on AFTER training, via with_special_tokens — they're
        # never part of what BPE training itself learns from corpus
        # statistics, since they need fixed, predictable ids rather than
        # ids that depend on how frequently they happen to appear in text.
        return ByteBPETokenizer.train(
            text=training_text,
            vocab_size=config.get("vocab_size", 512),
            min_frequency=config.get(
                "min_frequency", 2
            ),
        ).with_special_tokens(
            tuple(config.get("special_tokens", ()))
        )

    raise ValueError(
        f"Unknown tokenizer type: {tokenizer_type!r}"
    )


def tokenizer_from_dict(data: dict) -> Tokenizer:
    """The inverse of `Tokenizer.to_dict()` — restores a tokenizer from a
    checkpoint's saved metadata rather than retraining it from a corpus.

    This is what lets generate.py and evaluate.py reproduce a model's
    EXACT training-time vocabulary from just the checkpoint file, with no
    dependency on the original corpus still existing on disk in the same
    place, or on tokenizer training being deterministic enough to
    reproduce bit-for-bit (BPE training over a large corpus, run twice, is
    deterministic here — see the tie-breaking rule in
    ByteBPETokenizer.train — but restoring from disk is still simpler,
    faster, and more robust than depending on that).
    """

    tokenizer_type = data.get("type", "char")

    if tokenizer_type == "char":
        return CharTokenizer.from_dict(data)

    if tokenizer_type == "byte_bpe":
        return ByteBPETokenizer.from_dict(data)

    raise ValueError(
        f"Unknown tokenizer type: {tokenizer_type!r}"
    )


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_corpus_manifest(
    data_config: dict,
) -> dict | None:
    """Load the manifest a multi-document corpus was built with, if any.

    See corpus.py for what the manifest records and why (document-level
    provenance, SHA-256 hashes for leakage detection). Returns None for
    configs that don't use the document-corpus pipeline at all (e.g. the
    single-file tiny_shakespeare configs) — this function is meant to be
    safe to call unconditionally rather than requiring every caller to
    first check whether a manifest_path was configured.
    """

    manifest_path = data_config.get("manifest_path")

    if manifest_path is None:
        return None

    return json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )


def validate_corpus_manifest(
    data_config: dict,
    training_text: str,
    validation_text: str,
) -> dict | None:
    """Verify the train/val text on disk still matches what the manifest recorded.

    Hashing and comparing here protects against a specific, easy-to-make
    mistake: rebuilding data/raw/ (adding, removing, or editing a source
    document) without rerunning scripts/build_corpus.py, so
    data/processed/train.txt and val.txt silently go stale relative to
    manifest.json. Without this check, training would proceed on
    mismatched data with no error — you'd only notice something was wrong
    much later (or never), when metrics didn't match expectations. Raising
    immediately, before any expensive training starts, turns a silent
    correctness bug into a loud, immediate, cheap-to-fix one.
    """

    manifest = load_corpus_manifest(data_config)

    if manifest is None:
        return None

    for split_name, text in (
        ("train", training_text),
        ("val", validation_text),
    ):
        expected_hash = (
            manifest.get(split_name, {}).get("sha256")
        )

        if expected_hash is None:
            raise ValueError(
                f"corpus manifest is missing {split_name} SHA-256"
            )

        if sha256_text(text) != expected_hash:
            raise ValueError(
                f"{split_name} corpus does not match its manifest"
            )

    return manifest


def load_text_splits(
    data_config: dict,
) -> tuple[str, str]:
    """Load explicit document splits or a legacy split from one file.

    This function is the single entry point train.py, evaluate.py, and
    generate.py all use to obtain (training_text, validation_text),
    regardless of which era of config format a given YAML file uses:
    the newer `train_path`/`val_path`/`manifest_path` trio (multi-document
    corpus, see corpus.py) or the older single `path` + `train_split`
    (one file, split by position, see split_text() above). Centralizing
    this choice here means every consumer gets manifest verification and
    consistent error messages for free, instead of each script needing to
    reimplement (and potentially get slightly wrong) the same branching
    logic.
    """

    has_path = "path" in data_config
    has_train_path = "train_path" in data_config
    has_val_path = "val_path" in data_config

    if has_path and (has_train_path or has_val_path):
        raise ValueError(
            "data config cannot combine path with train_path/val_path"
        )

    if has_train_path != has_val_path:
        raise ValueError(
            "data config must provide both train_path and val_path"
        )

    if has_train_path:
        training_text = load_text(data_config["train_path"])
        validation_text = load_text(data_config["val_path"])

        if not training_text or not validation_text:
            raise ValueError(
                "train and validation files must both be non-empty"
            )

        validate_corpus_manifest(
            data_config,
            training_text,
            validation_text,
        )

        return training_text, validation_text

    if has_path:
        if "manifest_path" in data_config:
            raise ValueError(
                "manifest_path requires explicit train_path/val_path"
            )

        if "train_split" not in data_config:
            raise ValueError(
                "single-file data config requires train_split"
            )

        return split_text(
            load_text(data_config["path"]),
            float(data_config["train_split"]),
        )

    raise ValueError(
        "data config requires path or train_path/val_path"
    )


def train_test_split(data: torch.Tensor, train_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    # A tensor-level counterpart to split_text() above, kept for callers
    # that already have encoded token ids rather than raw text (mainly
    # test fixtures at this point — production training/evaluation code
    # now splits text BEFORE tokenizing, via load_text_splits(), so that
    # a BPE tokenizer's vocabulary is never fit on validation-adjacent
    # bytes; see build_tokenizer()'s docstring for why that matters).
    n = int(train_split * len(data))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random (input, target) windows for one training step.

    Rather than walking through the corpus sequentially, every training
    step draws `batch_size` independent random starting positions and
    slices out a `block_size`-token window from each — this is what makes
    each optimizer step see a fresh, unordered sample of the corpus, and
    is standard practice for training on a single long text: there's no
    natural notion of "epoch" here, just an effectively endless stream of
    random windows. `y` is the same windows shifted one token to the
    right, so y[i] is always the correct "next token" prediction for
    x[i] — this shift is what turns a single text into a supervised
    next-token-prediction dataset with no separate labels needed.
    """

    # randint's upper bound excludes block_size positions from the end of
    # `data` so that data[i : i + block_size + 1] (needed for the target
    # window y) never runs past the end of the tensor.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
