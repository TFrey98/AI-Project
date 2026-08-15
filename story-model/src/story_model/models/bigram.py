# `from __future__ import annotations` lets us write modern type hints like
# `torch.Tensor | None` even on Python 3.9, where that syntax would otherwise
# crash at import time. It just turns annotations into plain strings that
# tools can still read, so it doesn't change what the code does.
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """The simplest possible character-level language model.

    "Bigram" means it only ever looks at ONE character to predict the next
    one — it has no memory of anything earlier in the sequence. That makes
    it a weak model, but a great baseline: it proves the data pipeline,
    training loop, and generation loop all work before you invest in a
    model (e.g. an RNN or Transformer) that actually uses context.
    """

    def __init__(self, vocabulary_size: int = 256) -> None:
        super().__init__()

        # An nn.Embedding here doubles as a full lookup table of
        # "if the current character is X, here's a score for every possible
        # next character." Its weight matrix has shape
        # (vocabulary_size, vocabulary_size): row i holds the logits (raw,
        # un-normalized scores) for what comes after character i.
        #
        # Note: vocabulary_size must match whatever tokenizer produced your
        # training data (e.g. CharTokenizer.vocab_size) — train.py and
        # generate.py already pass that in explicitly, so the 256 default
        # here is only used if you construct the model without specifying it.
        self.transition_logits = nn.Embedding(
            vocabulary_size,
            vocabulary_size,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model on a batch of token ids.

        tokens: integer tensor of shape (batch, sequence_length) — the
            input characters, already converted to ids by the tokenizer.
        targets: same shape as tokens, holding the "correct" next character
            for each position. Only needed during training; leave as None
            when just generating text.

        Returns (logits, loss). logits has shape
        (batch, sequence_length, vocabulary_size) — one score distribution
        per input position. loss is a single number measuring how wrong
        those predictions were (None if targets weren't given).
        """
        logits = self.transition_logits(tokens)
        loss = None

        if targets is not None:
            batch, sequence, vocabulary = logits.shape

            # Cross-entropy loss expects a flat list of predictions and a
            # flat list of correct answers, so we collapse the
            # (batch, sequence) dimensions into one before comparing.
            loss = F.cross_entropy(
                logits.reshape(batch * sequence, vocabulary),
                targets.reshape(batch * sequence),
            )

        return logits, loss

    @torch.no_grad()  # generation doesn't need gradients, so skip tracking them
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend `tokens` by sampling `max_new_tokens` new characters, one at a time.

        tokens: integer tensor of shape (batch, sequence_length) to start from
            (e.g. a single "seed" character).

        Each step: run the model, look only at the prediction for the very
        last position, turn those logits into probabilities, and randomly
        sample the next character from that distribution. The sampled
        character is appended and fed back in as context for the next step.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)

            # Only the prediction for the most recent character matters —
            # a bigram model ignores everything before it anyway.
            last_step_logits = logits[:, -1, :]

            probabilities = F.softmax(last_step_logits, dim=-1)

            # Sample one character id per sequence in the batch, weighted
            # by probability (as opposed to always picking the top score,
            # which would make output deterministic and repetitive).
            next_token = torch.multinomial(probabilities, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
