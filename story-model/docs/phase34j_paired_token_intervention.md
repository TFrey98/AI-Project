# Phase 34j: paired token-identity intervention

Phase 34i rejected global boundary/type factorization. It reduced the target
B-to-I error by roughly 25–40%, but remained far above the registered ceiling
and introduced regressions in previously clean Phase 32 cells. Its two focus
tokens also moved in opposite directions: `or` was fixed completely while
`ro` became worse.

That result does not justify routing individual token IDs through different
heads. Such a rule would encode observations from the existing lexical and
transfer panels, while retaining the worst target failure. Phase 34j instead
uses a frozen paired intervention to determine whether the error follows the
token identity inside the same prompt context and, if so, which representation
path carries it.

No training, decoding change, data rewrite, full Phase 34 evaluation, or
checkpoint promotion is part of this phase.

## Pair contract

The audit visits Phase 31 lexical and transfer `scene_route` gold starts whose
first BPE token is the atomic two-byte token `or` or `ro`. For every start it
constructs one counterfactual sequence by exchanging those tokens.

A pair is valid only when:

- both strings are one canonical BPE token under the checkpoint tokenizer;
- the replacement changes exactly one token position;
- the token width remains two bytes;
- byte offsets and sequence length remain unchanged;
- the replaced token sequence is the canonical encoding of the altered
  prompt; and
- every other prompt and suffix token remains identical.

Any invalid focus pair invalidates the formal audit rather than silently
selecting an easier subset.

## Four frozen conditions

At the gold start, the proposer byte state is assembled from a contextual
component and a direct raw-byte embedding. The audit evaluates:

| Condition | Contextual state | Direct byte embedding |
|---|---|---|
| `original` | Original | Original |
| `byte_swap_only` | Original | Swapped token |
| `context_swap_only` | Swapped token | Original |
| `full_swap` | Swapped token | Swapped token |

The complete `original` and `full_swap` calculations are tested against the
ordinary model forward pass. The mixed conditions are internal causal
interventions; they do not represent generated text.

All margins are normalized as rendered `ro` minus rendered `or`, regardless
of which token appeared in the source row. A negative B-minus-I margin shift
therefore means that rendering `ro` makes the model more I-like.

## Tests

From `story-model/`:

```bash
.venv/bin/python -m pytest -q
```

The tests verify canonical equal-width pair construction, exactly one changed
token, equality between intervention endpoints and normal model inference,
normalization of both swap directions, checkpoint provenance, support floors,
and every formal decision branch.

## Run

Use the same frozen Phase 34d diagnostic checkpoint used by the earlier
boundary audits and the rejected Phase 34i pilot checkpoint:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_paired_token_intervention.py \
  --phase34d-checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/diagnostic-ineligible.pt \
  --phase34i-checkpoint checkpoints/explicit_offset_factorized_head_pilot/diagnostic-ineligible.pt \
  --phase34i-decision runs/phase34i-factorized-head-decision.json \
  --phase31-data-dir data/character/typed_span_resolver \
  --device mps \
  --batch-size 16 \
  --output runs/phase34j-paired-token-intervention.summary.json
```

This writes the summary plus
`runs/phase34j-paired-token-intervention.summary.rows.jsonl`, which contains
every paired prediction and margin.

Recompute and export the decision independently:

```bash
.venv/bin/python scripts/gate_paired_token_intervention.py \
  --summary runs/phase34j-paired-token-intervention.summary.json \
  --output runs/phase34j-paired-token-intervention.decision.json
```

## Decision rules

The audit requires at least 400 valid pairs, at least 200 starting from each
token identity, zero invalid focus pairs, matching tokenizers, the Phase 34d
boundary objective in both checkpoints, and factorization only in Phase 34i.

Token identity is considered causal only when, separately inside both the
original-`or` and original-`ro` context strata:

- rendered `ro` has at least 20 percentage points more B-to-I errors than
  rendered `or`; and
- its error rate is at least three times the rendered-`or` rate, using a 1%
  denominator floor.

The direct-byte or contextual path is selected only when it accounts for at
least 60% of the complete paired margin effect and the other path accounts for
less than 40%. If both reach 40%, the effect is classified as distributed.

| Branch | Meaning |
|---|---|
| `direct_byte_identity_path_indicated` | Both checkpoints show a causal identity effect carried primarily by the direct byte embedding |
| `contextual_token_identity_path_indicated` | Both show the effect primarily through the frozen contextual token state |
| `distributed_identity_path_indicated` | Both representation paths materially carry the effect |
| `prompt_context_proxy_indicated` | The error does not follow identity within fixed contexts; token ID was a proxy for prompt structure |
| `checkpoint_specific_identity_interaction_indicated` | Identity is causal in only one checkpoint |
| `mixed_identity_pathways_indicated` | Both show identity causality but localize it differently or cannot isolate one path |
| `invalid_paired_token_audit` | Pair construction, support, tokenizer, or checkpoint provenance is invalid |

Every branch reports:

```text
training_authorized: false
full_phase34_evaluation_authorized: false
checkpoint_promotion_authorized: false
```

Phase 34j identifies the next single variable. It does not authorize that
experiment itself, and Phase 33c remains the last accepted resolver
checkpoint.
