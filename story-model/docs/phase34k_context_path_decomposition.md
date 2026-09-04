# Phase 34k: contextual-path decomposition

Phase 34j established a strong paired token-identity effect in Phase 34i and
a near-threshold effect in Phase 34d. It also showed that the effect travels
through the contextual term, while the direct raw-byte embedding contributes
approximately none of it.

That contextual term is not atomic:

```text
proposal_key(h_current_token) + proposal_query(h_final_resolver_token)
```

Phase 34j swapped both hidden-state sources together. Phase 34k separates them
with one final frozen audit. It changes no training data, loss, decoder,
checkpoint, or model parameter.

## Premise and pair contract

The audit requires the frozen Phase 34j summary and recomputes its decision.
It proceeds only when that decision remains
`checkpoint_specific_identity_interaction_indicated` and all of Phase 34j's
authorization flags remain false.

It rebuilds the same Phase 31 lexical and transfer `scene_route` pairs, then
requires:

- at least 400 valid pairs and 200 pairs from each source token;
- zero invalid focus pairs;
- identical focus-token IDs, data hashes, and row counts;
- the exact Phase 34d and Phase 34i checkpoint hashes from Phase 34j; and
- matching tokenizer and architecture provenance.

This prevents the decomposition from silently changing the contexts that
produced the Phase 34j result.

## Five frozen conditions

The two original and two swapped sequences are evaluated in one batched
backbone call. Only tensor indexing and recombination differ afterward.

| Condition | Local key input | Final query input | Direct byte |
|---|---|---|---|
| `original` | Original | Original | Original |
| `local_key_swap_only` | Swapped | Original | Original |
| `query_swap_only` | Original | Swapped | Original |
| `full_context_swap` | Swapped | Swapped | Original |
| `full_token_swap` | Swapped | Swapped | Swapped |

The `original` and `full_token_swap` endpoints are tested against normal model
inference. The three mixed rows are causal interventions, not valid prompt
encodings.

All margin shifts are normalized as rendered `ro` minus rendered `or`, in both
source-token strata. Negative B-minus-I shifts mean that the intervention made
the representation more `I`-like.

## Tests

From `story-model/`:

```bash
.venv/bin/python -m pytest -q
```

The tests cover both legacy and factorized heads, endpoint equivalence, one
backbone call per paired batch, bidirectional margin normalization, Phase 34j
provenance, support floors, non-additive shares, and every formal branch.

## Run

Use the same two checkpoints and data used by Phase 34j:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_context_path_intervention.py \
  --phase34d-checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/diagnostic-ineligible.pt \
  --phase34i-checkpoint checkpoints/explicit_offset_factorized_head_pilot/diagnostic-ineligible.pt \
  --phase34j-summary runs/phase34j-paired-token-intervention.summary.json \
  --phase31-data-dir data/character/typed_span_resolver \
  --device mps \
  --batch-size 16 \
  --output runs/phase34k-context-path-intervention.summary.json
```

This writes the summary plus
`runs/phase34k-context-path-intervention.summary.rows.jsonl`.

Recompute and export the decision independently:

```bash
.venv/bin/python scripts/gate_context_path_intervention.py \
  --summary runs/phase34k-context-path-intervention.summary.json \
  --output runs/phase34k-context-path-intervention.decision.json
```

## Decision rules

The complete contextual effect must shift the normalized B-minus-I margin by
at least 0.25 in the expected direction. Within each source-token stratum and
checkpoint:

- a component is dominant at a share of at least 70%, provided the competing
  component is at most 30%;
- the effect is distributed when both components have shares of at least 30%;
  and
- every other result is unresolved.

These are post-`tanh`, post-head margin effects. They are not an additive
variance decomposition: shares need not sum to 100%, and either share may
exceed 100%. No clipping other than ignoring effects in the opposite direction
is used.

A substantive branch requires the same classification in both source-token
strata of both checkpoints.

| Branch | Next controlled action |
|---|---|
| `local_token_state_dominant` | Design a training-only compositional boundary counterbalance targeted at local token states, using refreshed token-disjoint held-out values |
| `resolver_query_state_dominant` | Audit a boundary-local or span-conditioned query that does not reuse the pooled final resolver state |
| `distributed_context_path_indicated` | Replace both BPE-context inputs with a boundary-specific contextual representation; do not return to the direct-byte identity path |
| `checkpoint_or_stratum_interaction_indicated` | Compare `proposal_key`/`proposal_query` feature distributions before selecting another model |
| `context_path_decomposition_insufficient` | Inspect per-pair margin interactions and projection activations without choosing an architecture |
| `invalid_context_path_audit` | Repair provenance, pair construction, or checkpoint joins before interpretation |

Every branch reports:

```text
training_authorized: false
full_phase34_evaluation_authorized: false
checkpoint_promotion_authorized: false
```

Phase 34k only identifies the next controlled variable. Phase 33c remains the
last accepted resolver checkpoint.
