# Phase 34: explicit-offset typed candidate proposal

Phase 33c passes every Phase 31/32 gate when a typed candidate inventory is
provided. Phase 34 removes that oracle inventory at runtime. A learned BIO
tagger proposes whole evidence spans directly from the prompt, carries their
UTF-8 byte offsets and types forward, and supplies the resulting inventory to
the frozen Phase 33c resolver.

This is an upstream proposal experiment. It does not change the successful
count-conditioned routing table, candidate scorer, action head, tokenizer, or
deterministic value realization.

## Controlled boundary

- The eligible Phase 33c `best.pt` checkpoint is loaded strictly and frozen.
- Only five new tensors train: `proposal_query.weight`,
  `proposal_key.weight`, `proposal_offset_embedding.weight`,
  `proposal_tag.weight`, and `proposal_tag.bias`.
- The tokenizer and all 525 existing token IDs remain unchanged.
- Proposal labels are byte-level typed BIO tags over the original prompt.
- Proposed values carry exact half-open UTF-8 byte offsets. A value cannot be
  admitted by a substring match at an unrelated occurrence.
- Lexical boundaries, expected type, unique value text, and the four-candidate
  capacity are checked before a proposed span enters the resolver.
- The supplied candidate inventory is used only offline to derive annotations.
  Candidate text is not serialized into the proposal model input.
- Exact source text is still realized deterministically after resolution; the
  model never spells an unfamiliar value token by token.
- Phase 31 and Phase 32 proposer examples remain balanced in every microbatch.
- The existing `wrong_type` controls are not proposer examples: their oracle
  inventories intentionally describe incompatible candidates and therefore do
  not define complete positive prompt-span annotations. They remain in the
  mandatory, complete Phase 33c oracle-candidate regression gate.

The learned proposer supports the five types already exercised by the current
skills: `action`, `color`, `container`, `person`, and `route`. This phase does
not claim arbitrary ontology induction, evidence polarity, multi-span
composition, or Vera-specific language behavior.

## Preflight

Run from `story-model/` with the successful Phase 33c pilot checkpoint:

```bash
pytest -q

python scripts/audit_explicit_offset_candidate_proposer.py \
  --checkpoint checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt

python scripts/overfit_explicit_offset_candidate_proposer.py --device mps
```

The audit must round-trip every eligible Phase 31/32 row from exact byte spans
to the same evidence-eligible candidate texts, validate every UTF-8 boundary,
and show that all encoded prompts fit the 1,024-token block. It reports
`wrong_type` exclusions separately.

The primitive must reach exact training-span F1, answer-candidate recall, and
offset validity of 1.0. Its held-out values are diagnostic; the natural lexical
and transfer splits remain the generalization gate.

## Smoke and bounded pilot

Start the 100-update smoke from Phase 33c, not from an older resolver:

```bash
python scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_candidate_proposer_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Confirm:

- MPS is selected and the proposal byte width is finite.
- The five new proposer tensors are the only trainable parameters.
- The frozen architecture is reported as
  `backbone, action_head, candidate_score`.
- The objective is `typed_byte_BIO`.
- Phase 31 and Phase 32 training counts are both nonzero.
- Losses and gradients remain finite.
- Validation span precision/recall, answer recall, type accuracy, offset
  validity, overflow, and checkpoint eligibility are printed.

Unlike Phase 33c, update 0 is expected to be ineligible because the proposer
heads are new. Do not require update-0 eligibility. Proceed only if the smoke
creates an eligible `best.pt`; an ineligible state is written separately as
`diagnostic-ineligible.pt`.

Then run the bounded 1,000-update pilot from the same Phase 33c checkpoint,
not from the smoke checkpoint:

```bash
python scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_candidate_proposer_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Do not extend training if the full gate fails.

## Evaluation

First preserve the complete oracle-candidate Phase 33c result. Reuse the
passing summary if it was produced from the same Phase 33c checkpoint, or
regenerate it:

```bash
PYTHONUNBUFFERED=1 python scripts/evaluate_unified_typed_span_resolver.py \
  --checkpoint checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 16 \
  --progress-every 50 \
  --output runs/phase33c-count-conditioned.jsonl
```

Evaluate the eligible Phase 34 `best.pt` with predicted candidates:

```bash
PYTHONUNBUFFERED=1 python scripts/evaluate_explicit_offset_candidate_proposer.py \
  --checkpoint checkpoints/explicit_offset_candidate_proposer_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34-explicit-offset-proposer.jsonl
```

The evaluator prints after each split and reports throughput. On MPS, increase
`--batch-size` only as memory permits; a larger batch reduces small-dispatch
overhead but multiplies the resolver's five option views.

Gate both summaries together:

```bash
python scripts/gate_explicit_offset_candidate_proposer.py \
  --summary runs/phase34-explicit-offset-proposer.summary.json \
  --phase33-summary runs/phase33c-count-conditioned.summary.json
```

## Gate

Every split and skill is checked. Aggregate proposal requirements are:

- Exact span precision and recall: at least 98%.
- Answer-candidate inventory recall: at least 99.5%.
- Boundary type accuracy: at least 99%.
- UTF-8 offset validity: 100%.
- More than four unique runtime candidates: 0%.

Predicted-candidate end-to-end behavior retains the Phase 33c thresholds:

| Split | Resolve | Pair |
|---|---:|---:|
| train | >=95% | >=90% |
| val | >=95% | >=90% |
| lexical | >=95% | >=90% |
| paraphrase | >=95% | >=90% |
| transfer | >=90% | >=80% |

Real-candidate top-1 and answer-inventory recall must be at least 99.5%; clarify
and sentinel accuracy at least 95%; generate at least 99%; mode at least 98%;
and missing, wrong-alternative, and false-sentinel rates at most 2%.

Interpret a failure by boundary:

- If exact proposal metrics fail, improve candidate proposal or annotations;
  do not retrain the resolver.
- If proposals pass but end-to-end resolution fails, inspect offset-to-inventory
  conversion and frozen-resolver interaction.
- If the oracle Phase 33c gate fails, reject the run as a resolver regression.
- If all gates pass, the next controlled question can add evidence polarity or
  one deferred reasoning family. It should not begin Vera-specific training by
  bundling several new mechanisms at once.
