# Phase 33: unified typed-span routing

Phase 32 preserved Phase 31 and learned perfect real-candidate ranking, but its
independent three-way action head rejected correct candidates on unfamiliar
`contradiction_correction`, `multi_turn_memory`, and `reference_tracking`
records. Saved-prediction oracle analysis showed real-candidate top-1 accuracy
of 100% in every skill, width, and split. The lexical/transfer failure followed
the action decision instead.

Phase 33 tests one controlled change: the candidate scorer also receives a
fifth `no_supported_candidate` option. The old action head is retained only for
`structured` versus `generate` routing:

| Mode/option result | Runtime action |
|---|---|
| `generate` mode | `generate` |
| Structured + real candidate 0--3 | `resolve` |
| Structured + no-support option | `clarify` |

The exact selected source string is still inserted deterministically. The
language model never spells the resolved value token by token.

## Controlled boundary

- The Phase 32 tokenizer and all 525 token IDs are unchanged.
- The backbone, `action_head`, and `candidate_score` parameter names and shapes
  are unchanged; the Phase 32 state dictionary loads strictly.
- Phase 32 shared inputs and all four real-candidate views are reused
  byte-for-byte.
- No dataset rebuild is required.
- Phase 31 and Phase 32 training rows remain balanced 50/50 in every
  microbatch.
- The only new model input is a no-support view. It marks any supplied
  candidate found in evidence and includes candidate/expected types, allowing
  missing-evidence and wrong-type controls to outrank unsupported real options.
- Real-candidate ranking is independently reported and gated at 99.5% so the
  new option cannot conceal regression in the already-proven scorer.

The legacy three logits remain in the checkpoint for strict compatibility.
Their resolve/clarify distinction is diagnostic only; Phase 33 uses
`logsumexp(resolve, clarify)` as the structured-mode logit and the existing
generate logit as the other mode.

## Preflight

Use the Phase 32 update-950 best checkpoint:

```bash
python scripts/audit_unified_typed_span_resolver.py \
  --checkpoint checkpoints/expanded_typed_span_resolver_pilot/best.pt

python scripts/overfit_unified_typed_span_resolver.py --device mps
```

The audit must pass every Phase 31/32 row and confirm that shared and real
candidate encodings are unchanged. The primitive must reach 100% training
structured accuracy, 8/8 held-out exact resolution, 8/8 real-candidate top-1,
and 4/4 held-out no-support selection.

## Smoke and bounded pilot

Run the 100-update wiring smoke test from the Phase 32 best checkpoint:

```bash
python scripts/train_unified_typed_span_resolver.py \
  --config configs/unified_typed_span_resolver_smoke.yaml \
  --warm-start checkpoints/expanded_typed_span_resolver_pilot/best.pt
```

Confirm:

- MPS is selected.
- Vocabulary remains 525.
- `new architecture parameters: none` is printed.
- The loss objective is `generate_mode+unified_structured_option`.
- Phase 31 and Phase 32 structured and sentinel metrics are nonzero and finite.
- Gradients remain finite and are clipped to 1.0.
- Both `best.pt` and `final.pt` are written.

Then run the bounded 1,000-update pilot from the Phase 32 best checkpoint again,
not from the smoke checkpoint:

```bash
python scripts/train_unified_typed_span_resolver.py \
  --config configs/unified_typed_span_resolver_pilot.yaml \
  --warm-start checkpoints/expanded_typed_span_resolver_pilot/best.pt
```

Do not extend the run if the gate fails. This phase changes the routing
abstraction, not the amount of training.

## Evaluation

Evaluate `best.pt`, not `final.pt`:

```bash
PYTHONUNBUFFERED=1 python scripts/evaluate_unified_typed_span_resolver.py \
  --checkpoint checkpoints/unified_typed_span_resolver_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 16 \
  --progress-every 50 \
  --output runs/phase33-unified-typed-span.jsonl

python scripts/gate_unified_typed_span_resolver.py \
  --summary runs/phase33-unified-typed-span.summary.json
```

The evaluator defaults to batch size 16 and prints progress during every split,
avoiding the silent, small-dispatch behavior of the Phase 32 evaluator.

## Gate

Every aggregate, skill, width, and skill-width cell is checked. Phase 31 is a
full regression gate; Phase 32 retains widths 2, 3, and 4.

| Split | Resolve | Pair |
|---|---:|---:|
| train | >=95% | >=90% |
| val | >=95% | >=90% |
| lexical | >=95% | >=90% |
| paraphrase | >=95% | >=90% |
| transfer | >=90% | >=80% |

Additional requirements:

- Real-candidate top-1 accuracy: at least 99.5%.
- Clarify and no-support sentinel accuracy: at least 95%.
- Generate accuracy: at least 99%.
- Mode accuracy: at least 98%.
- Missing, wrong-alternative, and false no-support rates: at most 2%.

If real-candidate ranking regresses, the new unified objective is rejected. If
ranking remains intact but the no-support option fails, inspect sentinel score
calibration and evidence marking; do not return to a detached three-way action
head. Only after this dual Phase 31/32 gate passes should the project attempt
upstream arbitrary candidate proposal.
