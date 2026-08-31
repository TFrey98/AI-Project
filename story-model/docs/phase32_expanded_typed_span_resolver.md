# Phase 32: expanded typed whole-span resolution

Phase 31 was the first Phase 26--31 experiment to pass every split. Phase 32
keeps that mechanism fixed and asks one narrower question: does the same
resolver generalize from two clean factual skills to a family of direct
single-span neutral-conversation skills when the candidate inventory grows
from two to as many as four?

This is a continuation of the successful Phase 31 checkpoint, not a return to
free-running lexical generation and not a new architecture search.

## Included family

Phase 32 adds four skills that each resolve one exact value already present in
the prompt:

| Skill | Required type | Source |
|---|---|---|
| `multi_turn_memory` | `container` | Earlier conversation turn |
| `contradiction_correction` | `person` | Corrected world fact |
| `reference_tracking` | `container` | Object-transfer world fact |
| `promise_recall` | `action` | Conversation and stored memory |

Four other neutral skills are deliberately deferred:

- `comparison` requires selecting after a numeric comparison.
- `cause_and_effect` requires multiple spans or a structured proposition.
- `missing_information` is primarily absence/policy routing.
- `privacy_boundary` requires refusal plus safe partial disclosure.

Combining those operations with direct span expansion would make a failed gate
ambiguous.

## Controlled changes

- Candidate width varies deterministically across 2, 3, and 4.
- Candidate types, boundaries, and exact source strings remain supplied by the
  upstream annotation layer.
- Every source row produces a matched counterfactual twin with the same
  candidate inventory and a flipped evidence value.
- Same-type distractors do not occur in the prompt.
- Candidate order rotates, and every legal position is represented at every
  width.
- One `clarify` and one `generate` control accompany every counterfactual pair.
- Lexical and paraphrase axes are factorized independently.

Nothing changes in the learned architecture. The Phase 32 resolver has the
same state-dictionary keys and parameter shapes as Phase 31. Variable candidate
width is a batch dimension, not a new head. The tokenizer remains the Phase 31
525-token tokenizer.

## Regression protection

Training warm-starts from:

```text
checkpoints/typed_span_resolver_pilot/best.pt
```

The training mixture contains all 6,400 Phase 31 train rows and all 6,400 new
Phase 32 train rows, with every microbatch split between the two phases. Phase
31 and Phase 32 validation losses are measured
separately and weighted equally when selecting `best.pt`, so the larger new
validation set cannot hide old-skill forgetting. Evaluation reruns all five
Phase 31 splits and all five Phase 32 splits. The final gate checks
every skill, candidate width, and skill-by-width cell so aggregate performance
cannot conceal one failed capability.

## Build and primitive proof

From `story-model/`:

```bash
python scripts/build_expanded_typed_span_dataset.py
python scripts/overfit_expanded_typed_span_resolver.py --device mps
python scripts/audit_expanded_typed_span_dataset.py \
  --checkpoint checkpoints/typed_span_resolver_pilot/best.pt
```

Default Phase 32 data sizes are:

| Split | Resolve pairs | Resolve rows | Clarify | Generate | Total |
|---|---:|---:|---:|---:|---:|
| train | 1,600 | 3,200 | 1,600 | 1,600 | 6,400 |
| val | 400 | 800 | 400 | 400 | 1,600 |
| lexical | 400 | 800 | 400 | 400 | 1,600 |
| paraphrase | 400 | 800 | 400 | 400 | 1,600 |
| transfer | 400 | 800 | 400 | 400 | 1,600 |

Do not run the natural-task smoke test unless the primitive reaches 100%
structured training accuracy and 8/8 exact held-out four-candidate resolution.
The tokenizer audit must also encode every Phase 31 and Phase 32 row within
the 1,024-token block using the actual Phase 31 tokenizer.

## Bounded smoke and pilot

Run the 100-update smoke test from the Phase 31 best checkpoint:

```bash
python scripts/train_expanded_typed_span_resolver.py \
  --config configs/expanded_typed_span_resolver_smoke.yaml \
  --warm-start checkpoints/typed_span_resolver_pilot/best.pt
```

Confirm:

- MPS is selected.
- The tokenizer remains at 525 tokens.
- `new architecture parameters: none` is printed.
- Phase 31 and Phase 32 row counts are both nonzero.
- Candidate widths report 2, 3, and 4.
- Losses and gradients remain finite.
- `best.pt` and `final.pt` are written.

Then run the bounded 1,000-update pilot from the Phase 31 best checkpoint
again, not from the smoke checkpoint:

```bash
python scripts/train_expanded_typed_span_resolver.py \
  --config configs/expanded_typed_span_resolver_pilot.yaml \
  --warm-start checkpoints/typed_span_resolver_pilot/best.pt
```

Do not extend beyond 1,000 updates before the dual gate is evaluated.

## Evaluation and gate

```bash
python scripts/evaluate_expanded_typed_span_resolver.py \
  --checkpoint checkpoints/expanded_typed_span_resolver_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --output runs/phase32-expanded-typed-span.jsonl

python scripts/gate_expanded_typed_span_resolver.py \
  --summary runs/phase32-expanded-typed-span.summary.json
```

The following thresholds apply both to aggregate results and to each skill:

| Split | Resolve | Pair | Clarify | Generate | Missing | Wrong alternative |
|---|---:|---:|---:|---:|---:|---:|
| train | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| val | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| lexical | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| paraphrase | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| transfer | >=90% | >=80% | >=95% | >=95% | <=2% | <=2% |

Phase 32 passes only if the original Phase 31 skills also retain these gates.
If the new family passes but Phase 31 regresses, adjust replay balance or
optimization; do not proceed. If two-candidate cases pass but wider candidate
sets fail, the next controlled variable is candidate-set ranking. Only after
the entire direct-span family passes should the project test derived,
multi-span, or policy-routing skills.
