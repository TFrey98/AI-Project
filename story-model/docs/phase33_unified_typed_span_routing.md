# Phase 33c: count-conditioned unified typed-span routing

Phase 32 preserved Phase 31 and learned perfect raw candidate ranking, but its
independent action head rejected correct candidates on unfamiliar Phase 32
records. Phase 33b then tried to make the no-support sentinel available only
when no candidate text occurred in the prompt. That rule exposed a structural
mistake: `scene_route` always names both routes, including when the blocking
fact needed to choose between them is absent. All of its clarify rows therefore
had two lexically present candidates and an unrepresentable sentinel target.

The existing data provides a cleaner controlled boundary. Across every Phase
31/32 skill and split, evidence-eligible candidate counts partition exactly:

| Eligible real candidates | Rows in the current corpus | Structured route |
|---:|---|---|
| 0 | Non-`scene_route` clarify | Deterministic `clarify` |
| 1 | Non-`scene_route` resolve | Deterministic `resolve` |
| 2+ | `scene_route` resolve and clarify | Legacy action head decides |

`generate` remains a separate mode and overrides the structured route. When a
multi-eligible row is resolved, the candidate scorer ranks only the eligible
real candidates. Exact source text is still inserted deterministically; the
language model never spells the value token by token.

## Controlled boundary

- The Phase 32 tokenizer and all 525 token IDs are unchanged.
- The backbone, `action_head`, and `candidate_score` names and shapes are
  unchanged; the Phase 32 state dictionary loads strictly.
- Phase 32 shared inputs and all four candidate views are reused byte-for-byte.
- No dataset rebuild or new architecture parameter is required.
- Evidence eligibility means a type-compatible exact lexical mention. It means
  the candidate is addressable, not that the prompt asserts it as the answer.
- The encoded sentinel is available on every structured row. Runtime routing,
  rather than lexical presence alone, decides whether it can be selected.
- Candidate-count routing is deterministic for zero and one eligible candidate.
  The legacy resolve/clarify logits are used only for two or more.
- The structured-versus-generate mode loss is computed over **all rows**. Only
  the legacy resolve-versus-clarify loss is restricted to multi-eligible rows.
  These objectives are orthogonal.
- Raw candidate ranking remains separately supervised on every resolve row,
  including deterministic one-candidate rows, and is gated at 99.5%.
- Phase 31 and Phase 32 rows remain balanced 50/50 in every microbatch.
- Fixed stratified train and validation panels retain four evenly spaced rows
  from every skill/action/case/width cell and evaluate in batches of 16.
- A checkpoint can become `best.pt` only if the Phase 31 panel retains the
  configured structured, resolve-option, raw-ranking, sentinel,
  multi-candidate-action, and mode floors.

The current files do not contain arbitrary source spans or evidence polarity.
A future candidate proposer should carry explicit offsets and relations rather
than treating lexical mention as truth.

## Preflight

Use the Phase 32 update-950 best checkpoint:

```bash
python scripts/audit_unified_typed_span_resolver.py \
  --checkpoint checkpoints/expanded_typed_span_resolver_pilot/best.pt

python scripts/overfit_unified_typed_span_resolver.py --device mps
```

The audit must pass every Phase 31/32 row, preserve the shared and real-candidate
encodings exactly, and print the lossless routing partition: `scene_route`
resolve and clarify rows have two eligible candidates; other resolve rows have
one; other clarify and generate rows have zero.

The primitive deliberately gives both resolve and clarify rows two mentioned
candidates. It must reach 100% structured training accuracy, 8/8 held-out exact
resolution, 8/8 raw candidate top-1, 4/4 sentinel selection, and 12/12
multi-candidate action decisions.

## Smoke and bounded pilot

Run the 100-update smoke test from the Phase 32 best checkpoint:

```bash
python scripts/train_unified_typed_span_resolver.py \
  --config configs/unified_typed_span_resolver_smoke.yaml \
  --warm-start checkpoints/expanded_typed_span_resolver_pilot/best.pt
```

Confirm:

- MPS is selected and vocabulary remains 525.
- `new architecture parameters: none` is printed.
- The objective is
  `generate_mode+multi_candidate_action+raw_candidate_ranking`.
- Fixed validation panel sizes remain constant.
- Phase 31 raw-candidate and multi-candidate-action metrics are reported.
- Update 0 is checkpoint-eligible; every saved best also reports eligible.
- Ineligible states are written only to `diagnostic-ineligible.pt`.
- Losses and gradients remain finite, and both `best.pt` and `final.pt` exist.

Phase 33c uses new checkpoint directories so Phase 33b artifacts cannot be
mistaken for count-conditioned results:

```text
checkpoints/count_conditioned_unified_typed_span_resolver_smoke/
checkpoints/count_conditioned_unified_typed_span_resolver_pilot/
```

Do not begin the pilot unless update 0 is eligible and every saved new best is
eligible. If the smoke passes, run the bounded 1,000-update pilot from the Phase
32 best checkpoint again, not from the smoke checkpoint:

```bash
python scripts/train_unified_typed_span_resolver.py \
  --config configs/unified_typed_span_resolver_pilot.yaml \
  --warm-start checkpoints/expanded_typed_span_resolver_pilot/best.pt
```

Do not extend the run if the gate fails.

## Evaluation

Evaluate `best.pt`, not `final.pt`:

```bash
PYTHONUNBUFFERED=1 python scripts/evaluate_unified_typed_span_resolver.py \
  --checkpoint checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 16 \
  --progress-every 50 \
  --output runs/phase33c-count-conditioned.jsonl

python scripts/gate_unified_typed_span_resolver.py \
  --summary runs/phase33c-count-conditioned.summary.json
```

The evaluator reports the raw candidate ranking and multi-candidate action
decision separately. This prevents deterministic zero/one routing from hiding
regression in either learned component.

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

- Raw candidate top-1 accuracy: at least 99.5%.
- Multi-candidate action accuracy: at least 95% where such rows exist.
- Clarify and sentinel accuracy: at least 95%.
- Generate accuracy: at least 99%.
- Mode accuracy: at least 98%.
- Missing, wrong-alternative, and false-sentinel rates: at most 2%.

If raw ranking or multi-candidate action accuracy regresses, reject this
objective rather than hiding the error behind deterministic routing. Only after
the complete Phase 31/32 gate passes should the project attempt upstream
arbitrary candidate proposal.
