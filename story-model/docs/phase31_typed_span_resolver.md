# Phase 31: typed whole-span resolution

Phase 30 proved that the pointer mechanism can copy exact bytes in a
primitive task, but its free-running copy gate was almost never invoked on
natural lexical and transfer prompts. Phase 31 tests the documented fallback:
select a complete, typed evidence candidate and realize it deterministically.
It does not authorize a longer Phase 30 run.

## Controlled question

Can a separate resolver learn all three structured actions while retaining
exact unfamiliar values?

- `resolve`: select one compatible candidate as a whole unit.
- `clarify`: fail closed when decisive evidence is missing or candidates have
  the wrong type.
- `generate`: leave ordinary response generation to the language model.

For `resolve`, the selected source string is inserted into one fixed atomic
placeholder, `<|resolved_value|>`. The language model never has to spell that
value token by token.

## Architecture and scope

The resolver is a separate checkpoint wrapped around the unchanged
foundation-v3 Transformer. It has an action head and a shared scalar candidate
scorer. Each candidate is evaluated in its own marked view: the candidate's
exact text is replaced by `<|candidate|>` before scoring. This makes the score
depend on the relationship between the evidence and the candidate position,
not on memorizing the spelling or fixed inventory index.

The model receives at most two annotated candidates. It selects an entire
candidate index; deterministic code then inserts the original source text into
the response frame. A predicted `resolve` with no compatible candidate is
converted to `clarify`.

This experiment deliberately assumes an upstream annotation layer supplies
candidate boundaries and types. It is not yet an arbitrary span extractor.
The two Phase 31 types are:

- `color` for `supplied_fact`
- `route` for `scene_route`

## Experimental isolation

- Phase 30 source rows and answer keys are reused exactly when available.
- If those files are absent, the builder reconstructs the Phase 28-equivalent
  semantic-transfer splits.
- The Phase 30 pointer model, configs, checkpoints, and evaluation remain
  untouched.
- Both Phase 31 runs warm-start from foundation-v3, not from a Phase 30
  checkpoint.
- Candidate inventories and ordering are identical within each
  counterfactual pair; only the selected label flips.
- Every source pair contributes two `resolve` rows, one `clarify` control, and
  one `generate` control.

The default dataset contains:

| Split | Resolve | Clarify | Generate | Total |
|---|---:|---:|---:|---:|
| train | 3,200 | 1,600 | 1,600 | 6,400 |
| val | 400 | 200 | 200 | 800 |
| lexical | 400 | 200 | 200 | 800 |
| paraphrase | 400 | 200 | 200 | 800 |
| transfer | 400 | 200 | 200 | 800 |

## Build and primitive proof

From `story-model/`:

```bash
python scripts/build_typed_span_resolver_dataset.py
python scripts/overfit_typed_span_resolver.py --device mps
```

Do not start a natural-task run unless the primitive reaches 100% structured
training accuracy and 8/8 exact held-out resolutions. The held-out cases use
unseen pairings of familiar pieces, so the test specifically checks candidate
selection and deterministic realization.

## Bounded smoke and pilot

Run the 100-update wiring smoke test:

```bash
python scripts/train_typed_span_resolver.py \
  --config configs/typed_span_resolver_smoke.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Confirm MPS selection, vocabulary expansion from 512 to 525, the four new
resolver parameters, finite losses/gradients, and both checkpoints. Then run
the bounded 1,000-update pilot from foundation-v3 again:

```bash
python scripts/train_typed_span_resolver.py \
  --config configs/typed_span_resolver_pilot.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Do not warm-start the pilot from the smoke checkpoint and do not extend the
run beyond 1,000 updates before evaluating autonomous structured decisions.

## Evaluation and gate

Evaluate the best pilot checkpoint across all five splits:

```bash
python scripts/evaluate_typed_span_resolver.py \
  --checkpoint checkpoints/typed_span_resolver_pilot/best.pt \
  --data-dir data/character/typed_span_resolver \
  --device mps \
  --output runs/phase31-typed-span-resolver.jsonl

python scripts/gate_typed_span_resolver.py \
  --summary runs/phase31-typed-span-resolver.summary.json
```

The gate is intentionally stricter than the earlier free-running language
generation gate because exact realization is deterministic once selection is
correct.

| Split | Resolve accuracy | Pair accuracy | Clarify | Generate | Missing | Wrong alternative |
|---|---:|---:|---:|---:|---:|---:|
| train | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| val | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| lexical | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| paraphrase | >=95% | >=90% | >=95% | >=95% | <=2% | <=2% |
| transfer | >=90% | >=80% | >=95% | >=95% | <=2% | <=2% |

If lexical values remain missing after the primitive passes, inspect action
and candidate accuracy separately. If candidate selection succeeds but action
routing fails, the next experiment should target routing supervision. If both
fail, candidate extraction/type annotations or the candidate scorer—not a
longer run—are the next controlled variables.
