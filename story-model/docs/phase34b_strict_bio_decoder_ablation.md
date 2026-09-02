# Phase 34b: strict BIO decoder ablation

Phase 34 did not produce an eligible checkpoint. By update 400--600, weighted
byte loss had largely flattened while exact span precision and recall remained
far below their 0.98 floors. Type accuracy was 1.0 whenever a predicted
boundary matched a gold boundary, isolating the unresolved operation to span
boundaries rather than type classification.

The current permissive decoder promotes an orphan `I-type` tag to the beginning
of a new span. It also closes the active span and promotes a type-mismatched
`I-type` tag. A single stray inside tag can therefore become a complete false
positive even though it contributes very little to average byte loss.

Phase 34b changes no weights and performs no training. It decodes each existing
logits tensor twice:

| Transition | Permissive decoder | Strict decoder |
|---|---|---|
| `B-type` | Close and start | Close and start |
| Matching `I-type` | Extend | Extend |
| Orphan `I-type` | Start | Discard |
| Mismatched `I-type` | Close and start | Close and discard |
| `O` | Close | Close |

The permissive policy remains the default in the Phase 34 runtime. This
ablation does not silently change prior evaluation behavior.

## Checkpoint

Use the pilot's `diagnostic-ineligible.pt`, not `final.pt` and not a smoke
checkpoint:

```text
checkpoints/explicit_offset_candidate_proposer_pilot/diagnostic-ineligible.pt
```

For the reported pilot this checkpoint is from update 800, the last update
whose diagnostic-selection key improved. It remains explicitly ineligible.

## Run

From `story-model/`:

```bash
PYTHONUNBUFFERED=1 python scripts/ablate_explicit_offset_decoding.py \
  --checkpoint checkpoints/explicit_offset_candidate_proposer_pilot/diagnostic-ineligible.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34b-strict-bio-ablation.jsonl
```

The script runs the proposer once per batch and applies both decoders to the
same CPU logits. It evaluates every eligible Phase 31/32 row across train,
validation, lexical, paraphrase, and transfer splits. Existing `wrong_type`
rows remain outside the incomplete proposer-span annotations, as in Phase 34.

The summary is written to:

```text
runs/phase34b-strict-bio-ablation.summary.json
```

For each policy, split, and skill it reports:

- Exact span precision, recall, and F1
- Answer-candidate recall
- False-positive and false-negative span counts
- Orphan and type-mismatched inside-tag counts
- Invalid UTF-8 and truncated-span counts
- False-positive boundary displacement relative to the nearest same-type gold
  span

The displacement histogram clips offsets beyond eight bytes into `<=-9` and
`>=+9` buckets. `no_same_type_gold` means the false positive could not be
aligned to any gold span of its predicted type.

## Pre-registered decision

The script prints and records exactly one branch.

### `decoder_only_pass`

Strict decoding must meet all original Phase 34 proposal floors on every split
and skill:

- Aggregate span precision and recall at least 0.98
- Aggregate answer-candidate recall at least 0.995
- Per-skill span precision and recall at least 0.95
- Per-skill answer-candidate recall at least 0.98
- Boundary type accuracy at least 0.99
- Offset validity exactly 1.0
- Proposal overflow exactly 0.0

Only this branch authorizes predicted-candidate end-to-end evaluation under
strict decoding. It does not retroactively make the existing checkpoint
eligible until that evaluation and the complete Phase 33c oracle gate pass.

### `strict_helpful_but_insufficient`

This branch requires:

- Overall strict precision improves by at least 0.20 absolute.
- Strict recall loses no more than 0.02 on every dataset/split.
- Strict answer-candidate recall loses no more than 0.01 on every
  dataset/split.
- At least one original proposal gate still fails.

This establishes strict decoding as a useful correction but does not authorize
checkpoint promotion or the full gate. It permits one subsequent
single-variable boundary-loss experiment with strict decoding held fixed.

### `strict_decoder_rejected`

Any smaller precision improvement or excessive recall/answer loss rejects the
strict policy. Retain permissive decoding and diagnose boundary supervision.

## Stop conditions

- Do not modify `tag_class_weights` during this ablation.
- Do not retrain either checkpoint.
- Do not apply both a decoder change and a loss-weight change in one run.
- Do not promote `diagnostic-ineligible.pt`.
- Do not run the Phase 34 end-to-end gate unless the recorded branch is
  `decoder_only_pass`.
