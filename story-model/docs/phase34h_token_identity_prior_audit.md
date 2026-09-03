# Phase 34h: token-identity and boundary-prior audit

Phase 34g's mandatory premise audit rejected end-of-token geometry before
training. Only 8.7% of the target errors occurred at token ends, no width-2
error occurred at its token's final byte, and token ends had one twentieth
the error rate of other positions. The residual is specifically concentrated
at byte offset zero inside width-2 BPE tokens.

Phase 34h is a frozen, audit-only experiment. It tests whether those failures
are explained by:

1. an exact two-byte token receiving mostly `I:route` rather than `B:route`
   supervision at byte offset zero during training;
2. an exact token identity that fails in lexical data and independently
   confirms in transfer data;
3. a first-byte identity that confirms across those splits even when the
   exact BPE token does not; or
4. no supported identity partition, indicating that the joint boundary/type
   head should be factorized.

No model forward pass, decoding change, training run, or checkpoint promotion
is part of this phase. The audit reconstructs exact token identities from the
Phase 34d checkpoint tokenizer and joins them to the already-frozen Phase 34e
start rows.

## Run

Use the same Phase 34d checkpoint that created the Phase 34e rows. The known
pilot diagnostic was saved at update 800:

```bash
.venv/bin/python scripts/audit_token_identity_priors.py \
  --checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/diagnostic-ineligible.pt \
  --start-rows runs/phase34e-begin-calibration.jsonl \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --output runs/phase34h-token-identity-audit.json
```

If Phase 34e used an eligible `best.pt`, provide that exact path instead. The
script automatically loads the adjacent
`runs/phase34e-begin-calibration.summary.json`; use `--start-summary` only if
the summary was moved.

The script rejects:

- a checkpoint other than the pre-geometry Phase 34d proposer;
- a Phase 34e summary that did not select the BPE-geometry branch;
- a checkpoint step different from the Phase 34e source step;
- any dataset/start-row metadata mismatch; and
- any changed token width or byte offset during the reconstructed join.

It writes:

- `runs/phase34h-token-identity-audit.json`, containing provenance, train/val
  label priors, familiar-start behavior, identity tables, and the formal
  decision; and
- `runs/phase34h-token-identity-audit.target.jsonl`, containing the enriched
  Phase 31 lexical/transfer `scene_route` start rows with exact token IDs,
  bytes, and offsets.

## Independent prior test

For every BPE identity present in the width-2/offset-0 target bucket, the
audit counts the label applied to byte offset zero at every occurrence of that
token in the Phase 31 and Phase 32 train and validation prompts. Train and
validation counts are reported separately. Only the **training** labels
classify a token as an inside-prior conflict:

- at least 10 `B:route` plus `I:route` observations; and
- at least 80% of those positive route labels are `I:route`.

The conflict hypothesis passes only when the preclassified tokens:

- and their complement each contain at least 20 target starts;
- capture at least 60% of target B-to-I errors;
- have at least three times the complement's error rate; and
- retain at least 10 starts per bucket and a two-times rate ratio separately
  in both lexical and transfer.

This test does not use held-out errors to decide which tokens have a training
prior, so it can support a narrowly counterbalanced-data experiment without
circularly choosing the problematic identities from the test result.

## Lexical-to-transfer confirmation

If the training-prior test fails, lexical acts as discovery and transfer as
confirmation. A token ID or first byte is discovered only with at least five
lexical starts, three lexical errors, and a lexical error rate of at least
50%. It confirms only when its transfer bucket and complement each contain at
least 20 starts, it captures at least 50% of transfer errors, and its transfer
error rate is at least three times the complement's.

Exact token identity is tested before first-byte identity. Token bytes are
also reported in hexadecimal because a BPE token and its exact byte sequence
are one-to-one in this tokenizer.

## Decision

| Branch | Meaning |
|---|---|
| `train_label_prior_conflict_indicated` | Independently defined train-time `I:route` priors explain the held-out errors; next test narrowly counterbalances those route boundary labels |
| `token_identity_concentration_indicated` | Exact lexical-discovered BPE identities confirm in transfer without a sufficient training-prior explanation; investigate the token/byte representation interaction |
| `byte_identity_concentration_indicated` | Exact tokens do not confirm, but first-byte identities do; test that byte-identity interaction as the sole next variable |
| `factorized_boundary_type_head_indicated` | No supported identity or prior partition explains the residual; separate boundary detection from type classification |
| `invalid_token_identity_audit` | Provenance, support, or row/token alignment is invalid; repair the audit inputs before interpreting it |

Every valid branch still reports `training_authorized: false`. Phase 34h
selects the next controlled experiment; it does not itself authorize a run.
Do not use Phase 34f or Phase 34g checkpoints for this audit, and do not run
the full Phase 34 proposer gate from its result.
