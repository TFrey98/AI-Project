# Phase 36a: crossed boundary identity-context audit

Phase 35 was correctly rejected. Its training-only counterbalance reduced the
original Phase 31 lexical/transfer `scene_route` B-to-I error from roughly
21â23% to 7.6â8.9%, but did not reach the registered 2% ceiling and introduced
retained Phase 32 type regressions. Its refreshed panel also separated sharply
by identity: all eight trained focus identities had zero focus B-to-I errors,
while three of four held-out identities remained strongly biased. The fourth
held-out identity succeeded in both lexical and transfer.

The complete-span result is more severe than the focus statistic alone. On
the trained-identity panels, predicted span counts were roughly 1.9â2.1 times
the gold counts, and exact-span precision ranged from 15.6% to 52.2%. Transfer
also lost 8.7% of gold inside bytes to O. Phase 35 therefore repaired selected
start decisions without learning a generally safe span proposal rule.

Phase 34k remains a valid path localization: the paired effect traveled
through `proposal_key(h_current_token)`, not the pooled query. It did not show
whether the information in that local key was a transferable rule, an
identity prior, or an identity-by-context interaction. Phase 36a separates
those explanations without training.

## Frozen contract

- Keep Phase 33c as the accepted resolver checkpoint.
- Treat the Phase 35 checkpoint as diagnostic-ineligible.
- Require the official `boundary_counterbalance_rejected` decision with both
  promotion and the full Phase 34 gate withheld.
- Compare the frozen Phase 34d and Phase 35 proposer checkpoints.
- Change no model parameter, architecture, decoder, loss, or dataset on disk.
- Use the exact Phase 31 lexical and transfer files frozen by the Phase 35
  manifest.
- Cross all eight trained, four held-out, and legacy `or`/`ro` identities
  through every supported `scene_route` pair in both target splits.
- Use one shared route scaffold for all identities. Every crossed prompt must
  have the same token count and differ from its peers only at positive focus
  positions, where one canonical two-byte token replaces another.
- Require at least 20 B and 20 I observations per identity per split and zero
  invalid matched contexts.

The audit reports complete-span metrics, exact focus-token B/I confusion, and
the typed route B-minus-I logit margin for every model, split, and identity.

## Run

From `story-model/`:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_crossed_boundary_identity.py \
  --phase34d-checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/diagnostic-ineligible.pt \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --phase35-retained-audit runs/phase35-retained-tag-audit.summary.json \
  --phase35-counterbalance-audit runs/phase35-boundary-counterbalance-audit.summary.json \
  --phase35-decision runs/phase35-boundary-counterbalance-decision.json \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase35-data-dir data/character/boundary_counterbalance \
  --device mps \
  --batch-size 8 \
  --output runs/phase36a-crossed-boundary-identity.summary.json
```

Recompute the decision independently:

```bash
.venv/bin/python scripts/gate_crossed_boundary_identity.py \
  --summary runs/phase36a-crossed-boundary-identity.summary.json \
  --output runs/phase36a-crossed-boundary-identity.decision.json
```

## Registered interpretations

An identity passes only when its crossed B-to-I rate is at most 2% in both
lexical and transfer. A severe identity has at least 10% B-to-I in both.
Complete-span precision or recall losses greater than two percentage points
from Phase 34d are reported separately; no branch can promote Phase 35.

| Branch | Interpretation |
|---|---|
| `trained_identity_memorization_dominant` | All eight trained identities pass in matched contexts while at least three held-out identities remain severe; design one paired identity-invariance objective rather than adding undifferentiated rows |
| `context_distribution_interaction_indicated` | At least three originally failing held-out identities pass when context is crossed; redesign identity coverage across splits |
| `heterogeneous_identity_priors_confirmed` | Passing and severe held-out identities coexist after context matching; use the paired margins to isolate the prior geometry |
| `broad_cross_identity_transfer` | Every trained, held-out, and legacy identity passes; the original Phase 35 failure was contextual rather than identity-specific |
| `mixed_crossed_identity_result` | No registered explanation dominates; inspect margins and span regressions before another training change |
| `invalid_crossed_identity_audit` | Repair provenance, support, or context matching before interpretation |

Every branch keeps training, decoder change, the inherited full Phase 34 gate,
and checkpoint promotion unauthorized. A later training phase must register one
new variable from the selected interpretation.
