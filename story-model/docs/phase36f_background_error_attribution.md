# Phase 36f: frozen background-error attribution audit

Phase 36e established a real background false-positive problem but could not
attribute it. Three specific gaps remained: it never ran the census against
Phase 35 (so it could not say whether the pattern predated identity-
invariance training at all), it recorded the row's crossed-panel focus
identity rather than the token that actually emitted each false span, and it
computed global positive-vs-outside confusion per row without exporting it.

Phase 36f closes all three. No training, weights, decoder, or threshold
changes.

## What it measures

1. **The same census against Phase 35** — how much of the pattern predates
   any identity-invariance training. Phase 35 never saw the swap mechanism.
2. **Actual emitting tokens and tags** — for every spurious span: the
   emitting input token id, its byte width, the byte offset *within* that
   token (a 1-byte predicted span need not come from a 1-byte input token),
   swap-pool membership, predicted type, and whether the span opens with a
   grammatically clean `B` or an orphan `I` (inside tag with no preceding
   begin). Recorded separately from the row's crossed-panel identity, which
   is not necessarily the culprit.
3. **Positive-vs-O across the whole prompt**, split by section and
   lexical/transfer, **normalized by gold-O bytes** — the actual opportunity
   for a background error, rather than total section length.

## Code

| File | Role |
|---|---|
| `scripts/audit_background_error_attribution.py` | The Phase 36f audit |
| `runs/phase36f-background-error-attribution.summary.json` | Output |

## Run

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_background_error_attribution.py \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --phase36b-checkpoint checkpoints/explicit_offset_paired_identity_invariance_pilot/diagnostic-ineligible.pt \
  --phase36d-checkpoint checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/diagnostic-ineligible.pt \
  --device mps --batch-size 8 \
  --output runs/phase36f-background-error-attribution.summary.json
```

## Findings

**The background false-positive problem predates identity-invariance
training.**

| | Phase 35 | Phase 36b | Phase 36d |
|---|---|---|---|
| Total spurious spans | 20,000 | 24,722 | 19,573 |
| Distinct emitting tokens | 48 | 61 | 48 |
| Opening kind: orphan `I` | 68% | 61% | 62% |

Phase 35 — before any swap mechanism existed — already produces more
background spurious spans than Phase 36d. 78% of Phase 35's and Phase 36d's
top-40 emitting tokens are the same token IDs, dominated by raw single-byte
ASCII letters (IDs 97–119 = `a`–`w`). Orphan-`I` openings dominate in all
three checkpoints including Phase 35.

**Positive-vs-outside, normalized by gold-O bytes (lexical / transfer):**

| Section | Phase 35 | Phase 36b | Phase 36d |
|---|---|---|---|
| `character` | 0.000% / 0.000% | 0.008% / 0.001% | 0.000% / 0.000% |
| `relationship` | 0.000% / 0.000% | 0.000% / 0.000% | 0.000% / 0.000% |
| `scene` | 0.58% / 0.32% | 0.56% / 0.33% | 0.38% / 0.18% |
| `memories` | 0.53% / 0.54% | 1.15% / 0.95% | 0.48% / 0.41% |
| `conversation` | 1.21% / 2.71% | 2.45% / 3.82% | 1.39% / 2.80% |
| `world_facts` | 5.51% / 5.84% | 6.21% / 6.86% | 5.23% / 5.51% |

`world_facts` is worst by roughly an order of magnitude in every checkpoint.
Phase 36b is worse than Phase 35 in `scene`, `memories`, `conversation`, and
`world_facts`; Phase 36d returns approximately to the Phase 35 level.

**One statistic in this audit is uninformative and is recorded as such.**
Swap-pool membership came out at 95.7–95.9% of emitting tokens for Phase
36b/36d — but the pool covers ~96% of the entire non-special vocabulary
(only 14 registered identities are excluded), so that figure is the base
rate, not evidence of swap-pool causation. It does not support the
swap-exposure hypothesis and should not be cited as if it did.

## Closure: documented partial success

Phase 36f closes with a **documented partial success**. Clean-anchor
supervision (Phase 36d):

- preserves the tested identity repair (14-identity B→I pass on the
  registered panel),
- reduces fragmentation versus Phase 36b (roughly halved), and
- brings background errors approximately back to Phase 35 levels, undoing
  the regression Phase 36b introduced.

**Status preserved unchanged:** the Phase 36d checkpoint remains rejected.
Phase 33c remains the accepted checkpoint. Complete-span extraction remains
the unresolved project blocker.

## Qualifications

- A pre-existing error does not imply an architecture defect. These results
  establish that background false positives were already present in Phase
  35. They do not establish their origin in Phase 34d, nor show that fixing
  them requires architectural change — training composition, supervision,
  and loss weighting can change this behavior with architecture and decoder
  held fixed.
- "Returns approximately to baseline" is supported; "matches or beats every
  section" is not. Phase 36d `conversation` exceeds Phase 35 (1.39% vs.
  1.21% lexical, 2.80% vs. 2.71% transfer), and Phase 36b `scene` on lexical
  is slightly lower than Phase 35 (0.56% vs. 0.58%).
- The overall −2.1% span-count difference versus Phase 35 is not an
  established improvement, given the checkpoint-step mismatch (Phase 35 is
  step 1000; Phase 36b/36d are step 800) and no replication.

## Deferred

Persistent background errors are deferred to a separately registered
training experiment, to be defined *after* Phase 36g so its effects remain
distinguishable from consistency strength. If they remain near baseline, the
next target is positive-versus-O discrimination on varied irrelevant text —
training-only hard negatives or a revised loss, architecture still fixed.
