# Phase 23: performance diagnostics and run monitoring

Long runs now have two complementary diagnostic layers:

1. `evaluate.py` reports its actual device immediately, encodes only the
   requested split, and emits throttled progress, elapsed time, throughput,
   and ETA during both BPE encoding and model scoring.
2. `run_with_monitor.py` is a separate parent process. It keeps working even
   if training/evaluation stops producing output, recording child CPU, RSS,
   process state, system memory, load, disk space, output silence, and monitor
   sample gaps.

The external monitor is intentionally standard-library-only. It does not need
administrator privileges, does not inspect model data, and samples only once
every ten seconds by default. Existing training output continues to provide
tokens/second, gradient norm, learning rate, and MPS tensor memory.

## Monitored training

Run from the `story-model/` directory:

```bash
python scripts/run_with_monitor.py \
  --name foundation-v2-train \
  -- \
  python -m story_model.train \
  --config configs/transformer_foundation_v2.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

The monitor creates a timestamped directory such as:

```text
runs/20260820-143500-foundation-v2-train/
  metadata.json
  metrics.jsonl
  output.log
  status.json
```

- `metadata.json` records the command, platform, working directory, Git
  commit, and whether the worktree was dirty.
- `output.log` is a complete copy of stdout/stderr while output still appears
  normally in the terminal.
- `metrics.jsonl` contains one resource sample per line for later analysis.
- `status.json` is atomically refreshed and always contains the latest known
  state and last output line.

The monitor warns when no child output has appeared for 120 seconds or when
its own sampling interval has a large gap, which can indicate system sleep or
a substantial hitch. Pressing `Ctrl-C` forwards the interrupt to the child and
marks the run interrupted without touching existing checkpoints.

## Monitored evaluation

Run only one MPS evaluation at a time:

```bash
python scripts/run_with_monitor.py \
  --name foundation-v2-new-val \
  -- \
  python -m story_model.evaluate \
  --checkpoint checkpoints/transformer_foundation_v2/best.pt \
  --data-config configs/transformer_foundation_v2.yaml \
  --split val \
  --device mps
```

Progress defaults to one line every ten seconds. Change this without changing
the numerical result:

```bash
python -m story_model.evaluate ... --progress-interval 30
```

`--no-progress` disables progress lines for automation. The final result now
also reports total evaluation time.

## Summarize and share a run

```bash
python scripts/summarize_run.py \
  runs/20260820-143500-foundation-v2-train
```

The summary includes peak child CPU/RSS, longest output silence, telemetry
hitches, exit state, Git identity, and the final output lines. When a run acts
strangely, preserve the entire timestamped directory; `metadata.json`,
`status.json`, the last portion of `output.log`, and the summary are normally
enough to distinguish slow tokenization, MPS scoring, memory pressure, process
failure, or a machine sleep/hitch.

## Interpreting common patterns

| Observation | Likely meaning |
|---|---|
| High CPU, low MPS activity during `encode` | BPE tokenization is active |
| Regular scoring progress and MPS memory | Accelerator evaluation is active |
| Process alive but output stale beyond threshold | Possible native stall or missing progress instrumentation |
| Large telemetry sample gap | Mac slept or the whole system hitched |
| RSS rises continually across repeated stages | Possible host-memory growth |
| MPS tensor memory rises continually in training logs | Possible accelerator allocation growth |
| Exit code nonzero with normal resource metrics | Read the final exception in `output.log` |

The vectorized BPE encoder preserves the original 256 merge ranks and exact
token sequence. It accelerates corpus-sized inputs but deliberately retains the
simple Python path for short prompts where NumPy setup would cost more than it
saves.
