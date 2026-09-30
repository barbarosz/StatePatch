# Two-hour execution runbook

## 0:00–0:10 — environment

```bash
cd StatePatch_MVP
export PYTHONPATH=$PWD:$PYTHONPATH
pip install -r requirements-core.txt
python scripts/smoke_test_core.py
python scripts/make_benchmark.py --out data --pairs-per-type 8
```

Success criterion: `StatePatch core smoke test: PASS` and 64 manifest rows.

## 0:10–0:25 — verify the model base

**Best path:** use Causal Forcing frame-wise, because its released checkpoint natively supports I2V. Confirm its own I2V inference works before adding hooks.

**Fallback plumbing path:** the public Self-Forcing release is chunk-wise T2V (`num_frame_per_block: 3`), so use `t2v-smoke` only to validate instrumentation.

## 0:25–0:50 — capture 16 examples

```bash
python scripts/run_self_forcing_capture.py \
  --model-root /path/to/Self-Forcing \
  --manifest data/statepatchbench.jsonl \
  --mode t2v-smoke --limit 16
```

This result is a software smoke test, **not evidence of memory**, because the text includes the state.

## 0:50–1:00 — probe heatmap

```bash
python scripts/train_probe_sweep.py
python scripts/plot_probe_heatmap.py
```

Inspect `results/probe_heatmap.png`.

## 1:00–1:40 — valid causal experiment

Use the released Causal Forcing frame-wise checkpoint, repeat capture in `--mode i2v`, select a high-signal call/block, then run `run_self_forcing_patch.py` on `color_000` and at least three more pairs.

Run alpha sweep: 0.25 / 0.5 / 0.75 / 1.0.

Then run three controls:

1. neighboring low-signal block;
2. wrong call;
3. donor from an unrelated pair.

## 1:40–2:00 — package evidence

Put into a single folder:

- `probe_heatmap.png`
- donor/unpatched/patched videos
- command log
- 4–8 example pairs
- table of intervention outcomes
- `paper/SHORT_PAPER_DRAFT.md`

If no valid I2V checkpoint is available, stop at the instrumentation/probe stage and write the result honestly as **infrastructure + preliminary confounded sanity check**. Do not turn a T2V text-conditioned result into a world-memory claim.
