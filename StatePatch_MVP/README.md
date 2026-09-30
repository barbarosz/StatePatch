# StatePatch MVP

**StatePatch: Causal Discovery and Repair of World-State Representations in Autoregressive Video Diffusion Models**

This repository is a fast, research-oriented implementation of the minimum viable StatePatch project:

1. create controlled paired state examples;
2. capture pooled hidden activations over Self-Forcing transformer blocks and generator calls;
3. train linear probes to map where state is decodable;
4. capture a matched donor activation;
5. causally patch that activation into the counterfactual rollout;
6. compare the unpatched and patched future video.

## Scientific warning

`--mode t2v-smoke` is **not a valid memory experiment** because state is injected through text and can be read from cross-attention. It exists only to verify the software path.

The actual experiment requires an image/video-conditioned or continuation setting in which the later prompt is neutral about the state. The public `configs/self_forcing_dmd.yaml` uses `num_frame_per_block: 3`; the official inference implementation indicates I2V is tied to a frame-wise path. Therefore do not claim an I2V causal-memory result using the chunk-wise public checkpoint unless you have a compatible frame-wise/I2V checkpoint/config.

## 0. Recommended base: Causal Forcing frame-wise I2V

For the actual causal-memory experiment, use the released **Causal Forcing frame-wise** model. It is built on the same Wan/Self-Forcing code stack, exposes the same `generator.model.blocks` structure used by StatePatch, and its frame-wise checkpoint natively supports I2V. This avoids the main blocker in the public Self-Forcing release, whose standard config is chunk-wise.

Inside the Causal-Forcing repository, download the Wan 1.3B base and the released frame-wise checkpoint, then use:

```text
configs/causal_forcing_dmd_framewise.yaml
checkpoints/framewise/causal_forcing.pt
```

The StatePatch adapter accepts either repository through `--model-root`.

### Alternative: official Self-Forcing repo for plumbing tests

Use the official repository and checkpoint instructions. The official project reports Linux, 64 GB RAM, and an NVIDIA GPU with at least 24 GB VRAM as its tested baseline.

Expected assets inside that repo include:

```text
wan_models/Wan2.1-T2V-1.3B/
checkpoints/self_forcing_dmd.pt
configs/self_forcing_dmd.yaml
```

## 1. Install StatePatch core

From this folder:

```bash
pip install -r requirements-core.txt
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/smoke_test_core.py
```

## 2. Generate the controlled benchmark

```bash
python scripts/make_benchmark.py --out data --pairs-per-type 8
```

This writes 64 synthetic initial-state images covering color, position, count, and shape, paired with a prompt that does **not** restate the state value.

## 3A. Plumbing-only T2V smoke test

This path works with the released chunk-wise T2V checkpoint but is scientifically confounded:

```bash
python scripts/run_self_forcing_capture.py \
  --model-root /path/to/Self-Forcing \
  --config configs/self_forcing_dmd.yaml \
  --checkpoint checkpoints/self_forcing_dmd.pt \
  --manifest data/statepatchbench.jsonl \
  --mode t2v-smoke \
  --limit 16
```

If this succeeds, hooks and feature capture work.

## 3B. Real state-memory experiment

Recommended: use the released frame-wise/I2V Causal Forcing checkpoint/config:

```bash
python scripts/run_self_forcing_capture.py \
  --model-root /path/to/Causal-Forcing \
  --config configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint checkpoints/framewise/causal_forcing.pt \
  --manifest data/statepatchbench.jsonl \
  --mode i2v \
  --limit 64
```

The script intentionally stops if `num_frame_per_block != 1` so a T2V/chunk-wise run is not accidentally presented as causal visual memory evidence.

## 4. Probe sweep

```bash
python scripts/train_probe_sweep.py --features results/features --out results/probe_sweep.csv
python scripts/plot_probe_heatmap.py --csv results/probe_sweep.csv --state-type color --out results/probe_heatmap_color.png
```

Choose candidate `(call, block)` locations that show high balanced accuracy while the object/state is no longer directly visible.

## 5. Killer causal experiment

For a valid I2V pair:

```bash
python scripts/run_self_forcing_patch.py \
  --model-root /path/to/Causal-Forcing \
  --config configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint checkpoints/framewise/causal_forcing.pt \
  --manifest data/statepatchbench.jsonl \
  --pair-id color_000 \
  --block 17 \
  --call 12 \
  --alpha 1.0 \
  --out results/patch_color_000
```

Outputs:

```text
donor.mp4
target_unpatched.mp4
target_patched.mp4
donor_activation.pt
```

A meaningful result is not merely that the patched video changes. The **target state should move toward the donor state while unrelated content remains stable**, and matched random/wrong-layer/wrong-pair controls should not show the same effect.

## 6. Two-hour objective

The fastest defensible deliverable is:

- working benchmark;
- validated activation instrumentation;
- one activation heatmap (even if initially T2V smoke, clearly labelled as confounded);
- causal-patching pipeline;
- one valid I2V patch experiment if compatible weights already exist;
- otherwise a paper-ready methods/results scaffold with no fabricated claims.

Never report a result you did not actually run.
