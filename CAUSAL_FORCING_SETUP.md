# Recommended model setup: Causal Forcing frame-wise

The actual StatePatch experiment needs visual initialization with a neutral text continuation. Causal Forcing is the most direct drop-in base because its released frame-wise Wan2.1-1.3B model natively supports I2V.

From the official Causal-Forcing repository:

```bash
conda create -n causal_forcing python=3.10 -y
conda activate causal_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation
python setup.py develop

hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download zhuhz22/Causal-Forcing framewise/causal_forcing.pt --local-dir checkpoints
```

Then, from StatePatch_MVP:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/run_self_forcing_capture.py \
  --model-root /path/to/Causal-Forcing \
  --config configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint checkpoints/framewise/causal_forcing.pt \
  --manifest data/statepatchbench.jsonl \
  --mode i2v --limit 16
```

Then:

```bash
python scripts/train_probe_sweep.py
python scripts/plot_probe_heatmap.py --state-type color --out results/probe_heatmap_color.png
```

Select a high-signal call/block and run:

```bash
python scripts/run_self_forcing_patch.py \
  --model-root /path/to/Causal-Forcing \
  --config configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint checkpoints/framewise/causal_forcing.pt \
  --manifest data/statepatchbench.jsonl \
  --pair-id color_000 --block 17 --call 12 --alpha 1.0 \
  --out results/patch_color_000
```

The `block` and `call` values above are placeholders until the probe sweep is run; choose them from actual measurements.
