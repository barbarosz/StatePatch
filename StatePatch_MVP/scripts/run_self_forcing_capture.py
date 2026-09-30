#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchvision.io import write_video
from statepatch.benchmark import read_manifest
from statepatch.hooks import ModelCallCounter, PooledActivationRecorder
from statepatch.self_forcing_adapter import load_pipeline, load_initial_image_latent, run

p = argparse.ArgumentParser(description="Capture pooled block activations from Self-Forcing on StatePatchBench")
p.add_argument("--model-root", "--self-forcing-root", dest="model_root", required=True)
p.add_argument("--config", default="configs/self_forcing_dmd.yaml")
p.add_argument("--checkpoint", default="checkpoints/self_forcing_dmd.pt")
p.add_argument("--manifest", default="data/statepatchbench.jsonl")
p.add_argument("--features-out", default="results/features")
p.add_argument("--videos-out", default="results/videos")
p.add_argument("--limit", type=int, default=16)
p.add_argument("--num-output-frames", type=int, default=21)
p.add_argument("--mode", choices=["i2v", "t2v-smoke"], default="i2v")
p.add_argument("--blocks", default="all", help="all or comma-separated block indices")
p.add_argument("--use-generator", action="store_true", help="use non-EMA generator instead of EMA")
a = p.parse_args()

manifest_path = Path(a.manifest).resolve()
base = manifest_path.parent
features_out = Path(a.features_out).resolve(); features_out.mkdir(parents=True, exist_ok=True)
videos_out = Path(a.videos_out).resolve(); videos_out.mkdir(parents=True, exist_ok=True)
items = read_manifest(manifest_path)[:a.limit]
selected = None if a.blocks == "all" else [int(x) for x in a.blocks.split(",")]

bundle = load_pipeline(a.model_root, a.config, a.checkpoint, use_ema=not a.use_generator)
print(f"Loaded {len(bundle.blocks)} transformer blocks")

for j, item in enumerate(items):
    print(f"[{j+1}/{len(items)}] {item['sample_id']} {item['state_type']}={item['state_value']}")
    initial_latent = None
    prompt = item["prompt"]
    if a.mode == "i2v":
        if getattr(bundle.pipeline, "num_frame_per_block", 1) != 1:
            raise RuntimeError(
                "Official Self-Forcing I2V requires a frame-wise checkpoint/config (num_frame_per_block=1). "
                "The released self_forcing_dmd.yaml is chunk-wise (3). Use an I2V/frame-wise checkpoint if available, "
                "or run --mode t2v-smoke only for plumbing validation."
            )
        initial_latent = load_initial_image_latent(bundle, base / item["image"])
    else:
        # Deliberately labelled smoke test: text carries the state and is therefore NOT a valid memory result.
        prompt = f"{item['state_value']} object. {item['prompt']}"

    counter = ModelCallCounter(bundle.model)
    recorder = PooledActivationRecorder(bundle.blocks, counter, selected_blocks=selected)
    with counter, recorder:
        video, _ = run(bundle, prompt, item["seed"], a.num_output_frames, initial_latent=initial_latent)

    calls = np.asarray([r["call"] for r in recorder.records], dtype=np.int32)
    blocks = np.asarray([r["block"] for r in recorder.records], dtype=np.int16)
    features = torch.cat([r["feature"] for r in recorder.records], dim=0).numpy().astype(np.float32)
    np.savez_compressed(
        features_out / f"{item['sample_id']}.npz",
        sample_id=item["sample_id"], pair_id=item["pair_id"], state_type=item["state_type"],
        state_value=item["state_value"], label=item["label"], calls=calls, blocks=blocks, features=features,
        confounded_text_smoke=(a.mode == "t2v-smoke"),
    )
    frames = (video[0].detach().cpu().permute(0,2,3,1).clamp(0,1) * 255).to(torch.uint8)
    write_video(str(videos_out / f"{item['sample_id']}.mp4"), frames, fps=16)
    if hasattr(bundle.pipeline.vae.model, "clear_cache"):
        bundle.pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()
