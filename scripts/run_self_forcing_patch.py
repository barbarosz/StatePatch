#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision.io import write_video
from statepatch.benchmark import read_manifest
from statepatch.hooks import ModelCallCounter, SingleActivationCapture, ActivationPatcher
from statepatch.self_forcing_adapter import load_pipeline, load_initial_image_latent, run

p=argparse.ArgumentParser(description='Matched counterfactual activation patching on a StatePatchBench pair')
p.add_argument('--model-root','--self-forcing-root',dest='model_root',required=True); p.add_argument('--config',default='configs/self_forcing_dmd.yaml'); p.add_argument('--checkpoint',default='checkpoints/self_forcing_dmd.pt')
p.add_argument('--manifest',default='data/statepatchbench.jsonl'); p.add_argument('--pair-id',required=True); p.add_argument('--block',type=int,required=True); p.add_argument('--call',type=int,required=True); p.add_argument('--alpha',type=float,default=1.0); p.add_argument('--mode',choices=['full','delta'],default='delta'); p.add_argument('--out',default='results/patch'); p.add_argument('--num-output-frames',type=int,default=21)
a=p.parse_args()
manifest=Path(a.manifest).resolve(); base=manifest.parent; out=Path(a.out).resolve(); rows=[x for x in read_manifest(manifest) if x['pair_id']==a.pair_id]
if len(rows)!=2: raise SystemExit(f'Expected exactly 2 rows for pair {a.pair_id}, got {len(rows)}')
rows=sorted(rows,key=lambda x:x['label']); donor_item,target_item=rows
bundle=load_pipeline(a.model_root,a.config,a.checkpoint,use_ema=True)
if getattr(bundle.pipeline,'num_frame_per_block',1)!=1: raise RuntimeError('Causal I2V patch experiment requires frame-wise/I2V config/checkpoint (num_frame_per_block=1).')
out.mkdir(parents=True,exist_ok=True)

def save(video,name):
    frames=(video[0].detach().cpu().permute(0,2,3,1).clamp(0,1)*255).to(torch.uint8); write_video(str(out/name),frames,fps=16)

donor_lat=load_initial_image_latent(bundle,base/donor_item['image'])
counter=ModelCallCounter(bundle.model); cap=SingleActivationCapture(bundle.blocks[a.block],counter,a.call)
with counter,cap: donor_video,_=run(bundle,donor_item['prompt'],donor_item['seed'],a.num_output_frames,initial_latent=donor_lat)
if cap.tensor is None: raise RuntimeError('Target call/block was never reached in donor run. Run capture first and inspect call indices.')
torch.save(cap.tensor,out/'donor_activation.pt'); save(donor_video,'donor.mp4')

target_lat=load_initial_image_latent(bundle,base/target_item['image'])
target_video,_=run(bundle,target_item['prompt'],target_item['seed'],a.num_output_frames,initial_latent=target_lat); save(target_video,'target_unpatched.mp4')

# Reload pipeline to reset every cache/model state before intervention.
bundle=load_pipeline(a.model_root,a.config,a.checkpoint,use_ema=True)
target_lat=load_initial_image_latent(bundle,base/target_item['image'])
counter=ModelCallCounter(bundle.model); patch=ActivationPatcher(bundle.blocks[a.block],counter,a.call,cap.tensor,alpha=a.alpha,mode=a.mode)
with counter,patch: patched_video,_=run(bundle,target_item['prompt'],target_item['seed'],a.num_output_frames,initial_latent=target_lat)
if patch.num_applied != 1: raise RuntimeError(f'Expected patch once, applied {patch.num_applied} times')
save(patched_video,'target_patched.mp4')
print(f'Wrote donor/unpatched/patched videos to {out}')
