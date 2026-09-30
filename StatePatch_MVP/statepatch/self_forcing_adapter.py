from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import os
import sys
import numpy as np
import torch
from PIL import Image


@dataclass
class SelfForcingBundle:
    pipeline: object
    device: torch.device
    root: Path

    @property
    def model(self):
        return self.pipeline.generator.model

    @property
    def blocks(self):
        return self.model.blocks


def load_pipeline(
    self_forcing_root: str | Path,
    config_path: str,
    checkpoint_path: str,
    use_ema: bool = True,
    device: str = "cuda",
) -> SelfForcingBundle:
    """Load the official Self-Forcing inference pipeline.

    This intentionally mirrors the official inference.py loading logic.
    Paths may be absolute or relative to the Self-Forcing repository root.
    """
    root = Path(self_forcing_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Official code uses relative paths for Wan weights, so run from repo root.
    os.chdir(root)

    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    config = OmegaConf.load(str(cfg_path))
    default_config = OmegaConf.load(str(root / "configs/default_config.yaml"))
    config = OmegaConf.merge(default_config, config)
    dev = torch.device(device)
    if hasattr(config, "denoising_step_list"):
        pipe = CausalInferencePipeline(config, device=dev)
    else:
        pipe = CausalDiffusionInferencePipeline(config, device=dev)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    key = "generator_ema" if use_ema and "generator_ema" in state_dict else "generator"
    gen_sd = state_dict[key]
    try:
        pipe.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipe.generator.load_state_dict(fixed, strict=False)

    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(device=dev)
    pipe.generator.to(device=dev)
    pipe.vae.to(device=dev)
    pipe.eval()
    torch.set_grad_enabled(False)
    return SelfForcingBundle(pipe, dev, root)


def load_initial_image_latent(bundle: SelfForcingBundle, image_path: str | Path):
    """Encode a 480x832 image to the first latent frame with the official Wan VAE."""
    im = Image.open(image_path).convert("RGB").resize((832, 480), Image.Resampling.BICUBIC)
    x = np.asarray(im).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x * 2.0 - 1.0
    x = x.unsqueeze(0).unsqueeze(2).to(bundle.device, dtype=torch.bfloat16)  # B,C,F,H,W
    latent = bundle.pipeline.vae.encode_to_latent(x).to(bundle.device, dtype=torch.bfloat16)
    return latent


def make_noise(bundle: SelfForcingBundle, seed: int, num_output_frames: int, initial_latent=None):
    g = torch.Generator(device=bundle.device).manual_seed(int(seed))
    # Official I2V inference uses num_output_frames - 1 generated latent frames for a one-frame initial latent.
    n = num_output_frames - (1 if initial_latent is not None else 0)
    return torch.randn((1, n, 16, 60, 104), generator=g, device=bundle.device, dtype=torch.bfloat16)


def run(bundle: SelfForcingBundle, prompt: str, seed: int, num_output_frames: int = 21, initial_latent=None):
    noise = make_noise(bundle, seed, num_output_frames, initial_latent=initial_latent)
    with torch.inference_mode():
        video, latents = bundle.pipeline.inference(
            noise=noise,
            text_prompts=[prompt],
            initial_latent=initial_latent,
            return_latents=True,
        )
    return video, latents
