from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import torch


def _as_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported module output type: {type(output)}")


def _replace_tensor(output, new_tensor):
    if torch.is_tensor(output):
        return new_tensor
    if isinstance(output, tuple):
        return (new_tensor, *output[1:])
    if isinstance(output, list):
        return [new_tensor, *output[1:]]
    raise TypeError(f"Unsupported module output type: {type(output)}")


@dataclass
class ModelCallCounter:
    """Counts top-level model forward calls so block activations can be indexed reproducibly."""
    model: torch.nn.Module
    call_index: int = -1
    _handle: Optional[torch.utils.hooks.RemovableHandle] = field(default=None, init=False)

    def __enter__(self):
        def _pre_hook(_module, _args, _kwargs):
            self.call_index += 1
        self._handle = self.model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()


class PooledActivationRecorder:
    """Records token-pooled transformer block outputs on CPU.

    Records one vector per (model_call, block). This is intentionally cheap enough
    for full layer/time sweeps without saving enormous token-level tensors.
    """
    def __init__(
        self,
        blocks: Sequence[torch.nn.Module],
        call_counter: ModelCallCounter,
        selected_blocks: Optional[Sequence[int]] = None,
        pooling: str = "mean",
        dtype: torch.dtype = torch.float32,
    ):
        self.blocks = list(blocks)
        self.call_counter = call_counter
        self.selected_blocks = set(selected_blocks if selected_blocks is not None else range(len(self.blocks)))
        self.pooling = pooling
        self.dtype = dtype
        self.records: List[Dict] = []
        self._handles = []

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            return x
        # Wan block output is [B, L, C]. Preserve batch dimension and pool tokens.
        if self.pooling == "mean":
            return x.mean(dim=-2)
        if self.pooling == "last":
            return x[..., -1, :]
        if self.pooling == "first":
            return x[..., 0, :]
        raise ValueError(f"Unknown pooling={self.pooling}")

    def __enter__(self):
        for block_idx, block in enumerate(self.blocks):
            if block_idx not in self.selected_blocks:
                continue
            def make_hook(idx):
                def _hook(_module, _args, output):
                    x = _as_tensor(output).detach()
                    pooled = self._pool(x).to(device="cpu", dtype=self.dtype)
                    self.records.append({
                        "call": int(self.call_counter.call_index),
                        "block": int(idx),
                        "feature": pooled,
                    })
                return _hook
            self._handles.append(block.register_forward_hook(make_hook(block_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class SingleActivationCapture:
    """Captures one full block activation for one top-level model call."""
    def __init__(self, block: torch.nn.Module, call_counter: ModelCallCounter, target_call: int):
        self.block = block
        self.call_counter = call_counter
        self.target_call = target_call
        self.tensor: Optional[torch.Tensor] = None
        self._handle = None

    def __enter__(self):
        def _hook(_module, _args, output):
            if self.call_counter.call_index == self.target_call:
                self.tensor = _as_tensor(output).detach().cpu()
        self._handle = self.block.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()


class ActivationPatcher:
    """Patches one block output at one model call.

    mode='full' replaces the complete activation.
    mode='delta' applies target + alpha*(donor-target).
    mode='subspace' applies only a low-rank projector U, where U is [C, r].
    """
    def __init__(
        self,
        block: torch.nn.Module,
        call_counter: ModelCallCounter,
        target_call: int,
        donor: torch.Tensor,
        alpha: float = 1.0,
        mode: str = "delta",
        subspace: Optional[torch.Tensor] = None,
    ):
        self.block = block
        self.call_counter = call_counter
        self.target_call = target_call
        self.donor = donor
        self.alpha = float(alpha)
        self.mode = mode
        self.subspace = subspace
        self._handle = None
        self.num_applied = 0

    def __enter__(self):
        def _hook(_module, _args, output):
            if self.call_counter.call_index != self.target_call:
                return output
            x = _as_tensor(output)
            donor = self.donor.to(device=x.device, dtype=x.dtype)
            if donor.shape != x.shape:
                raise ValueError(f"Donor shape {tuple(donor.shape)} != target shape {tuple(x.shape)}")
            if self.mode == "full":
                patched = donor
            elif self.mode == "delta":
                patched = x + self.alpha * (donor - x)
            elif self.mode == "subspace":
                if self.subspace is None:
                    raise ValueError("subspace mode requires subspace U [C, r]")
                U = self.subspace.to(device=x.device, dtype=x.dtype)
                if U.ndim == 1:
                    U = U[:, None]
                # Projection of donor-target onto column space(U).
                delta = donor - x
                proj = torch.matmul(torch.matmul(delta, U), U.transpose(-2, -1))
                patched = x + self.alpha * proj
            else:
                raise ValueError(f"Unknown patch mode={self.mode}")
            self.num_applied += 1
            return _replace_tensor(output, patched)
        self._handle = self.block.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
