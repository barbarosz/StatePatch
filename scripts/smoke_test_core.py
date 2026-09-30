#!/usr/bin/env python
"""CPU-only proof that hook/capture/patch/probe plumbing works before burning GPU time."""
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from statepatch.hooks import ModelCallCounter, PooledActivationRecorder, SingleActivationCapture, ActivationPatcher
from statepatch.probes import fit_binary_probe

class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=nn.ModuleList([nn.Linear(8,8,bias=False) for _ in range(4)])
        for b in self.blocks: nn.init.eye_(b.weight)
    def forward(self,x):
        for b in self.blocks: x=torch.tanh(b(x))
        return x

m=Toy(); x0=torch.randn(1,5,8); x1=x0.clone(); x1[...,0]+=2
counter=ModelCallCounter(m); rec=PooledActivationRecorder(m.blocks,counter)
with counter,rec: m(x0); m(x1)
assert len(rec.records)==8
counter=ModelCallCounter(m); cap=SingleActivationCapture(m.blocks[2],counter,0)
with counter,cap: m(x1)
assert cap.tensor is not None
counter=ModelCallCounter(m); patch=ActivationPatcher(m.blocks[2],counter,0,cap.tensor,mode='full')
with counter,patch: y=m(x0)
assert patch.num_applied==1
X=np.r_[np.random.randn(30,8)-1,np.random.randn(30,8)+1]; yy=np.r_[np.zeros(30),np.ones(30)]
_,metrics=fit_binary_probe(X,yy,cv=3)
assert metrics['balanced_accuracy']>.8
print('StatePatch core smoke test: PASS',metrics)
