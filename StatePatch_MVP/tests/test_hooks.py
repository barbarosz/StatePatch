import torch
from torch import nn
from statepatch.hooks import ModelCallCounter, PooledActivationRecorder, SingleActivationCapture, ActivationPatcher

class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=nn.ModuleList([nn.Identity(),nn.Identity()])
    def forward(self,x):
        for b in self.blocks:x=b(x)
        return x

def test_recorder_and_patch():
    m=Toy(); x=torch.randn(1,3,4); donor=x+3
    c=ModelCallCounter(m); r=PooledActivationRecorder(m.blocks,c)
    with c,r:m(x)
    assert len(r.records)==2 and r.records[0]['feature'].shape==(1,4)
    c=ModelCallCounter(m); cap=SingleActivationCapture(m.blocks[0],c,0)
    with c,cap:m(donor)
    c=ModelCallCounter(m); p=ActivationPatcher(m.blocks[0],c,0,cap.tensor,mode='full')
    with c,p:y=m(x)
    assert torch.allclose(y,donor) and p.num_applied==1
