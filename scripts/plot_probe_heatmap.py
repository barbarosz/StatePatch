#!/usr/bin/env python
import argparse, csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('--csv',default='results/probe_sweep.csv'); p.add_argument('--out',default='results/probe_heatmap.png'); p.add_argument('--metric',default='balanced_accuracy'); p.add_argument('--state-type',default='color'); a=p.parse_args()
rows=[r for r in csv.DictReader(open(a.csv)) if r.get('state_type') == a.state_type]
if not rows: raise SystemExit(f'No rows for state_type={a.state_type}')
calls=sorted({int(r['call']) for r in rows}); blocks=sorted({int(r['block']) for r in rows})
M=np.full((len(blocks),len(calls)),np.nan)
ci={c:i for i,c in enumerate(calls)}; bi={b:i for i,b in enumerate(blocks)}
for r in rows: M[bi[int(r['block'])],ci[int(r['call'])]]=float(r[a.metric])
fig,ax=plt.subplots(figsize=(max(8,len(calls)*.35),8)); im=ax.imshow(M,aspect='auto',origin='lower',vmin=.5,vmax=1.0)
ax.set_xlabel('Top-level generator forward call'); ax.set_ylabel('Transformer block'); ax.set_xticks(range(len(calls))); ax.set_xticklabels(calls,rotation=90,fontsize=7); ax.set_yticks(range(len(blocks))); ax.set_yticklabels(blocks,fontsize=7)
fig.colorbar(im,ax=ax,label=a.metric); fig.tight_layout(); Path(a.out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.out,dpi=180); print(a.out)
