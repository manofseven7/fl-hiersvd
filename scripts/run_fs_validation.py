"""Training-only HAR retention sweep used to select rho (seeds 42--44).

The official test set is never loaded by this script. Ten percent of the
official training partition is held out with stratification and used only for
configuration selection.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from configs.presets import preset
from flcore.data import load_har, _dirichlet_split
from flcore.engine import train
from flcore.utils import get_logger, save_json, set_seed

SEEDS=(42,43,44); RHOS=(0.50,0.70,0.90); OUT="results/fs_validation"
os.makedirs(OUT,exist_ok=True); log=get_logger("fs-validation")
(X,y), _, d, classes=load_har()

for seed in SEEDS:
    tr_idx,val_idx=train_test_split(np.arange(len(y)),test_size=.10,random_state=seed,
                                    stratify=y.numpy())
    Xtr,ytr=X[tr_idx],y[tr_idx]; Xval,yval=X[val_idx],y[val_idx]
    parts=_dirichlet_split(ytr.numpy(),50,.1,np.random.default_rng(seed))
    def client_loader(i):
        ds=TensorDataset(Xtr[parts[i]],ytr[parts[i]])
        gen=torch.Generator().manual_seed(seed*1000+i)
        return DataLoader(ds,batch_size=64,shuffle=True,generator=gen),len(ds)
    val_loader=DataLoader(TensorDataset(Xval,yval),batch_size=256,shuffle=False)
    for rho in RHOS:
        cfg=preset("har",fs=True); cfg.method="proposed"; cfg.seed=seed; cfg.rho=rho
        set_seed(seed)
        traj,comm=train(cfg,client_loader,val_loader,d,classes,False,parts,log)
        save_json(f"{OUT}/har_proposed_rho{rho:.2f}_s{seed}.json",
                  dict(cfg=cfg.__dict__,split="training-only stratified 90/10",
                       traj=traj,final_acc=traj[-1]["acc"],final_f1=traj[-1]["f1"],
                       comm_mb=comm.mb()))

print("rho mean_validation_accuracy mean_validation_macro_f1")
for rho in RHOS:
    recs=[json.load(open(f"{OUT}/har_proposed_rho{rho:.2f}_s{s}.json")) for s in SEEDS]
    print(f"{rho:.2f}",f"{np.mean([r['final_acc'] for r in recs]):.3f}",
          f"{np.mean([r['final_f1'] for r in recs]):.3f}")
