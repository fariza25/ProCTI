#!/usr/bin/env python3

import os, argparse, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from engine.solver import Trainer
from engine.logger import Logger


# ============================================================
# Repro
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):
    def __init__(self, arr):
        self.x = arr.astype(np.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


def load_shared_windows(shared_dir):

    train = np.load(os.path.join(shared_dir,"train_windows.npy"))
    val   = np.load(os.path.join(shared_dir,"val_windows.npy"))
    test  = np.load(os.path.join(shared_dir,"test_windows.npy"))

    return train.astype(np.float32),val.astype(np.float32),test.astype(np.float32)


def load_maskbank(mask_dir,split,r,seed):

    path=os.path.join(mask_dir,f"{split}_maskbank_seed{seed}.npz")
    obj=np.load(path)

    key=f"{r:.2f}"

    if key not in obj:
        raise ValueError(f"{key} missing from {path}")

    evalmask=obj[key]   # 1 = masked

    keep=1.0-evalmask

    return keep.astype(np.float32)


# ============================================================
# Markov fallback
# ============================================================

def markov_mask(B,K,L,r_masked,lm,device):

    r_keep=1-r_masked
    p_m=1/lm
    p_u=p_m*(1-r_keep)/max(r_keep,1e-12)

    out=np.ones((B,K,L),dtype=np.float32)

    for b in range(B):
        for k in range(K):

            state=int(np.random.rand()<r_keep)

            for t in range(L):

                out[b,k,t]=state

                if np.random.rand()<(p_m if state==0 else p_u):
                    state=1-state

    return torch.from_numpy(out).to(device)


# ============================================================
# Main
# ============================================================

def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir",required=True)
    ap.add_argument("--seq_len",type=int,default=96)

    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--lr",type=float,default=1e-5)

    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--lm",type=float,default=6.0)

    ap.add_argument("--use_shared_evalmask",action="store_true")
    ap.add_argument("--shared_evalmask_dir",type=str,default="")

    ap.add_argument("--name",default="diffusionts_beijing")
    ap.add_argument("--output",default="OUTPUT")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--log_frequency", type=int, default=100)
    #ap.add_argument("--milestone", type=int, default=0)
    ap.add_argument("--milestone", type=int, default=0)

    ap.add_argument("--out_txt",default="diffusionts_beijing_metrics.txt")

    args=ap.parse_args()

    set_seed(args.seed)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir=args.shared_data_dir

    args.save_dir = os.path.join(args.output, args.name)
    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================
    # Load Beijing windows
    # ============================================================

    train_w,val_w,test_w=load_shared_windows(args.shared_data_dir)

    K=train_w.shape[-1]
    L=args.seq_len

    train_loader=DataLoader(WindowDataset(train_w),batch_size=args.batch,shuffle=True,drop_last=True)
    val_loader=DataLoader(WindowDataset(val_w),batch_size=args.batch)
    test_loader=DataLoader(WindowDataset(test_w),batch_size=args.batch)


    # ============================================================
    # Diffusion-TS Model
    # ============================================================

    model=Diffusion_TS(
        seq_length=L,
        feature_size=K,
        n_layer_enc=4,
        n_layer_dec=4,
        d_model=96,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1',
        beta_schedule='cosine',
        n_heads=4,
        use_ff=True,
        eta=0.0
    ).to(device)

    config = {
        "solver": {
            "base_lr": args.lr,
            "max_epochs": args.steps,
            "gradient_accumulate_every": 2,
            "save_cycle": 2000,
            "results_folder": os.path.join(args.save_dir, "checkpoints"),
            "ema": {"decay": 0.995, "update_interval": 10},
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5,
                    "patience": 1000,
                    "min_lr": args.lr,
                    "threshold": 1e-1,
                    "threshold_mode": "rel",
                    "warmup_lr": 8e-4,
                    "warmup": 500,
                    "verbose": False
                }
            }
        }
    }
    args.save_dir = os.path.join(args.output, args.name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger=Logger(args)

    trainer=Trainer(config,args,model,{"dataloader":train_loader,"dataset":None},logger)

    print("Training Diffusion-TS on Beijing...")
    trainer.train()

    model=trainer.ema.ema_model
    model.eval()


    # ============================================================
    # Evaluation
    # ============================================================

    ratios=[0.10,0.30,0.50,0.70]
    rows=[]

    for split_name,loader in [("val",val_loader),("test",test_loader)]:

        for r in ratios:

            total_abs=0
            total_sq=0
            total_den=0

            total_robs=0

            for i,xb in enumerate(loader):

                xb=xb.to(device)

                B=xb.shape[0]

                if args.use_shared_evalmask:

                    keep_nlk=load_maskbank(args.shared_evalmask_dir,split_name,r,args.seed)

                    start=i*args.batch
                    keep_BLK=torch.from_numpy(
                        keep_nlk[start:start+B]
                    ).to(device)

                else:

                    keep_BKL=markov_mask(B,K,L,r,args.lm,device)

                    keep_BLK=keep_BKL.permute(0,2,1)

                sample=model.fast_sample_infill(
                    shape=xb.shape,
                    target=xb*keep_BLK,
                    partial_mask=keep_BLK.bool(),
                    model_kwargs={"coef":1e-2,"learning_rate":5e-2},
                    sampling_timesteps=250
                )

                evalmask=1.0-keep_BLK

                diff=(sample-xb)*evalmask

                den=float(evalmask.sum().item())

                if den<1:
                    continue

                total_abs+=diff.abs().sum().item()
                total_sq+=(diff**2).sum().item()
                total_den+=den

                total_robs+=keep_BLK.mean().item()*den

            total_den=max(total_den,1)

            mae=total_abs/total_den
            mse=total_sq/total_den
            rmse=np.sqrt(mse)

            r_obs=total_robs/total_den

            print(f"[{split_name}] r={r:.2f}  MAE={mae:.6f} RMSE={rmse:.6f}")

            rows.append((split_name,r,r_obs,mae,mse,rmse))


    # ============================================================
    # Write metrics
    # ============================================================

    write_header=not os.path.exists(args.out_txt)

    with open(args.out_txt,"a") as f:

        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split,r,robs,mae,mse,rmse in rows:
            f.write(f"{split}\t{r:.2f}\t{robs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")


    print("Done.")


if __name__=="__main__":
    main()
