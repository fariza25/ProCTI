#!/usr/bin/env python3

import os
import time
import random
import argparse
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.iTransformer import Model as iTransformerModel


# --------------------------------------------------
# Repro
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_batch_size(requested, n):
    return max(1, min(int(requested), int(n)))


# --------------------------------------------------
# Shared Beijing data
# --------------------------------------------------

def load_shared_windows(shared_dir):

    train = np.load(os.path.join(shared_dir, "train_windows.npy")).astype(np.float32)
    val = np.load(os.path.join(shared_dir, "val_windows.npy")).astype(np.float32)
    test = np.load(os.path.join(shared_dir, "test_windows.npy")).astype(np.float32)

    return train, val, test


def load_shared_keepmask(mask_dir, split, ratio, seed):

    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")

    obj = np.load(path)
    key = f"{ratio:.2f}"

    evalmask = obj[key].astype(np.float32)

    return 1.0 - evalmask


# --------------------------------------------------
# Markov masking
# --------------------------------------------------

def markov_keep_mask(B, K, L, r_masked, lm, device):

    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1 - r_keep) / max(r_keep, 1e-12)

    out = np.ones((B, K, L), dtype=np.float32)

    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):

                out[b, k, t] = state

                if np.random.rand() < (p_m if state == 0 else p_u):
                    state = 1 - state

    return torch.from_numpy(out).to(device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class WindowDataset(Dataset):

    def __init__(self, windows):
        self.x = windows.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


# --------------------------------------------------
# iTransformer config
# --------------------------------------------------

def build_configs(args, K):

    return SimpleNamespace(
        task_name="imputation",
        seq_len=args.seq_len,
        pred_len=args.seq_len,
        enc_in=K,
        c_out=K,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        factor=args.factor,
        activation="gelu",
        embed="fixed",
        freq="h",
        output_attention=False
    )


def model_forward(model, x):
    out = model(x, None, None, None, None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_epoch(model, loader, opt, device, K, L, r_masked, lm):

    model.train()
    losses = []

    for xb in loader:

        xb = xb.to(device)

        keep = markov_keep_mask(xb.shape[0], K, L, r_masked, lm, device)
        keep = keep.permute(0, 2, 1)

        x_obs = xb * keep

        pred = model_forward(model, x_obs)

        evalmask = 1.0 - keep

        denom = evalmask.sum().clamp_min(1)

        loss = ((pred - xb).abs() * evalmask).sum() / denom

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    return np.mean(losses)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

@torch.no_grad()
def evaluate_split(model, loader, split, ratios, device, K, L, lm,
                   use_shared, shared_dir, seed):

    model.eval()

    rows = []

    for r in ratios:

        total_abs = 0
        total_sq = 0
        total_count = 0

        keep_NLK = None

        if use_shared:
            keep_NLK = load_shared_keepmask(shared_dir, split, r, seed)

        offset = 0

        for xb in loader:

            xb = xb.to(device)

            B = xb.shape[0]

            if keep_NLK is not None:
                keep = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                offset += B
            else:
                keep = markov_keep_mask(B, K, L, r, lm, device).permute(0,2,1)

            x_obs = xb * keep
            pred = model_forward(model, x_obs)

            evalmask = 1.0 - keep

            diff = (pred - xb) * evalmask

            total_abs += diff.abs().sum().item()
            total_sq += (diff**2).sum().item()
            total_count += evalmask.sum().item()

        mae = total_abs / total_count
        mse = total_sq / total_count
        rmse = np.sqrt(mse)

        rows.append((split, r, 1-r, mae, mse, rmse))

    return rows


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=96)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)

    ap.add_argument("--eval_masked_ratios", default="0.10,0.30,0.50,0.70")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", default=None)

    ap.add_argument("--out_txt", default="itransformer_beijing_metrics.txt")

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=1)

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    ratios = parse_ratios(args.eval_masked_ratios)

    train, val, test = load_shared_windows(args.shared_data_dir)

    K = train.shape[-1]
    L = args.seq_len

    train_loader = DataLoader(WindowDataset(train),
                              batch_size=safe_batch_size(args.batch,len(train)),
                              shuffle=True)

    val_loader = DataLoader(WindowDataset(val),
                            batch_size=safe_batch_size(args.batch,len(val)))

    test_loader = DataLoader(WindowDataset(test),
                             batch_size=safe_batch_size(args.batch,len(test)))

    configs = build_configs(args,K)

    model = iTransformerModel(configs).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):

        loss = train_epoch(model,train_loader,opt,device,K,L,args.r_train_masked,args.lm)

        print(f"[train] epoch={ep} loss={loss:.6f}")

    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)

    new_file = not os.path.exists(args.out_txt)

    with open(args.out_txt,"a") as f:

        if new_file:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split,loader in [("val",val_loader),("test",test_loader)]:

            rows = evaluate_split(
                model,loader,split,ratios,device,
                K,L,args.lm,args.use_shared_evalmask,
                args.shared_evalmask_dir,args.seed
            )

            for r in rows:

                f.write(f"{r[0]}\t{r[1]:.2f}\t{r[2]:.2f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\n")

                print(r)


if __name__ == "__main__":
    main()
