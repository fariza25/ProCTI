#!/usr/bin/env python3

import os
import json
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.SCINet import SCINet


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -----------------------------
# Shared Beijing loading
# -----------------------------
def load_shared_windows(shared_data_dir: str):
    train_path = os.path.join(shared_data_dir, "train_windows.npy")
    val_path = os.path.join(shared_data_dir, "val_windows.npy")
    test_path = os.path.join(shared_data_dir, "test_windows.npy")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing shared window file: {p}")

    train_w = np.load(train_path).astype(np.float32)
    val_w = np.load(val_path).astype(np.float32)
    test_w = np.load(test_path).astype(np.float32)

    if train_w.ndim != 3 or val_w.ndim != 3 or test_w.ndim != 3:
        raise ValueError(
            f"Expected windows of shape (N,L,K), got "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    if train_w.shape[1:] != val_w.shape[1:] or train_w.shape[1:] != test_w.shape[1:]:
        raise ValueError(
            f"Window shape mismatch: "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    return train_w, val_w, test_w


def load_shared_keepmask_nlk(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    """
    Load shared Beijing maskbank and convert:
      evalmask (1=masked, 0=observed)
    to:
      keepmask (1=observed, 0=masked)
    """
    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared maskbank not found: {path}")

    obj = np.load(path)
    key = f"{r_m:.2f}"
    if key not in obj:
        raise KeyError(f"Ratio key {key} not found in {path}. Available keys: {obj.files}")

    evalmask = obj[key].astype(np.float32)   # (N,L,K), 1=masked
    keepmask = 1.0 - evalmask
    return keepmask


# -----------------------------
# Markov keep-mask
# keepmask: 1=observed/kept, 0=masked
# -----------------------------
def markov_keep_mask_from_masked_ratio(
    B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device
) -> torch.Tensor:
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < p[state]:
                    state = 1 - state
    return torch.from_numpy(out).to(device=device)


# -----------------------------
# Dataset
# -----------------------------
class WindowDatasetWithIndex(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(idx)  # (L,K), index


# -----------------------------
# SCINet wrapper
# -----------------------------
class SCINetWrapper(nn.Module):
    def __init__(self, K: int, seq_len: int, num_levels: int = 3, num_stacks: int = 1):
        super().__init__()
        self.model = SCINet(
            output_len=seq_len,
            input_len=seq_len,
            input_dim=K,
            hid_size=1,
            num_stacks=num_stacks,
            num_levels=num_levels,
        )

    def forward(self, x_BLK: torch.Tensor) -> torch.Tensor:
        return self.model(x_BLK)


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, opt, device, lm: float, r_train_masked: float):
    model.train()
    total = 0.0

    for x, _idx in loader:
        x = x.to(device)  # (B,L,K)
        B, L, K = x.shape

        keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_train_masked, lm, device)  # (B,K,L)
        keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()  # (B,L,K)

        x_masked = x * keep_BLK
        pred = model(x_masked)

        evalmask = 1.0 - keep_BLK
        denom = evalmask.sum().clamp_min(1.0)
        loss = ((pred - x) ** 2 * evalmask).sum() / denom

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def eval_with_shared_or_markov(model, loader, device, lm: float, r_masked: float,
                               split_name: str, seed: int,
                               use_shared_evalmask: bool, shared_evalmask_dir: str):
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    denom = 0.0
    total_robs_weighted = 0.0

    keep_nlk = None
    if use_shared_evalmask:
        keep_nlk = load_shared_keepmask_nlk(shared_evalmask_dir, split_name, r_masked, seed)
        if keep_nlk.shape[0] != len(loader.dataset):
            raise ValueError(
                f"Mask/window count mismatch for split={split_name}, r={r_masked:.2f}: "
                f"mask has N={keep_nlk.shape[0]}, dataset has N={len(loader.dataset)}"
            )

    offset = 0
    for x, _idx in loader:
        x = x.to(device)
        B, L, K = x.shape

        if keep_nlk is not None:
            keep_BLK = torch.from_numpy(keep_nlk[offset:offset+B]).to(device).float()  # (B,L,K)
            offset += B
        else:
            keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_masked, lm, device)
            keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

        pred = model(x * keep_BLK)
        evalmask = 1.0 - keep_BLK
        diff = (pred - x) * evalmask

        batch_den = evalmask.sum().item()
        sum_abs += diff.abs().sum().item()
        sum_sq += (diff ** 2).sum().item()
        denom += batch_den
        total_robs_weighted += float(keep_BLK.mean().item()) * batch_den

    denom = max(1.0, denom)
    mae = sum_abs / denom
    mse = sum_sq / denom
    rmse = float(np.sqrt(mse))
    r_obs = float(total_robs_weighted / denom)
    return mae, mse, rmse, r_obs


@torch.no_grad()
def collect_test_arrays(model, loader, device, lm: float, r_masked: float, seed: int,
                        use_shared_evalmask: bool, shared_evalmask_dir: str):
    model.eval()
    gt_all, imp_all, cond_all, eval_all = [], [], [], []

    keep_nlk = None
    if use_shared_evalmask:
        keep_nlk = load_shared_keepmask_nlk(shared_evalmask_dir, "test", r_masked, seed)
        if keep_nlk.shape[0] != len(loader.dataset):
            raise ValueError(
                f"Mask/window count mismatch for split=test, r={r_masked:.2f}: "
                f"mask has N={keep_nlk.shape[0]}, dataset has N={len(loader.dataset)}"
            )

    offset = 0
    for x, _idx in loader:
        x = x.to(device)
        B, L, K = x.shape

        if keep_nlk is not None:
            keep_BLK = torch.from_numpy(keep_nlk[offset:offset+B]).to(device).float()
            offset += B
        else:
            keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_masked, lm, device)
            keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

        pred = model(x * keep_BLK)
        evalmask = 1.0 - keep_BLK
        imputed_full = keep_BLK * x + evalmask * pred

        gt_all.append(x.detach().cpu().numpy())
        imp_all.append(imputed_full.detach().cpu().numpy())
        cond_all.append(keep_BLK.detach().cpu().numpy())
        eval_all.append(evalmask.detach().cpu().numpy())

    return (
        np.concatenate(gt_all, axis=0),
        np.concatenate(imp_all, axis=0),
        np.concatenate(cond_all, axis=0),
        np.concatenate(eval_all, axis=0),
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_scinet_beijing")

    ap.add_argument("--out_txt", type=str, default="scinet_beijing_metrics.txt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    # -----------------------------
    # Data
    # -----------------------------
    train, val, test = load_shared_windows(args.shared_data_dir)

    print(f"[Beijing] windows train/val/test = {len(train)}/{len(val)}/{len(test)}  L={args.seq_len}  K={train.shape[-1]}")

    meta_path = os.path.join(args.shared_data_dir, "split_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "train_stations" in meta and "val_stations" in meta and "test_stations" in meta:
                print(f"[split] train_stations={meta['train_stations']}")
                print(f"[split] val_stations={meta['val_stations']}")
                print(f"[split] test_stations={meta['test_stations']}")
        except Exception as e:
            print(f"[warn] Could not read split_meta.json: {e}")

    train_loader = DataLoader(
        WindowDatasetWithIndex(train),
        batch_size=safe_batch_size(args.batch, len(train)),
        shuffle=True,
        drop_last=(len(train) > 1),
    )
    val_loader = DataLoader(
        WindowDatasetWithIndex(val),
        batch_size=safe_batch_size(args.batch, len(val)),
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowDatasetWithIndex(test),
        batch_size=safe_batch_size(args.batch, len(test)),
        shuffle=False,
    )

    # -----------------------------
    # Model
    # -----------------------------
    K = int(train.shape[-1])
    if train.shape[1] != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={train.shape[1]}, args.seq_len={args.seq_len}")

    model = SCINetWrapper(K, args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # Train
    # -----------------------------
    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, device, args.lm, args.r_train_masked)
        print(f"[train] epoch={ep:03d} loss={loss:.6f}")

    # -----------------------------
    # Eval + log
    # -----------------------------
    ratios = parse_ratios(args.eval_masked_ratios)

    file_exists = os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if not file_exists:
            f.write("seed\tsplit\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split, loader in [("val", val_loader), ("test", test_loader)]:
            for r in ratios:
                mae, mse, rmse, r_obs = eval_with_shared_or_markov(
                    model, loader, device, args.lm, r, split, args.seed,
                    args.use_shared_evalmask, args.shared_evalmask_dir
                )
                f.write(f"{args.seed}\t{split}\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{split}] r_masked={r:.2f} r_obs={r_obs:.2f} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")

                if args.save_test_arrays and split == "test":
                    os.makedirs(args.save_dir, exist_ok=True)
                    gt, imp, cond, evalm = collect_test_arrays(
                        model, loader, device, args.lm, r, args.seed,
                        args.use_shared_evalmask, args.shared_evalmask_dir
                    )
                    tag = f"scinet_beijing_test_r{r:.2f}_seed{args.seed}_L{args.seq_len}"
                    np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt)
                    np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp)
                    np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond)
                    np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), evalm)

    if args.save_test_arrays:
        meta_out = {
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_data_dir": args.shared_data_dir,
            "r_train_masked": float(args.r_train_masked),
            "lm": float(args.lm),
        }
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, f"metadata_seed{args.seed}.json"), "w") as f:
            json.dump(meta_out, f, indent=2)


if __name__ == "__main__":
    main()
