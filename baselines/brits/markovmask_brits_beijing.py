#!/usr/bin/env python3

import os
import json
import time
import math
import random
import argparse
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilities
# -----------------------------
def safe_batch_size(requested: int, n: int) -> int:
    if n <= 0:
        return 1
    return max(1, min(int(requested), int(n)))


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
            f"Expected all windows to have shape (N,L,K), got "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    if train_w.shape[1:] != val_w.shape[1:] or train_w.shape[1:] != test_w.shape[1:]:
        raise ValueError(
            f"Shared window shape mismatch: "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    return train_w, val_w, test_w


def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    """
    Loads:
      {split}_maskbank_seed{seed}.npz
    where each key is "0.10", "0.30", etc and stores evalmask:
      1 = masked
      0 = observed

    Converts to keepmask:
      keepmask = 1 - evalmask
    """
    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared maskbank not found: {path}")

    obj = np.load(path)
    key = f"{r_m:.2f}"
    if key not in obj:
        raise KeyError(f"Ratio key {key} not found in {path}. Available keys: {obj.files}")

    evalmask = obj[key].astype(np.float32)
    keepmask = 1.0 - evalmask
    return keepmask


def pointwise_keep_mask(shape, masked_ratio: float, rng: np.random.RandomState):
    keep_prob = 1.0 - float(masked_ratio)
    return (rng.rand(*shape) < keep_prob).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class BeijingWindowDataset(Dataset):
    """
    Returns raw window tensor only.
    Masking is done in training/evaluation loops.
    """
    def __init__(self, windows_nlk: np.ndarray):
        self.x = windows_nlk.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


# -----------------------------
# BRITS-style model
# -----------------------------
class FeatureRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size)
        self.register_buffer("mask", m)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.matmul(x, self.weight * self.mask) + self.bias


class TemporalDecay(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, d):
        gamma = torch.relu(torch.matmul(d, self.weight.t()) + self.bias)
        return torch.exp(-gamma)


class BRITSBeijing(nn.Module):
    """
    A compact BRITS-style bidirectional recurrent imputer.

    Input convention:
      x:         (B,L,K) raw full window
      keepmask:  (B,L,K) 1=observed/kept, 0=masked

    Returns:
      imputed_full: (B,L,K)
      train_loss: scalar
    """
    def __init__(self, k: int, hidden_size: int):
        super().__init__()
        self.k = k
        self.hidden_size = hidden_size

        self.rnn_f = nn.LSTMCell(input_size=2 * k, hidden_size=hidden_size)
        self.rnn_b = nn.LSTMCell(input_size=2 * k, hidden_size=hidden_size)

        self.temp_decay_h = TemporalDecay(k, hidden_size)
        self.temp_decay_x = TemporalDecay(k, k)

        self.hist_reg = nn.Linear(hidden_size, k)
        self.feat_reg = FeatureRegression(k)
        self.weight_combine = nn.Linear(2 * k, k)
        self.out_proj = nn.Linear(2 * hidden_size, k)

    def _compute_deltas(self, keepmask: torch.Tensor) -> torch.Tensor:
        """
        keepmask: (B,L,K), 1=observed
        returns deltas: (B,L,K)
        """
        B, L, K = keepmask.shape
        deltas = torch.zeros_like(keepmask)
        for t in range(1, L):
            deltas[:, t, :] = 1.0 + (1.0 - keepmask[:, t - 1, :]) * deltas[:, t - 1, :]
        return deltas

    def _run_direction(self, x: torch.Tensor, keepmask: torch.Tensor, reverse: bool = False):
        B, L, K = x.shape
        deltas = self._compute_deltas(keepmask)
        if reverse:
            x = torch.flip(x, dims=[1])
            keepmask = torch.flip(keepmask, dims=[1])
            deltas = torch.flip(deltas, dims=[1])

        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)

        preds = []
        total_loss = 0.0
        total_denom = 0.0

        x_prev = torch.zeros(B, K, device=x.device)

        for t in range(L):
            x_t = x[:, t, :]
            m_t = keepmask[:, t, :]
            d_t = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d_t)
            gamma_x = self.temp_decay_x(d_t)
            h = h * gamma_h

            x_hist = self.hist_reg(h)
            z_hist = m_t * x_t + (1.0 - m_t) * x_hist

            x_feat = self.feat_reg(z_hist)
            alpha = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m_t], dim=-1)))
            x_c = alpha * x_feat + (1.0 - alpha) * x_hist

            x_input = m_t * x_t + (1.0 - m_t) * x_c

            total_loss = total_loss + (((x_t - x_hist) * m_t).abs().sum()
                                       + ((x_t - x_feat) * m_t).abs().sum()
                                       + ((x_t - x_c) * m_t).abs().sum())
            total_denom = total_denom + (3.0 * m_t.sum())

            rnn_in = torch.cat([x_input, m_t], dim=-1)
            h, c = self.rnn_f(rnn_in, (h, c)) if not reverse else self.rnn_b(rnn_in, (h, c))

            preds.append(x_c)
            x_prev = x_input

        preds = torch.stack(preds, dim=1)  # (B,L,K)
        if reverse:
            preds = torch.flip(preds, dims=[1])

        return preds, total_loss, total_denom

    def forward(self, x: torch.Tensor, keepmask: torch.Tensor):
        pred_f, loss_f, denom_f = self._run_direction(x, keepmask, reverse=False)
        pred_b, loss_b, denom_b = self._run_direction(x, keepmask, reverse=True)

        fused = 0.5 * (pred_f + pred_b)
        imputed_full = keepmask * x + (1.0 - keepmask) * fused

        consistency = (pred_f - pred_b).abs().mean()

        denom = torch.clamp(denom_f + denom_b, min=1.0)
        train_loss = (loss_f + loss_b) / denom + 0.1 * consistency
        return imputed_full, train_loss


# -----------------------------
# Training / eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, train_masked_ratio: float, seed: int):
    model.train()
    losses = []
    rng = np.random.RandomState(seed)

    for xb in loader:
        xb = xb.to(device).float()
        observed_mask = torch.isfinite(xb).float()
        xb = torch.where(torch.isfinite(xb), xb, torch.zeros_like(xb))

        keep_np = pointwise_keep_mask(xb.shape, train_masked_ratio, rng)
        keep = torch.from_numpy(keep_np).to(device) * observed_mask

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, keep)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_split(model, loader, device, r_m: float, seed: int, split_name: str,
               use_shared_evalmask: bool, shared_evalmask_dir: str):
    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0
    total_keep = 0.0
    total_keep_count = 0.0

    keep_nlk = None
    if use_shared_evalmask:
        keep_nlk = load_shared_keepmask(shared_evalmask_dir, split=split_name, r_m=r_m, seed=seed)
        if keep_nlk.shape[0] != len(loader.dataset):
            raise ValueError(
                f"Mask/window count mismatch for split={split_name}, r={r_m:.2f}: "
                f"mask has N={keep_nlk.shape[0]}, dataset has N={len(loader.dataset)}"
            )

    offset = 0
    rng = np.random.RandomState(seed + 1000)

    for xb in loader:
        xb = xb.to(device).float()
        B = xb.shape[0]
        observed_mask = torch.isfinite(xb).float()
        xb = torch.where(torch.isfinite(xb), xb, torch.zeros_like(xb))

        if keep_nlk is not None:
            keep = torch.from_numpy(keep_nlk[offset:offset + B]).to(device).float()
            keep = keep * observed_mask
            offset += B
        else:
            keep_np = pointwise_keep_mask(xb.shape, r_m, rng)
            keep = torch.from_numpy(keep_np).to(device).float() * observed_mask

        imputed_full, _ = model(xb, keep)

        evalmask = (1.0 - keep) * observed_mask
        diff = (imputed_full - xb) * evalmask

        sum_abs = diff.abs().sum().item()
        sum_sq = (diff ** 2).sum().item()
        denom = evalmask.sum().item()

        total_abs += sum_abs
        total_sq += sum_sq
        total_count += denom
        total_keep += keep.mean().item() * max(denom, 1.0)
        total_keep_count += max(denom, 1.0)

    total_count = max(total_count, 1.0)
    mse = total_sq / total_count
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_count
    r_obs = float(total_keep / max(total_keep_count, 1.0))
    return mae, mse, rmse, r_obs


@torch.no_grad()
def collect_test_arrays(model, loader, device, r_m: float, seed: int,
                        use_shared_evalmask: bool, shared_evalmask_dir: str):
    model.eval()

    gt_all, imp_all, cond_all, eval_all = [], [], [], []

    keep_nlk = None
    if use_shared_evalmask:
        keep_nlk = load_shared_keepmask(shared_evalmask_dir, split="test", r_m=r_m, seed=seed)
        if keep_nlk.shape[0] != len(loader.dataset):
            raise ValueError(
                f"Mask/window count mismatch for split=test, r={r_m:.2f}: "
                f"mask has N={keep_nlk.shape[0]}, dataset has N={len(loader.dataset)}"
            )

    offset = 0
    rng = np.random.RandomState(seed + 2000)

    for xb in loader:
        xb = xb.to(device).float()
        B = xb.shape[0]
        observed_mask = torch.isfinite(xb).float()
        xb = torch.where(torch.isfinite(xb), xb, torch.zeros_like(xb))

        if keep_nlk is not None:
            keep = torch.from_numpy(keep_nlk[offset:offset + B]).to(device).float()
            keep = keep * observed_mask
            offset += B
        else:
            keep_np = pointwise_keep_mask(xb.shape, r_m, rng)
            keep = torch.from_numpy(keep_np).to(device).float() * observed_mask

        imputed_full, _ = model(xb, keep)
        evalmask = (1.0 - keep) * observed_mask

        gt_all.append(xb.cpu().numpy())
        imp_all.append(imputed_full.cpu().numpy())
        cond_all.append(keep.cpu().numpy())
        eval_all.append(evalmask.cpu().numpy())

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

    ap.add_argument("--shared_data_dir", type=str, required=True,
                    help="Directory containing train_windows.npy, val_windows.npy, test_windows.npy and maskbanks")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_size", type=int, default=108)
    ap.add_argument("--train_masked_ratio", type=float, default=0.15)

    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_brits_beijing_sharedmask")
    ap.add_argument("--out_txt", type=str, default="brits_beijing_sharedmask.txt")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    # ---- load shared windows
    train_w, val_w, test_w = load_shared_windows(args.shared_data_dir)
    L = train_w.shape[1]
    K = train_w.shape[2]

    if L != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={L}, but args.seq_len={args.seq_len}")

    print(f"[data] Beijing shared windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={L}")

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
        BeijingWindowDataset(train_w),
        batch_size=safe_batch_size(args.batch, len(train_w)),
        shuffle=True,
        drop_last=(len(train_w) > 1),
    )
    val_loader = DataLoader(
        BeijingWindowDataset(val_w),
        batch_size=safe_batch_size(args.batch, len(val_w)),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        BeijingWindowDataset(test_w),
        batch_size=safe_batch_size(args.batch, len(test_w)),
        shuffle=False,
        drop_last=False,
    )

    model = BRITSBeijing(k=K, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- train
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_masked_ratio=args.train_masked_ratio,
            seed=args.seed + ep,
        )
        if ep == 1 or ep % 10 == 0:
            print(f"[epoch {ep:03d}] loss={loss:.6f}  time={time.time()-t0:.1f}s")

    # ---- evaluate
    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    all_rows = []

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        for r_m in eval_masked:
            mae, mse, rmse, r_obs = eval_split(
                model=model,
                loader=loader,
                device=device,
                r_m=r_m,
                seed=args.seed,
                split_name=split_name,
                use_shared_evalmask=args.use_shared_evalmask,
                shared_evalmask_dir=args.shared_evalmask_dir,
            )
            all_rows.append((split_name, r_m, r_obs, mae, mse, rmse))
            print(
                f"[{split_name}] r_masked={r_m:.2f}  "
                f"MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  r_obs={r_obs:.2f}"
            )

    # ---- save test arrays
    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)
        for r_m in eval_masked:
            gt, imp, cond, evalm = collect_test_arrays(
                model=model,
                loader=test_loader,
                device=device,
                r_m=r_m,
                seed=args.seed,
                use_shared_evalmask=args.use_shared_evalmask,
                shared_evalmask_dir=args.shared_evalmask_dir,
            )
            tag = f"brits_beijing_test_r{r_m:.2f}_seed{args.seed}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), evalm)
            print(
                f"[save] {tag}_*.npy  shapes "
                f"gt={gt.shape} imputed={imp.shape} cond={cond.shape} eval={evalm.shape}"
            )

        meta_out = {
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_data_dir": args.shared_data_dir,
            "hidden_size": int(args.hidden_size),
            "train_masked_ratio": float(args.train_masked_ratio),
        }
        with open(os.path.join(args.save_dir, f"metadata_seed{args.seed}.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

    # ---- write metrics
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split_name, r_m, r_obs, mae, mse, rmse in all_rows:
            f.write(f"{split_name}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
