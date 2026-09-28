#!/usr/bin/env python3

import os
import time
import random
import argparse
from types import SimpleNamespace
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.iTransformer import Model as iTransformerModel


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


# -----------------------------
# Markov keep-mask (segment-based) with correct stationary distribution
# -----------------------------
def markov_keep_mask_from_masked_ratio(
    B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device
) -> torch.Tensor:
    """
    Returns (B,K,L) keepmask with 1=kept, 0=masked.
    r_masked is fraction masked (missingness).
    """
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
# PhysioNet helpers
# -----------------------------
def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    pid_col = "Patient_ID"
    if pid_col not in df.columns:
        raise ValueError(f"{pid_col} column not found in CSV.")

    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce")
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)
    return df


def get_time_col(df: pd.DataFrame) -> str:
    if "ICULOS" in df.columns:
        return "ICULOS"
    if "Hour" in df.columns:
        return "Hour"
    raise ValueError("Expected a time column ICULOS or Hour in physionet2019.csv.")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {"Patient_ID"}
    for c in ["SepsisLabel", "Hour", "ICULOS", "HospAdmTime"]:
        if c in df.columns:
            drop_cols.add(c)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found after dropping id/labels/time columns.")
    return feat_cols


def split_patients(df: pd.DataFrame, seed: int, train_ratio=0.7, val_ratio=0.15) -> Tuple[Set[int], Set[int], Set[int]]:
    pids = df["Patient_ID"].dropna().unique().tolist()
    pids = [int(x) for x in pids]
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(pids[:n_train])
    val_p = set(pids[n_train:n_train + n_val])
    test_p = set(pids[n_train + n_val:])
    return train_p, val_p, test_p


def make_windows_for_patients(
    df: pd.DataFrame,
    patient_ids: Set[int],
    feat_cols: List[str],
    time_col: str,
    seq_len: int,
) -> np.ndarray:
    patient_ids = set(int(x) for x in patient_ids)
    windows = []

    for pid, g in df.groupby("Patient_ID", sort=False):
        pid = int(pid)
        if pid not in patient_ids:
            continue
        g = g.sort_values(time_col, kind="mergesort")
        X = g[feat_cols].astype(np.float32).ffill().bfill()
        arr = X.to_numpy()
        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))

    if not windows:
        raise ValueError("No windows formed for this split (check seq_len or patient IDs).")
    return np.concatenate(windows, axis=0).astype(np.float32)


def standardize_by_train_drop_invalid(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):

    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All features invalid / zero-std after TRAIN stats. Check CSV.")

    train = train[:, :, valid]
    val = val[:, :, valid]
    test = test[:, :, valid]
    mu = mu[valid].astype(np.float32)
    sd = np.maximum(sd[valid], eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = np.isnan(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = np.take(mu, ks)
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


# -----------------------------
# Maskbank metadata 
# -----------------------------
def load_maskbank_metadata(mask_dir: str, seq_len: int, seed: int) -> Dict[str, np.ndarray]:
    meta_path = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Maskbank metadata not found: {meta_path}")
    z = np.load(meta_path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def enforce_feature_order(df: pd.DataFrame, feature_cols_raw: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols_raw if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing {len(missing)} columns required by maskbank metadata. Example: {missing[:5]}")
    return df.copy()


def standardize_with_maskbank(train: np.ndarray, val: np.ndarray, test: np.ndarray, meta: Dict[str, np.ndarray]):
    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)
    eps = float(meta["eps"]) if "eps" in meta else 1e-6

    if train.shape[-1] != len(valid):
        raise ValueError(
            f"[maskbank] K_raw mismatch: data K={train.shape[-1]} but valid_feature_mask len={len(valid)}. "
            f"Ensure you used feature_cols_raw ordering from metadata."
        )

    mu = mu_raw[valid]
    sd = np.maximum(sd_raw[valid], eps)

    def fill_and_z(x):
        x = x[:, :, valid].copy().astype(np.float32)
        nanmask = np.isnan(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = np.take(mu, ks)
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization (maskbank scaling).")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu.astype(np.float32), sd.astype(np.float32), valid


def load_shared_keepmask(
    mask_dir: str,
    split: str,
    r_m: float,
    seq_len: int,
    seed: int,
    invert: bool,
    *,
    valid_feature_mask: Optional[np.ndarray] = None,
    expected_K: Optional[int] = None,
) -> np.ndarray:
   
    path = os.path.join(mask_dir, f"physionet_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must have shape (N,L,K), got {keep.shape} from {path}")
    if invert or os.environ.get("PHYSIONET_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep

    if expected_K is None:
        return keep
    if int(keep.shape[2]) == int(expected_K):
        return keep

    if valid_feature_mask is not None:
        valid_feature_mask = np.asarray(valid_feature_mask).astype(bool)
        K_full = int(len(valid_feature_mask))
        K_kept = int(valid_feature_mask.sum())
        if int(expected_K) != K_kept:
            raise ValueError(f"Internal mismatch: expected_K={expected_K} but valid_feature_mask.sum()={K_kept}.")
        if int(keep.shape[2]) == K_full:
            keep2 = keep[:, :, valid_feature_mask]
            if int(keep2.shape[2]) != int(expected_K):
                raise ValueError(f"After slicing keepmask K={keep2.shape[2]} != expected_K={expected_K}. File: {path}")
            return keep2

    raise ValueError(
        f"Shared keepmask K mismatch: maskbank K={keep.shape[2]} but model expects K={expected_K}. "
        f"File: {path}."
    )


# -----------------------------
# Dataset
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i])  # (L,K)


# -----------------------------
# iTransformer helpers
# -----------------------------
def build_configs(args, K: int):
    return SimpleNamespace(
        task_name="imputation",
        seq_len=int(args.seq_len),
        pred_len=int(args.seq_len),
        enc_in=int(K),
        c_out=int(K),

        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        e_layers=int(args.e_layers),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        factor=int(args.factor),
        activation=str(args.activation),

        embed=str(args.embed),
        freq=str(args.freq),

        output_attention=bool(args.output_attention),
    )


def model_forward_impute(model: torch.nn.Module, x_enc: torch.Tensor) -> torch.Tensor:
    out = model(x_enc, None, None, None, None)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, opt, device, K: int, L: int, r_train_masked: float, lm: float, clip_grad: float):
    model.train()
    losses = []

    for xb in loader:
        xb = xb.to(device).float()  # (B,L,K)

        keep_BKL = markov_keep_mask_from_masked_ratio(B=xb.shape[0], K=K, L=L, r_masked=r_train_masked, lm=lm, device=device)
        keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()  # (B,L,K)

        x_obs = xb * keep_BLK
        pred = model_forward_impute(model, x_obs)

        evalmask = 1.0 - keep_BLK
        denom = evalmask.sum().clamp_min(1.0)
        loss = ((pred - xb).abs() * evalmask).sum() / denom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        opt.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_split(model, loader, split_name: str, ratios: List[float], device, K: int, L: int,
               lm: float, use_shared: bool, shared_dir: Optional[str], seed: int, invert_shared: bool,
               valid_feature_mask: Optional[np.ndarray] = None):
    model.eval()
    rows = []

    for r_m in ratios:
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0.0

        total_keep = 0.0
        total_keep_count = 0.0

        keep_NLK = None
        offset = 0
        if use_shared:
            keep_NLK = load_shared_keepmask(
                shared_dir, split_name, r_m, L, seed, invert_shared,
                valid_feature_mask=valid_feature_mask, expected_K=K
            )

        for xb in loader:
            xb = xb.to(device).float()
            B = xb.shape[0]

            if keep_NLK is not None:
                keep_BLK = torch.from_numpy(keep_NLK[offset:offset + B]).to(device).float()
                offset += B
            else:
                keep_BKL = markov_keep_mask_from_masked_ratio(B=B, K=K, L=L, r_masked=r_m, lm=lm, device=device)
                keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

            x_obs = xb * keep_BLK
            pred = model_forward_impute(model, x_obs)

            evalmask = 1.0 - keep_BLK
            diff = (pred - xb) * evalmask

            total_abs += float(diff.abs().sum().item())
            total_sq += float((diff ** 2).sum().item())
            total_count += float(evalmask.sum().item())

            total_keep += float(keep_BLK.sum().item())
            total_keep_count += float(keep_BLK.numel())

        mae = total_abs / max(total_count, 1.0)
        mse = total_sq / max(total_count, 1.0)
        rmse = float(np.sqrt(mse))
        r_obs = total_keep / max(total_keep_count, 1.0)

        rows.append((split_name, r_m, r_obs, mae, mse, rmse))

    return rows


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)

    # training
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_grad", type=float, default=5.0)

    # iTransformer hyperparams
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--embed", type=str, default="fixed")
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument("--output_attention", action="store_true")

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # shared eval masks (optional)
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default=None)
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # output
    ap.add_argument("--out_txt", type=str, default="itransformer_physionet_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    ratios = parse_ratios(args.eval_masked_ratios)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        raise ValueError("--use_shared_evalmask requires --shared_evalmask_dir")

    # ---- load df
    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)

    meta = None
    valid_feature_mask = None

    # ---- if using shared masks, enforce Option-A metadata ordering + splits + scaling
    if args.use_shared_evalmask:
        meta = load_maskbank_metadata(args.shared_evalmask_dir, args.seq_len, args.seed)
        feature_cols_raw = [str(c) for c in list(meta["feature_cols_raw"])]
        # ensure df has required columns; we will select by feature_cols_raw ordering
        _ = enforce_feature_order(df, feature_cols_raw)

        # Option-A patient split from metadata (preferred)
        if "patients_train" in meta and "patients_val" in meta and "patients_test" in meta:
            train_p = set(int(x) for x in meta["patients_train"].tolist())
            val_p = set(int(x) for x in meta["patients_val"].tolist())
            test_p = set(int(x) for x in meta["patients_test"].tolist())
        else:
            train_p, val_p, test_p = split_patients(df, args.seed)

        feat_cols = feature_cols_raw
    else:
        train_p, val_p, test_p = split_patients(df, args.seed)
        feat_cols = get_feature_cols(df)

    # ---- windows (N,L,K_raw)
    train_raw = make_windows_for_patients(df, train_p, feat_cols, time_col, args.seq_len)
    val_raw = make_windows_for_patients(df, val_p, feat_cols, time_col, args.seq_len)
    test_raw = make_windows_for_patients(df, test_p, feat_cols, time_col, args.seq_len)

    # ---- scaling
    if meta is not None:
        train_w, val_w, test_w, mu, sd, valid_feature_mask = standardize_with_maskbank(train_raw, val_raw, test_raw, meta)
    else:
        train_w, val_w, test_w, mu, sd, valid_feature_mask = standardize_by_train_drop_invalid(train_raw, val_raw, test_raw)

    K = int(train_w.shape[-1])
    L = int(args.seq_len)

    print(f"[physionet] windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  L={L} K={K}  time_col={time_col}")

    train_loader = DataLoader(WindowDataset(train_w), batch_size=safe_batch_size(args.batch, len(train_w)),
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(WindowDataset(val_w), batch_size=safe_batch_size(args.batch, len(val_w)),
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowDataset(test_w), batch_size=safe_batch_size(args.batch, len(test_w)),
                             shuffle=False, drop_last=False)

    # ---- model
    configs = build_configs(args, K)
    model = iTransformerModel(configs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- train
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, opt, device, K=K, L=L,
                           r_train_masked=args.r_train_masked, lm=args.lm, clip_grad=args.clip_grad)
        if ep == 1 or ep % 10 == 0:
            print(f"[train] epoch={ep:03d} loss={loss:.6f} time={time.time()-t0:.1f}s")

    # ---- eval + write metrics
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    new_file = (not os.path.exists(args.out_txt)) or (os.path.getsize(args.out_txt) == 0)

    with open(args.out_txt, "a") as f:
        if new_file:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            rows = eval_split(
                model, loader, split_name, ratios, device,
                K=K, L=L, lm=args.lm,
                use_shared=args.use_shared_evalmask,
                shared_dir=args.shared_evalmask_dir,
                seed=args.seed,
                invert_shared=args.invert_shared_keepmask,
                valid_feature_mask=valid_feature_mask
            )
            for (sp, r_m, r_obs, mae, mse, rmse) in rows:
                f.write(f"{sp}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{sp}] r_masked={r_m:.2f} r_obs={r_obs:.2f}  MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
