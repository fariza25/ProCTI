#!/usr/bin/env python3

import os, sys, time, random, argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util


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


# -----------------------------
# Markov keep-mask (segment-based) with correct stationary distribution
# -----------------------------
def markov_keep_mask_KT(
    K: int, T: int, r_masked: float, lm: float, rng: np.random.Generator
) -> np.ndarray:
    """
    keepmask (K,T): 1=kept/observed, 0=masked
    Markov segments along time for each channel independently.
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm                          # 0->1 (masked->keep)
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)  # 1->0 (keep->masked)
    p = [p_m, p_u]                          # index by state (0 masked, 1 keep)

    out = np.ones((K, T), dtype=np.float32)
    for k in range(K):
        state = int(rng.random() < r_keep)
        for t in range(T):
            out[k, t] = state
            if rng.random() < p[state]:
                state = 1 - state
    return out


# -----------------------------
# Load TIDER from a file path
# -----------------------------
def import_tider_class(tider_py: str):
    """Import TIDER class from a python file safely.

    Some TIDER scripts define an argparse CLI at import time and call parse_args()
    unguarded. To avoid them consuming this wrapper's CLI args, we temporarily
    replace sys.argv with a minimal list during import.
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [tider_py]  # prevent unrecognized-args errors from TIDER.py on import
        spec = importlib.util.spec_from_file_location("tider_mod", tider_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv

    if not hasattr(mod, "TIDER"):
        raise AttributeError(f"{tider_py} loaded, but no TIDER class found.")
    return mod.TIDER


# -----------------------------
# stock loading + splitting
# -----------------------------
def load_stock_csv_ordered(csv_path: str, feature_cols_raw: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.columns = df.columns.astype(str)

    if feature_cols_raw is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(cols) == 0:
            raise ValueError("No numeric columns found in stock CSV.")
    else:
        cols = [str(c) for c in list(feature_cols_raw)]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing {len(missing)} columns required by maskbank metadata. Example: {missing[:5]}")

    X = df[cols].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32), cols


def split_indices_simple(T: int) -> Tuple[int, int]:
    train_end = int(0.70 * T)
    val_end = int(0.85 * T)
    return train_end, val_end


def truncate_to_windows(seg_TK: np.ndarray, seq_len: int) -> np.ndarray:
    T, K = seg_TK.shape
    n = T // seq_len
    T2 = n * seq_len
    return seg_TK[:T2].astype(np.float32)


def standardize_by_train_simple(full_TK: np.ndarray, train_T: int, eps=1e-6):
    """
    Compute mean/std on first train_T rows only, then apply to full.
    Fill non-finite with train mean.
    """
    train_flat = full_TK[:train_T].reshape(-1, full_TK.shape[-1])
    mu = np.nanmean(train_flat, axis=0).astype(np.float32)
    sd = np.nanstd(train_flat, axis=0).astype(np.float32)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

    X = full_TK.copy().astype(np.float32)
    nanmask = ~np.isfinite(X)
    if nanmask.any():
        feat_idx = np.where(nanmask)[1]
        X[nanmask] = mu[feat_idx]
    X = (X - mu[None, :]) / sd[None, :]
    if not np.isfinite(X).all():
        raise ValueError("Non-finite after standardization.")
    return X.astype(np.float32), mu, sd


def load_maskbank_meta(shared_dir: str, seq_len: int, seed: int):
    meta_path = os.path.join(shared_dir, f"stock_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    meta = np.load(meta_path, allow_pickle=True)
    return meta_path, meta


def optionA_align_full_matrix(csv_path: str, meta, seq_len: int):

    feature_cols_raw = meta["feature_cols_raw"]
    X_TK_raw, cols_used = load_stock_csv_ordered(csv_path, feature_cols_raw=feature_cols_raw)
    train_end = int(meta["split_row_train_end"])
    val_end   = int(meta["split_row_val_end"])

    Xtr = truncate_to_windows(X_TK_raw[:train_end], seq_len)
    Xva = truncate_to_windows(X_TK_raw[train_end:val_end], seq_len)
    Xte = truncate_to_windows(X_TK_raw[val_end:], seq_len)

    full_raw = np.concatenate([Xtr, Xva, Xte], axis=0)  # (T_full, K_raw)

    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)

    mu = mu_raw[valid]
    sd = sd_raw[valid]
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)

    X = full_raw[:, valid].astype(np.float32)
    nan = ~np.isfinite(X)
    if nan.any():
        feat_idx = np.where(nan)[1]
        X[nan] = mu[feat_idx]
    X = (X - mu[None, :]) / sd[None, :]
    if not np.isfinite(X).all():
        raise ValueError("Non-finite after Option-A z-score.")
    # lengths
    Ttr = Xtr.shape[0]
    Tva = Xva.shape[0]
    Tte = Xte.shape[0]
    return X.astype(np.float32), cols_used, mu, sd, valid, (Ttr, Tva, Tte)


# -----------------------------
# Loss on observed entries (like TIDER obsLossF)
# -----------------------------
def obs_mse_loss(Xhat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(X)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=X.device)
    return F.mse_loss(Xhat[mask], X[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # TIDER.py location
    ap.add_argument("--tider_py", type=str, default="TIDER.py",
                    help="Path to TIDER.py (uploaded). If running elsewhere, pass absolute path.")

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size over channels (roads)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=1e-2, help="L2 weight")
    ap.add_argument("--lambda_ar", type=float, default=0.2)
    ap.add_argument("--lambda_trend", type=float, default=0.1)
    ap.add_argument("--dim_size", type=int, default=50, help="hidden size for embeddings")
    ap.add_argument("--bias_dimension", type=int, default=5)
    ap.add_argument("--lag_list", type=str, default="list(range(5))")
    ap.add_argument("--season_num", type=int, default=30)
    ap.add_argument("--seasonality", type=float, default=168.0)

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # sharedmask option 
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default=None)

    # logging
    ap.add_argument("--out_txt", type=str, default="tider_stock_metrics.txt")
    ap.add_argument("--save_path", type=str, default="TIDER_stock.pt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    TIDER = import_tider_class(args.tider_py)

    ratios = parse_ratios(args.eval_masked_ratios)
    lag_list = eval(args.lag_list)

    # ---------------- data
    if args.use_shared_evalmask:
        if not args.shared_evalmask_dir:
            raise ValueError("--use_shared_evalmask requires --shared_evalmask_dir")
        meta_path, meta = load_maskbank_meta(args.shared_evalmask_dir, args.seq_len, args.seed)
        print(f"[maskbank] {meta_path}")
        Xz_TK, cols_used, mu, sd, valid, (Ttr, Tva, Tte) = optionA_align_full_matrix(args.csv, meta, args.seq_len)
    else:
        X_TK_raw, cols_used = load_stock_csv_ordered(args.csv, feature_cols_raw=None)
        T0 = X_TK_raw.shape[0]
        train_end, val_end = split_indices_simple(T0)

        Xtr = truncate_to_windows(X_TK_raw[:train_end], args.seq_len)
        Xva = truncate_to_windows(X_TK_raw[train_end:val_end], args.seq_len)
        Xte = truncate_to_windows(X_TK_raw[val_end:], args.seq_len)

        full_raw = np.concatenate([Xtr, Xva, Xte], axis=0)  # (T_full, K)
        Ttr, Tva, Tte = Xtr.shape[0], Xva.shape[0], Xte.shape[0]
        train_T = Ttr  # mean/std from train only
        Xz_TK, mu, sd = standardize_by_train_simple(full_raw, train_T)
        valid = None

    T_full = int(Xz_TK.shape[0])
    K = int(Xz_TK.shape[1])
    print(f"[data] rows(full)={T_full}  split_trunc(Ttr/Tva/Tte)={Ttr}/{Tva}/{Tte}  K={K}  seq_len={args.seq_len}")

    # Convert to TIDER matrix: (K,T)
    X_full_KT = torch.from_numpy(Xz_TK.T).to(device)  # (K, T)

    # Time index ranges
    tr0, tr1 = 0, Ttr
    va0, va1 = Ttr, Ttr + Tva
    te0, te1 = Ttr + Tva, T_full

    # Build training observation matrix:
    # - only TRAIN time region is observed (others NaN)
    X_train_KT = torch.full((K, T_full), float("nan"), device=device)
    X_train_KT[:, tr0:tr1] = X_full_KT[:, tr0:tr1]

    # Apply TRAIN masking (segment-based) within train region: set masked entries to NaN
    rng_train = np.random.default_rng(args.seed + 12345)
    keep_train = markov_keep_mask_KT(K, Ttr, args.r_train_masked, args.lm, rng_train)  # (K,Ttr)
    keep_train_t = torch.from_numpy(keep_train).to(device)
    X_train_KT[:, tr0:tr1] = torch.where(keep_train_t > 0, X_train_KT[:, tr0:tr1], torch.tensor(float("nan"), device=device))

    # Validation observation matrix (only VAL time region observed)
    X_val_KT = torch.full((K, T_full), float("nan"), device=device)
    X_val_KT[:, va0:va1] = X_full_KT[:, va0:va1]

    # --------------- Model
    model = TIDER(
        n=K,
        t=T_full,
        hid_size=args.dim_size,
        bias_lag_list=lag_list,
        bias_dim_=args.bias_dimension,
        season_num_=args.season_num,
        seasonality_=args.seasonality,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    
    def l2loss(eta: float):
        if eta <= 0:
            return torch.tensor(0.0, device=device)
        roads_all = torch.arange(K, device=device)
        times_all = torch.arange(T_full, device=device)
        u = model.getu(roads_all)
        v = model.getv(times_all)
        return eta * (torch.linalg.norm(u) + torch.linalg.norm(v))

    def arloss_bias(lambda_ar: float):
        if lambda_ar <= 0:
            return torch.tensor(0.0, device=device)
        times_all = torch.arange(T_full, device=device)
        y = model.bias_loss(times_all)
        return lambda_ar * torch.linalg.norm(y)

    def trend_loss(lambda_trend: float):
        if lambda_trend <= 0:
            return torch.tensor(0.0, device=device)
        times_all = torch.arange(T_full, device=device)
        trend = model.t_embeddings_trend(times_all)  # (T,dim)
        diff = trend[:, 1:] - trend[:, :-1]
        return lambda_trend * torch.linalg.norm(diff)

    def forward_KT() -> torch.Tensor:
        roads_all = torch.arange(K, device=device, dtype=torch.long)
        # model(roads_all) -> (K, T_full)
        return model(roads_all)

    # --------------- Train with val selection by val obs mse
    best_val = float("inf")
    best_ep = -1

    idx = np.arange(K)
    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)
        losses = []

        # channel mini-batches
        for st in range(0, K, args.batch_size):
            roads = torch.from_numpy(idx[st:st + args.batch_size]).to(device=device, dtype=torch.long)

            Xhat = model(roads)  # (B, T_full)
            Xobs = X_train_KT[roads]  # (B, T_full)

            loss = obs_mse_loss(Xhat, Xobs) + l2loss(args.eta) + arloss_bias(args.lambda_ar) + trend_loss(args.lambda_trend)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        # validation (no grad)
        if ep == 1 or ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                Xhat_all = forward_KT()
                val_loss = obs_mse_loss(Xhat_all[:, va0:va1], X_val_KT[:, va0:va1]).item() if (va1 > va0) else 0.0
            print(f"[train] epoch={ep:03d} loss={float(np.mean(losses)):.6f}  val_obs_mse={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_ep = ep
                torch.save(model.state_dict(), args.save_path)

    print(f"[train] best_epoch={best_ep} best_val_obs_mse={best_val:.6f}  saved={args.save_path}")

    # load best
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()

    # --------------- Evaluate on TEST region with segment-based masking
    if not os.path.exists(args.out_txt):
        with open(args.out_txt, "w") as f:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

    with torch.no_grad():
        Xhat_all = forward_KT()  # (K, T_full)
        gt_test = X_full_KT[:, te0:te1]  # (K, Tte)

        for r_m in ratios:
            rng = np.random.default_rng(args.seed + int(round(r_m * 1000)) + 777)
            keep = markov_keep_mask_KT(K, Tte, r_m, args.lm, rng)  # (K,Tte)
            keep_t = torch.from_numpy(keep).to(device).float()
            evalmask = 1.0 - keep_t

            pred_test = Xhat_all[:, te0:te1]

            finite = torch.isfinite(gt_test)
            m = (evalmask > 0) & finite
            den = m.sum().clamp_min(1)

            diff = (pred_test - gt_test)
            mae = (diff.abs()[m]).sum().item() / den.item()
            mse = ((diff[m]) ** 2).sum().item() / den.item()
            rmse = float(np.sqrt(mse))
            r_obs = float(keep_t.mean().item())

            print(f"[test] r_masked={r_m:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  r_obs={r_obs:.2f}")
            with open(args.out_txt, "a") as f:
                f.write(f"test\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()

