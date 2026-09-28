#!/usr/bin/env python3

import os, sys, time, random, argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
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
def markov_keep_mask_KT(K: int, T: int, r_masked: float, lm: float, rng: np.random.Generator) -> np.ndarray:
    """
    keepmask (K,T): 1=kept/observed, 0=masked
    Markov segments along time for each channel independently.
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]  # state 0 masked, 1 keep

    out = np.ones((K, T), dtype=np.float32)
    for k in range(K):
        state = int(rng.random() < r_keep)
        for t in range(T):
            out[k, t] = state
            if rng.random() < p[state]:
                state = 1 - state
    return out


# -----------------------------
# Safe import of TIDER
# -----------------------------
def import_tider_class(tider_py: str):
    """Import TIDER class from a python file safely.

    Some TIDER scripts define an argparse CLI at import time and call parse_args()
    unguarded. To avoid them consuming this wrapper's CLI args, we temporarily
    replace sys.argv with a minimal list during import.
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [tider_py]
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
# Weather loading
# -----------------------------
def load_weather_windows(csv_path: str, seq_len: int, time_col: Optional[str]):
    df = pd.read_csv(csv_path)

    if time_col is None:
        if "date" in df.columns:
            time_col = "date"
        elif "time" in df.columns:
            time_col = "time"
        else:
            time_col = df.columns[0]

    t = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[~t.isna()].copy()
    df["_t"] = pd.to_datetime(df[time_col])
    df = df.sort_values("_t", kind="mergesort").drop(columns=["_t"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found in weather CSV.")
    X = df[num_cols].astype(np.float32).to_numpy()
    X[~np.isfinite(X)] = np.nan

    T, K = X.shape
    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}")
    Xw = X[: n_win * seq_len].reshape(n_win, seq_len, K)  # (N,L,K)
    return Xw.astype(np.float32), num_cols, time_col


def chrono_split_windows(windows_NLK: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(windows_NLK)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = windows_NLK[:n_train]
    val = windows_NLK[n_train:n_train + n_val]
    test = windows_NLK[n_train + n_val:]
    return train, val, test


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0).astype(np.float32)
    sd = np.nanstd(train_flat, axis=0).astype(np.float32)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = ~np.isfinite(x)
        if nanmask.any():
            feat_idx = np.where(nanmask)[2]
            x[nanmask] = mu[feat_idx]
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd


def windows_to_TK(w_NLK: np.ndarray) -> np.ndarray:
    # (N,L,K) -> (T,K) with T=N*L, preserving chronological order
    return w_NLK.reshape(-1, w_NLK.shape[-1]).astype(np.float32)


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
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # TIDER.py location
    ap.add_argument("--tider_py", type=str, default="TIDER.py")

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size over channels")
    ap.add_argument("--lr", type=float, default=0.05)

    # regularizers
    ap.add_argument("--eta", type=float, default=1e-2, help="L2 weight")
    ap.add_argument("--lambda_ar", type=float, default=0.2)
    ap.add_argument("--lambda_trend", type=float, default=0.1)
    ap.add_argument("--dim_size", type=int, default=50)
    ap.add_argument("--bias_dimension", type=int, default=5)
    ap.add_argument("--lag_list", type=str, default="list(range(5))")
    ap.add_argument("--season_num", type=int, default=30)
    ap.add_argument("--seasonality", type=float, default=168.0)

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # outputs
    ap.add_argument("--out_txt", type=str, default="tider_weather_metrics.txt")
    ap.add_argument("--save_path", type=str, default="TIDER_weather.pt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    TIDER = import_tider_class(args.tider_py)
    ratios = parse_ratios(args.eval_masked_ratios)
    lag_list = eval(args.lag_list)

    # --------- data 
    windows, feat_cols, tcol = load_weather_windows(args.csv, args.seq_len, args.time_col)
    train_w, val_w, test_w = chrono_split_windows(windows)
    train_w, val_w, test_w, mu, sd = standardize_by_train(train_w, val_w, test_w)

    Ntr, Nva, Nte = len(train_w), len(val_w), len(test_w)
    K = int(train_w.shape[-1])
    Ttr = int(Ntr * args.seq_len)
    Tva = int(Nva * args.seq_len)
    Tte = int(Nte * args.seq_len)

    # build full matrix (T,K) then transpose to (K,T)
    full_TK = np.concatenate([windows_to_TK(train_w), windows_to_TK(val_w), windows_to_TK(test_w)], axis=0)
    T_full = int(full_TK.shape[0])

    X_full_KT = torch.from_numpy(full_TK.T).to(device)  # (K, T_full)

    # time ranges
    tr0, tr1 = 0, Ttr
    va0, va1 = Ttr, Ttr + Tva
    te0, te1 = Ttr + Tva, T_full

    print(f"[data] windows train/val/test={Ntr}/{Nva}/{Nte}  rows(train/val/test)={Ttr}/{Tva}/{Tte}  K={K}  L={args.seq_len}  time_col={tcol}")

    # training observation matrix: only train segment observed
    X_train_KT = torch.full((K, T_full), float("nan"), device=device)
    X_train_KT[:, tr0:tr1] = X_full_KT[:, tr0:tr1]

    # apply train missingness within train segment (segment-based)
    rng_train = np.random.default_rng(args.seed + 12345)
    keep_train = markov_keep_mask_KT(K, Ttr, args.r_train_masked, args.lm, rng_train)
    keep_train_t = torch.from_numpy(keep_train).to(device)
    X_train_KT[:, tr0:tr1] = torch.where(
        keep_train_t > 0,
        X_train_KT[:, tr0:tr1],
        torch.tensor(float("nan"), device=device),
    )

    # validation observation matrix: only val segment observed
    X_val_KT = torch.full((K, T_full), float("nan"), device=device)
    X_val_KT[:, va0:va1] = X_full_KT[:, va0:va1]

    # --------- model
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

    # regularizers (same as stock wrapper)
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
        trend = model.t_embeddings_trend(times_all)
        diff = trend[:, 1:] - trend[:, :-1]
        return lambda_trend * torch.linalg.norm(diff)

    def forward_KT() -> torch.Tensor:
        roads_all = torch.arange(K, device=device, dtype=torch.long)
        return model(roads_all)

    # --------- train with best val obs mse
    best_val = float("inf")
    best_ep = -1
    idx = np.arange(K)

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)
        losses = []

        for st in range(0, K, args.batch_size):
            roads = torch.from_numpy(idx[st:st + args.batch_size]).to(device=device, dtype=torch.long)

            Xhat = model(roads)        # (B, T_full)
            Xobs = X_train_KT[roads]   # (B, T_full)

            loss = obs_mse_loss(Xhat, Xobs) + l2loss(args.eta) + arloss_bias(args.lambda_ar) + trend_loss(args.lambda_trend)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

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

    # --------- evaluation on test region with segment-based masking
    if not os.path.exists(args.out_txt):
        with open(args.out_txt, "w") as f:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

    with torch.no_grad():
        Xhat_all = forward_KT()            # (K, T_full)
        gt_test = X_full_KT[:, te0:te1]    # (K, Tte)
        pred_test = Xhat_all[:, te0:te1]

        for r_m in ratios:
            rng = np.random.default_rng(args.seed + int(round(r_m * 1000)) + 777)
            keep = markov_keep_mask_KT(K, Tte, r_m, args.lm, rng)
            keep_t = torch.from_numpy(keep).to(device).float()
            evalmask = 1.0 - keep_t

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

