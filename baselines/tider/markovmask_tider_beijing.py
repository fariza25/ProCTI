#!/usr/bin/env python3


import os
import sys
import time
import json
import math
import random
import argparse
import importlib.util
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
# Markov keep-mask
# -----------------------------
def markov_keep_mask_KT(K: int, T: int, r_masked: float, lm: float, rng: np.random.Generator) -> np.ndarray:
    """
    keepmask (K,T): 1=kept/observed, 0=masked
    Markov segments along time for each channel independently.
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]

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
    """
    Import TIDER class from a python file safely
    (avoid argparse side effects during import).
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
# Shared Beijing loaders
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
            f"Expected all windows to have shape (N,L,K), got "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    if train_w.shape[1:] != val_w.shape[1:] or train_w.shape[1:] != test_w.shape[1:]:
        raise ValueError(
            f"Shared window shape mismatch: "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    return train_w, val_w, test_w


def load_shared_keepmask_nlk(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    """
    Loads evalmask from:
      {split}_maskbank_seed{seed}.npz
    key like "0.10"

    Saved mask semantics:
      1 = masked
      0 = observed

    Returns keepmask:
      1 = observed
      0 = masked
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


def windows_to_TK(w_nlk: np.ndarray) -> np.ndarray:
    """
    (N,L,K) -> (T,K) by flattening windows in order
    """
    return w_nlk.reshape(-1, w_nlk.shape[-1]).astype(np.float32)


def keepmask_nlk_to_KT(keep_nlk: np.ndarray) -> np.ndarray:
    """
    (N,L,K) keepmask -> (K,T)
    """
    keep_tk = keep_nlk.reshape(-1, keep_nlk.shape[-1]).astype(np.float32)  # (T,K)
    return keep_tk.T  # (K,T)


def kt_to_nlk(x_kt: np.ndarray, n_windows: int, seq_len: int, k: int) -> np.ndarray:
    """
    (K,T) -> (N,L,K), where T=N*L
    """
    x_tk = x_kt.T
    return x_tk.reshape(n_windows, seq_len, k)


# -----------------------------
# Loss on observed entries
# -----------------------------
def obs_mse_loss(Xhat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(X)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=X.device)
    return F.mse_loss(Xhat[mask], X[mask])


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", type=str, required=True,
                    help="Directory containing train_windows.npy, val_windows.npy, test_windows.npy and maskbanks")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # TIDER
    ap.add_argument("--tider_py", type=str)

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size over channels")
    ap.add_argument("--lr", type=float, default=0.05)

    # regularizers
    ap.add_argument("--eta", type=float, default=1e-2)
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

    # shared eval masks
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    # outputs
    ap.add_argument("--out_txt", type=str, default="tider_beijing_sharedmask.txt")
    ap.add_argument("--save_path", type=str, default="TIDER_beijing.pt")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_tider_beijing_sharedmask")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    TIDER = import_tider_class(args.tider_py)
    ratios = parse_ratios(args.eval_masked_ratios)
    lag_list = eval(args.lag_list)

    # -------- load shared windows
    train_w, val_w, test_w = load_shared_windows(args.shared_data_dir)

    Ntr, Nva, Nte = len(train_w), len(val_w), len(test_w)
    L = int(train_w.shape[1])
    K = int(train_w.shape[2])

    if L != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={L}, but args.seq_len={args.seq_len}")

    Ttr = int(Ntr * L)
    Tva = int(Nva * L)
    Tte = int(Nte * L)

    full_tk = np.concatenate(
        [windows_to_TK(train_w), windows_to_TK(val_w), windows_to_TK(test_w)],
        axis=0
    )  # (T_full, K)
    T_full = int(full_tk.shape[0])

    X_full_KT = torch.from_numpy(full_tk.T).to(device)  # (K, T_full)

    tr0, tr1 = 0, Ttr
    va0, va1 = Ttr, Ttr + Tva
    te0, te1 = Ttr + Tva, T_full

    print(f"[data] Beijing shared windows train/val/test={Ntr}/{Nva}/{Nte}  K={K}  L={L}")

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

    # -------- train observations: only TRAIN region observed
    X_train_KT = torch.full((K, T_full), float("nan"), device=device)
    X_train_KT[:, tr0:tr1] = X_full_KT[:, tr0:tr1]

    # Additional segment-based masking inside TRAIN region
    rng_train = np.random.default_rng(args.seed + 12345)
    keep_train = markov_keep_mask_KT(K, Ttr, args.r_train_masked, args.lm, rng_train)
    keep_train_t = torch.from_numpy(keep_train).to(device)

    X_train_KT[:, tr0:tr1] = torch.where(
        keep_train_t > 0,
        X_train_KT[:, tr0:tr1],
        torch.tensor(float("nan"), device=device),
    )

    # Validation observations
    X_val_KT = torch.full((K, T_full), float("nan"), device=device)
    X_val_KT[:, va0:va1] = X_full_KT[:, va0:va1]

    # -------- model
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
        trend = model.t_embeddings_trend(times_all)
        diff = trend[:, 1:] - trend[:, :-1]
        return lambda_trend * torch.linalg.norm(diff)

    def forward_KT() -> torch.Tensor:
        roads_all = torch.arange(K, device=device, dtype=torch.long)
        return model(roads_all)

    # -------- train with best val obs mse
    best_val = float("inf")
    best_ep = -1
    idx = np.arange(K)

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)
        losses = []

        for st in range(0, K, args.batch_size):
            roads = torch.from_numpy(idx[st:st + args.batch_size]).to(device=device, dtype=torch.long)
            Xhat = model(roads)       # (B, T_full)
            Xobs = X_train_KT[roads]  # (B, T_full)

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

    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()

    # -------- write metrics header
    if not os.path.exists(args.out_txt):
        with open(args.out_txt, "w") as f:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

    # -------- evaluation helper
    def evaluate_region(split_name: str, region_gt: torch.Tensor, region_start: int, region_end: int,
                        n_windows: int):
        with torch.no_grad():
            Xhat_all = forward_KT()                      # (K, T_full)
            pred_region = Xhat_all[:, region_start:region_end]  # (K, T_region)

            rows = []
            for r_m in ratios:
                if args.use_shared_evalmask:
                    keep_nlk = load_shared_keepmask_nlk(
                        args.shared_evalmask_dir, split=split_name, r_m=r_m, seed=args.seed
                    )
                    if keep_nlk.shape != (n_windows, L, K):
                        raise ValueError(
                            f"Shared keepmask shape mismatch for split={split_name}, r={r_m:.2f}: "
                            f"keep={keep_nlk.shape}, expected={(n_windows, L, K)}"
                        )
                    keep = torch.from_numpy(keepmask_nlk_to_KT(keep_nlk)).to(device).float()  # (K,T_region)
                else:
                    rng = np.random.default_rng(args.seed + int(round(r_m * 1000)) + (111 if split_name == "val" else 777))
                    keep_np = markov_keep_mask_KT(K, region_end - region_start, r_m, args.lm, rng)
                    keep = torch.from_numpy(keep_np).to(device).float()

                evalmask = 1.0 - keep
                finite = torch.isfinite(region_gt)
                m = (evalmask > 0) & finite
                den = m.sum().clamp_min(1)

                diff = pred_region - region_gt
                mae = diff.abs()[m].sum().item() / den.item()
                mse = (diff[m] ** 2).sum().item() / den.item()
                rmse = float(np.sqrt(mse))
                r_obs = float(keep.mean().item())

                print(f"[{split_name}] r_masked={r_m:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  r_obs={r_obs:.2f}")
                with open(args.out_txt, "a") as f:
                    f.write(f"{split_name}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

                rows.append((r_m, keep))
            return pred_region, rows

    # -------- val/test evaluation
    gt_val = X_full_KT[:, va0:va1]
    gt_test = X_full_KT[:, te0:te1]

    _, _ = evaluate_region("val", gt_val, va0, va1, Nva)
    pred_test, test_rows = evaluate_region("test", gt_test, te0, te1, Nte)

    # -------- save test arrays
    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)

        gt_test_np = gt_test.detach().cpu().numpy()  # (K,Tte)
        pred_test_np = pred_test.detach().cpu().numpy()

        for r_m, keep_t in test_rows:
            keep_np = keep_t.detach().cpu().numpy()      # (K,Tte)
            eval_np = 1.0 - keep_np
            imputed_full_np = keep_np * gt_test_np + (1.0 - keep_np) * pred_test_np

            gt_nlk = kt_to_nlk(gt_test_np, Nte, L, K)
            imp_nlk = kt_to_nlk(imputed_full_np, Nte, L, K)
            cond_nlk = kt_to_nlk(keep_np, Nte, L, K)
            eval_nlk = kt_to_nlk(eval_np, Nte, L, K)

            tag = f"tider_beijing_test_r{r_m:.2f}_seed{args.seed}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_nlk)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_nlk)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_nlk)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_nlk)

            print(
                f"[save] {tag}_*.npy  shapes "
                f"gt={gt_nlk.shape} imputed={imp_nlk.shape} cond={cond_nlk.shape} eval={eval_nlk.shape}"
            )

        meta_out = {
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_data_dir": args.shared_data_dir,
            "r_train_masked": float(args.r_train_masked),
            "lm": float(args.lm),
        }
        with open(os.path.join(args.save_dir, f"metadata_seed{args.seed}.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
