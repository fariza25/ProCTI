#!/usr/bin/env python3

import os, sys, time, argparse, random
from typing import List, Tuple

import numpy as np
import pandas as pd
import pickle as pk
import torch
from torch.optim import Adam

# Repo-root on path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from dataloader.dataloader import generate_train_dataloader, generate_val_test_dataloader
from models.model import MTSCI


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------
# Weather -> window -> chrono split (70/15/15 windows) -> zscore(train) -> pickles + scaler.pkl
# -------------------------
def make_weather_mtsci_pickles(
    csv_path: str,
    out_dir: str,
    seq_len: int,
    time_col: str | None = None,
) -> Tuple[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if time_col is None:
        for cand in ["date", "datetime", "time", "timestamp", "Date", "DATE", "Time", "Timestamp"]:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[~df[time_col].isna()].copy()
    df = df.sort_values(time_col, kind="mergesort")

    # Numeric features only; drop all-NaN cols
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = df_num.dropna(axis=1, how="all")
    cols = list(df_num.columns)

    X = df_num.to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    T, K = X.shape

    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}.")
    Xw = X[: n_win * seq_len].reshape(n_win, seq_len, K)  # (N,L,K)

    # Chronological split over windows
    n_train = int(np.floor(0.70 * n_win))
    n_val = int(np.floor(0.15 * n_win))
    train_w = Xw[:n_train]
    val_w = Xw[n_train:n_train + n_val]
    test_w = Xw[n_train + n_val:]

    # Standardize by TRAIN only (important for stability)
    train_flat = train_w.reshape(-1, K)
    mean = np.nanmean(train_flat, axis=0).astype(np.float32)
    std = np.nanstd(train_flat, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def zscore(arr):
        arr = arr.astype(np.float32)
        # fill NaNs with train mean
        nanmask = ~np.isfinite(arr)
        if nanmask.any():
            feat_idx = np.where(nanmask)[2]
            arr[nanmask] = mean[feat_idx]
        return ((arr - mean[None, None, :]) / std[None, None, :]).astype(np.float32)

    train_z = zscore(train_w)
    val_z = zscore(val_w)
    test_z = zscore(test_w)

    # MTSCI dataloader expects (T,K) arrays in pickles, not windowed arrays
    pk.dump(train_z.reshape(-1, K), open(os.path.join(out_dir, "train_set.pkl"), "wb"))
    pk.dump(val_z.reshape(-1, K), open(os.path.join(out_dir, "val_set.pkl"), "wb"))
    pk.dump(test_z.reshape(-1, K), open(os.path.join(out_dir, "test_set.pkl"), "wb"))
    pk.dump((mean, std), open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    print(f"Weather cols ({K}): {cols[:10]}{'...' if len(cols) > 10 else ''}")
    print(f"Weather windows: total={n_win}, train={n_train}, val={n_val}, test={len(test_w)}")
    return out_dir, K


# -------------------------
# Markov segment-wise keepmask (1 keep / 0 masked)
# -------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float) -> np.ndarray:
    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]  # state 0 masked, 1 keep

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < p[state]:
                    state = 1 - state
    return out


# -------------------------
def override_test_batch_with_markov(batch, r_masked: float, lm: float, device: torch.device):
    X, mask, X_Tilde, gt_mask, indicating = batch
    B, L, K = X_Tilde.shape

    keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_masked=r_masked, lm=lm)  # (B,K,L)
    keep_BLK = np.transpose(keep_BKL, (0, 2, 1))  # (B,L,K)
    keep_BLK = torch.from_numpy(keep_BLK).to(device=device)

    # indicating_mask = 1 where masked (targets)
    indicating_new = (1.0 - keep_BLK) * gt_mask.to(device=device)

    # observed inputs are kept positions
    X_new = X_Tilde.to(device=device) * (1.0 - indicating_new)
    mask_new = gt_mask.to(device=device) * (1.0 - indicating_new)

    return (X_new, mask_new, X_Tilde.to(device=device), gt_mask.to(device=device), indicating_new)


def masked_metrics_from_samples(samples_BnKL: torch.Tensor, X_true_BKL: torch.Tensor, eval_mask_BKL: torch.Tensor):
    # mean over samples
    pred = samples_BnKL.mean(dim=1)  # (B,K,L)
    m = eval_mask_BKL
    denom = torch.clamp(m.sum(), min=1.0)
    mae = (torch.abs(pred - X_true_BKL) * m).sum() / denom
    mse = (((pred - X_true_BKL) ** 2) * m).sum() / denom
    rmse = torch.sqrt(mse + 1e-12)
    return mae.item(), mse.item(), rmse.item()


def _append_row(out_txt: str, header: str, row: str):
    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = (not os.path.exists(out_txt)) or (os.path.getsize(out_txt) == 0)
    with open(out_txt, "a") as f:
        if write_header:
            f.write(header)
        f.write(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--work_dir", type=str, default="datasets/weather_mtsci_chrono")
    ap.add_argument("--out_txt", type=str, default="runs/mtsci_weather_metrics.txt")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_mtsci_weather")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    
    dataset_dir, K = make_weather_mtsci_pickles(args.csv, args.work_dir, args.seq_len)

    
    config = {
        "model": {
            "timeemb": 64,
            "featureemb": 16,
            "is_unconditional": 0,
            "target_strategy": "block",
        },
        "diffusion": {
            "num_steps": 50,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule": "linear",
            "channels": 64,
            "diffusion_embedding_dim": 128,
            "nheads": 8,
            "layers": 4,
            "seqlen": int(args.seq_len),   
        },
        "train": {
            "lr": 1e-4,
            "lambda_cons": 1.0,
        },
    }

    
    train_loader = generate_train_dataloader(
        dataset_dir, seq_len=args.seq_len,
        missing_ratio=0.2, missing_pattern="block",
        batch_size=args.batch_size, mode="train",
    )
    test_loader_clean = generate_val_test_dataloader(
        dataset_dir, seq_len=args.seq_len,
        missing_ratio=0.0, missing_pattern="point",
        batch_size=args.batch_size, mode="test",
    )

    
    model = MTSCI(config=config, device=str(device), target_dim=K, seq_len=args.seq_len).to(device)
    optim = Adam(model.parameters(), lr=float(config["train"]["lr"]))

    
    model.train()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = tuple(x.to(device) for x in batch)
            optim.zero_grad()

            loss_noise, loss_cons = model(batch, is_train=1)
            loss = loss_noise + float(config["train"]["lambda_cons"]) * loss_cons

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {ep}: loss={loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_sum += loss.item()
            n_batches += 1

        print(f"Epoch {ep:03d} | loss={loss_sum/max(n_batches,1):.6f} | time={time.time()-t0:.1f}s")

    
    ratios = parse_ratios(args.eval_masked_ratios)
    os.makedirs(args.save_dir, exist_ok=True)

    header = "seed\tseq_len\tepochs\tbatch_size\tnsample\tlm\tr_masked\tMAE\tMSE\tRMSE\n"

    model.eval()
    for r in ratios:
        all_mae, all_mse, all_rmse = [], [], []
        saved = {"samples": [], "X_Tilde": [], "eval_mask": [], "gt_mask": [], "tp": []}

        for batch in test_loader_clean:
            batch = tuple(x.to(device) for x in batch)
            batch_m = override_test_batch_with_markov(batch, r_masked=r, lm=args.lm, device=device)

            # MTSCI evaluate(batch, n_samples) API
            samples, X_Tilde_BKL, eval_mask_BKL, X_Tilde_mask_BKL, tp = model.evaluate(batch_m, n_samples=args.nsample)

            mae, mse, rmse = masked_metrics_from_samples(samples, X_Tilde_BKL, eval_mask_BKL)
            all_mae.append(mae); all_mse.append(mse); all_rmse.append(rmse)

            if args.save_test_arrays:
                saved["samples"].append(samples.detach().cpu())
                saved["X_Tilde"].append(X_Tilde_BKL.detach().cpu())
                saved["eval_mask"].append(eval_mask_BKL.detach().cpu())
                saved["gt_mask"].append(X_Tilde_mask_BKL.detach().cpu())
                saved["tp"].append(tp.detach().cpu())

        mean_mae = float(np.mean(all_mae))
        mean_mse = float(np.mean(all_mse))
        mean_rmse = float(np.mean(all_rmse))

        print(f"test    r={r:.2f}  MAE={mean_mae:.6f}\tMSE={mean_mse:.6f}\tRMSE={mean_rmse:.6f}")

        row = (
            f"{args.seed}\t{args.seq_len}\t{args.epochs}\t{args.batch_size}\t{args.nsample}\t{args.lm}\t"
            f"{r:.2f}\t{mean_mae:.6f}\t{mean_mse:.6f}\t{mean_rmse:.6f}\n"
        )
        _append_row(args.out_txt, header, row)

        if args.save_test_arrays:
            samples = torch.cat(saved["samples"], dim=0).numpy()
            gt = torch.cat(saved["X_Tilde"], dim=0).numpy()
            evalmask = torch.cat(saved["eval_mask"], dim=0).numpy()

            np.save(os.path.join(args.save_dir, f"mtsci_weather_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_samples.npy"), samples)
            np.save(os.path.join(args.save_dir, f"mtsci_weather_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_gt.npy"), gt)
            np.save(os.path.join(args.save_dir, f"mtsci_weather_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_evalmask.npy"), evalmask)

    print("Done.")


if __name__ == "__main__":
    main()
