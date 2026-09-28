#!/usr/bin/env python3

import os, sys, time, argparse, random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

def _append_row(out_txt: str, header: str, row: str):
    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_header = (not os.path.exists(out_txt)) or (os.path.getsize(out_txt) == 0)
    with open(out_txt, "a") as f:
        if write_header:
            f.write(header)
        f.write(row)

# -------------------------
# Make repo imports robust
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from models.model import MTSCI  

from dataloader.dataloader import (
    generate_train_dataloader,
    generate_val_test_dataloader,
)

try:
    from utils import sample_mask 
except Exception:
    # fallback if utils is a package: utils/utils.py
    try:
        from utils.utils import sample_mask 
    except Exception:
        pass


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
def make_stock_mtsci_pickles(csv_path: str, out_dir: str) -> Tuple[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    df = df.select_dtypes(include=[np.number]).copy()
    df = df.dropna(axis=1, how="all")
    cols = list(df.columns)

    x = df.to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = np.nan

    n = x.shape[0]
    K = x.shape[1]

    n_train = int(np.floor(0.70 * n))
    n_val = int(np.floor(0.15 * n))

    train = x[:n_train]
    val = x[n_train : n_train + n_val]
    test = x[n_train + n_val :]

    # ---- compute scaler on TRAIN only (ignore NaNs) ----
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def standardize(arr):
        arr = arr.astype(np.float32)
        # fill NaNs with train mean, then z-score
        nanmask = ~np.isfinite(arr)
        if nanmask.any():
            feat_idx = np.where(nanmask)[1]
            arr[nanmask] = mean[feat_idx]
        arr = (arr - mean[None, :]) / std[None, :]
        return arr.astype(np.float32)

    train_z = standardize(train)
    val_z = standardize(val)
    test_z = standardize(test)

    import pickle as pk
    with open(os.path.join(out_dir, "train_set.pkl"), "wb") as f:
        pk.dump(train_z, f)
    with open(os.path.join(out_dir, "val_set.pkl"), "wb") as f:
        pk.dump(val_z, f)
    with open(os.path.join(out_dir, "test_set.pkl"), "wb") as f:
        pk.dump(test_z, f)


    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pk.dump((mean, std), f)

    print(f"cols: {cols}")
    print("raw min/max per col:", np.nanmin(x, axis=0), np.nanmax(x, axis=0))
    print("Saved MTSCI-formatted datasets + scaler.pkl")
    print("train/val/test shapes:", train.shape, val.shape, test.shape)
    return out_dir, K


# -------------------------
# Markov segment-wise keep-mask (1=keep/observed, 0=masked)
# -------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float) -> np.ndarray:
    """
    Returns keepmask (B,K,L) with segmenty Markov behavior.
    r_masked is fraction masked; keep ratio = 1 - r_masked.
    lm is mean segment length (roughly).
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # 0 -> 1 prob (masked -> keep)
    p_m = 1.0 / lm
    # 1 -> 0 prob (keep -> masked), chosen for desired stationary distribution
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]  # index by state: 0 masked, 1 keep

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < p[state]:
                    state = 1 - state
    return out


def override_test_batch_with_markov(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    r_masked: float,
    lm: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, mask, X_Tilde, gt_mask, indicating = batch  # shapes are (B,L,K) in dataloader

    # gt_mask is 1 where original observed (for stock it will be all 1s)
    # Create Markov keepmask in (B,K,L), then convert to indicating in (B,L,K)
    B, L, K = X_Tilde.shape
    keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_masked=r_masked, lm=lm)  # (B,K,L)
    keep_BLK = np.transpose(keep_BKL, (0, 2, 1))  # (B,L,K)
    keep_BLK = torch.from_numpy(keep_BLK).to(device=device)

    # indicating_mask is 1 where artificially masked
    indicating_new = (1.0 - keep_BLK) * gt_mask.to(device=device)

    # X = X_Tilde * (1 - indicating_mask)
    X_new = X_Tilde.to(device=device) * (1.0 - indicating_new)

    # mask = gt_mask * (1 - indicating_mask)
    mask_new = gt_mask.to(device=device) * (1.0 - indicating_new)

    # X_Tilde stays the ground truth window, gt_mask stays the same (original observed)
    return (X_new, mask_new, X_Tilde.to(device=device), gt_mask.to(device=device), indicating_new)


def masked_metrics_from_samples(samples_BnKL: torch.Tensor, X_true_BKL: torch.Tensor, eval_mask_BKL: torch.Tensor):
    # Take mean over samples
    pred = samples_BnKL.mean(dim=1)  # (B,K,L)

    # Masked positions are where eval_mask==1 (by MTSCI definition)
    m = eval_mask_BKL
    denom = torch.clamp(m.sum(), min=1.0)
    mae = (torch.abs(pred - X_true_BKL) * m).sum() / denom
    mse = (((pred - X_true_BKL) ** 2) * m).sum() / denom
    rmse = torch.sqrt(mse + 1e-12)
    return mae.item(), mse.item(), rmse.item()

def append_metrics_to_txt(
    out_txt,
    seed,
    seq_len,
    epochs,
    batch_size,
    nsample,
    lm,
    r,
    mae,
    mse,
    rmse,
):
    header = (
        "seed\tseq_len\tepochs\tbatch_size\tnsample\tlm\tr_masked\tMAE\tMSE\tRMSE\n"
    )

    file_exists = os.path.exists(out_txt)

    with open(out_txt, "a") as f:
        if not file_exists:
            f.write(header)

        f.write(
            f"{seed}\t{seq_len}\t{epochs}\t{batch_size}\t{nsample}\t{lm}\t"
            f"{r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.1,0.3,0.5,0.7")
    ap.add_argument("--lm", type=float, default=12.0, help="mean segment length for Markov masking")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--work_dir", type=str, default="datasets/stock_mtsci_rowsplit")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_mtsci_stock")
    ap.add_argument("--out_txt", type=str, default="mtsci_stock_results.txt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    
    dataset_dir, K = make_stock_mtsci_pickles(args.csv, args.work_dir)

    
    config = {
        "model": {
            "timeemb": 32,       # time embedding dim
            "featureemb": 32,    # feature embedding dim
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
            "lr": 1e-3,
            "lambda_cons": 1.0,  # intra-consistency loss weight
        },
    }

    
    train_loader = generate_train_dataloader(
        dataset_dir, seq_len=args.seq_len,
        missing_ratio=0.25, missing_pattern="block",
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
            loss.backward()
            optim.step()

            loss_sum += loss.item()
            n_batches += 1

        print(f"Epoch {ep:03d} | loss={loss_sum/max(n_batches,1):.6f} | time={time.time()-t0:.1f}s")

    
    model.eval()
    ratios = parse_ratios(args.eval_masked_ratios)

    os.makedirs(args.save_dir, exist_ok=True)

    for r in ratios:
        all_mae, all_mse, all_rmse = [], [], []
        saved = {"samples": [], "X_Tilde": [], "eval_mask": [], "X_Tilde_mask": [], "tp": []}

        for batch in test_loader_clean:
            batch = tuple(x.to(device) for x in batch)
            batch_m = override_test_batch_with_markov(batch, r_masked=r, lm=args.lm, device=device)

            
            samples, X_Tilde_BKL, eval_mask_BKL, X_Tilde_mask_BKL, tp = model.evaluate(batch_m, n_samples=args.nsample)

            mae, mse, rmse = masked_metrics_from_samples(samples, X_Tilde_BKL, eval_mask_BKL)
            all_mae.append(mae); all_mse.append(mse); all_rmse.append(rmse)

            if args.save_test_arrays:
                saved["samples"].append(samples.detach().cpu())
                saved["X_Tilde"].append(X_Tilde_BKL.detach().cpu())
                saved["eval_mask"].append(eval_mask_BKL.detach().cpu())
                saved["X_Tilde_mask"].append(X_Tilde_mask_BKL.detach().cpu())
                saved["tp"].append(tp.detach().cpu())

        mean_mae = float(np.mean(all_mae))
        mean_mse = float(np.mean(all_mse))
        mean_rmse = float(np.mean(all_rmse))

        line = (
            f"test\tr={r:.2f}\tMAE={mean_mae:.6f}\tMSE={mean_mse:.6f}\tRMSE={mean_rmse:.6f}"
        )
        print(line)

        # ---- append to txt ----
        header = "seed\tseq_len\tepochs\tbatch_size\tnsample\tlm\tr_masked\tMAE\tMSE\tRMSE\n"
        row = (
            f"{args.seed}\t{args.seq_len}\t{args.epochs}\t{args.batch_size}\t{args.nsample}\t{args.lm}\t"
            f"{r:.2f}\t{mean_mae:.6f}\t{mean_mse:.6f}\t{mean_rmse:.6f}\n"
        )
        _append_row(args.out_txt, header, row)

        if args.save_test_arrays:
            # concatenate along batch dimension
            samples = torch.cat(saved["samples"], dim=0)       # (N,ns,K,L)
            X_Tilde = torch.cat(saved["X_Tilde"], dim=0)       # (N,K,L)
            eval_mask = torch.cat(saved["eval_mask"], dim=0)   # (N,K,L)
            X_Tilde_mask = torch.cat(saved["X_Tilde_mask"], dim=0)
            tp = torch.cat(saved["tp"], dim=0)

            np.save(os.path.join(args.save_dir, f"mtsci_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_samples.npy"), samples.numpy())
            np.save(os.path.join(args.save_dir, f"mtsci_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_gt.npy"), X_Tilde.numpy())
            np.save(os.path.join(args.save_dir, f"mtsci_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_evalmask.npy"), eval_mask.numpy())
            np.save(os.path.join(args.save_dir, f"mtsci_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_gtmask.npy"), X_Tilde_mask.numpy())
            np.save(os.path.join(args.save_dir, f"mtsci_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_tp.npy"), tp.numpy())

    print("Done.")


if __name__ == "__main__":
    main()
