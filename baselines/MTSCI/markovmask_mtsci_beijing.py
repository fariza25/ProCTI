#!/usr/bin/env python3

import os
import sys
import time
import json
import argparse
import random
import pickle as pk
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import Adam

# Repo root on path
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


def safe_batch_size(requested: int, n: int) -> int:
    if n <= 0:
        return 1
    return max(1, min(int(requested), int(n)))


# -------------------------
# Shared Beijing loading
# -------------------------
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

    evalmask = obj[key].astype(np.float32)   # (N,L,K)
    keepmask = 1.0 - evalmask
    return keepmask


# -------------------------
# Build MTSCI dataset pickles from shared windows
# -------------------------
def make_beijing_mtsci_pickles(shared_data_dir: str, out_dir: str, seq_len: int):
    os.makedirs(out_dir, exist_ok=True)

    train_w, val_w, test_w = load_shared_windows(shared_data_dir)
    Ntr, L, K = train_w.shape
    Nva = val_w.shape[0]
    Nte = test_w.shape[0]

    if L != seq_len:
        raise ValueError(f"seq_len mismatch: windows have L={L}, requested seq_len={seq_len}")

    train_tk = train_w.reshape(-1, K).astype(np.float32)
    val_tk = val_w.reshape(-1, K).astype(np.float32)
    test_tk = test_w.reshape(-1, K).astype(np.float32)

    # Scaler is already applied upstream, but we keep MTSCI artifact structure
    mu = np.nanmean(train_tk, axis=0).astype(np.float32)
    sd = np.nanstd(train_tk, axis=0).astype(np.float32)
    sd = np.maximum(sd, 1e-6).astype(np.float32)

    pk.dump(train_tk, open(os.path.join(out_dir, "train_set.pkl"), "wb"))
    pk.dump(val_tk, open(os.path.join(out_dir, "val_set.pkl"), "wb"))
    pk.dump(test_tk, open(os.path.join(out_dir, "test_set.pkl"), "wb"))
    pk.dump((mu, sd), open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    meta = {
        "seq_len": int(seq_len),
        "n_windows_train": int(Ntr),
        "n_windows_val": int(Nva),
        "n_windows_test": int(Nte),
        "K": int(K),
        "shared_data_dir": shared_data_dir,
    }
    pk.dump(meta, open(os.path.join(out_dir, "metadata.pkl"), "wb"))

    print(
        f"[data] Beijing windows train/val/test={Ntr}/{Nva}/{Nte}  "
        f"K={K}  L={L}"
    )
    return out_dir, K, Ntr, Nva, Nte


# -------------------------
# Random Markov keepmask fallback
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
# Override clean test batch with shared keepmask or Markov keepmask
# -------------------------
def override_test_batch_with_keepmask(batch, keep_blK: torch.Tensor, device: torch.device):
    """
    MTSCI test loader yields:
      (X, mask, X_Tilde, gt_mask, indicating_mask)

    We replace the evaluation masking only.
    keep_blK is (B,L,K), 1=observed, 0=masked.
    """
    X, mask, X_Tilde, gt_mask, indicating = batch

    X_Tilde = X_Tilde.to(device)
    gt_mask = gt_mask.to(device)
    keep_blK = keep_blK.to(device).float()

    indicating_new = (1.0 - keep_blK) * gt_mask
    X_new = X_Tilde * (1.0 - indicating_new)
    mask_new = gt_mask * (1.0 - indicating_new)

    return (X_new, mask_new, X_Tilde, gt_mask, indicating_new)


def masked_metrics_from_samples(samples_BnKL: torch.Tensor, X_true_BKL: torch.Tensor, eval_mask_BKL: torch.Tensor):
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


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", type=str, required=True,
                    help="Directory containing shared Beijing windows and maskbanks")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--work_dir", type=str, default="datasets/beijing_mtsci_shared")
    ap.add_argument("--out_txt", type=str, default="runs/mtsci_beijing_metrics.txt")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_mtsci_beijing")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    
    dataset_dir, K, Ntr, Nva, Nte = make_beijing_mtsci_pickles(
        args.shared_data_dir, args.work_dir, args.seq_len
    )

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
        dataset_dir,
        seq_len=args.seq_len,
        missing_ratio=0.2,
        missing_pattern="block",
        batch_size=args.batch_size,
        mode="train",
    )
    val_loader_clean = generate_val_test_dataloader(
        dataset_dir,
        seq_len=args.seq_len,
        missing_ratio=0.0,
        missing_pattern="point",
        batch_size=args.batch_size,
        mode="val",
    )
    test_loader_clean = generate_val_test_dataloader(
        dataset_dir,
        seq_len=args.seq_len,
        missing_ratio=0.0,
        missing_pattern="point",
        batch_size=args.batch_size,
        mode="test",
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

        if ep == 1 or ep % 10 == 0:
            print(f"Epoch {ep:03d} | loss={loss_sum/max(n_batches,1):.6f} | time={time.time()-t0:.1f}s")

   
    ratios = parse_ratios(args.eval_masked_ratios)
    os.makedirs(args.save_dir, exist_ok=True)

    header = "split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n"

    def eval_one_loader(split_name, loader_clean, n_windows):
        model.eval()
        rows = []

        for r in ratios:
            all_mae, all_mse, all_rmse = [], [], []
            all_r_obs = []
            saved = {"samples": [], "X_Tilde": [], "eval_mask": [], "cond_mask": []}

            keep_nlk = None
            if args.use_shared_evalmask:
                keep_nlk = load_shared_keepmask_nlk(
                    args.shared_evalmask_dir, split=split_name, r_m=r, seed=args.seed
                )
                expected = (n_windows, args.seq_len, K)
                if keep_nlk.shape != expected:
                    raise ValueError(
                        f"Shared keepmask shape mismatch for split={split_name}, r={r:.2f}: "
                        f"keep={keep_nlk.shape}, expected={expected}"
                    )

            offset = 0
            for batch in loader_clean:
                batch = tuple(x.to(device) for x in batch)
                X, mask, X_Tilde, gt_mask, indicating = batch
                B, L, KK = X_Tilde.shape

                if keep_nlk is not None:
                    keep_blk = torch.from_numpy(keep_nlk[offset:offset+B]).to(device).float()  # (B,L,K)
                    offset += B
                else:
                    keep_bkl = markov_keep_mask_from_masked_ratio(B, K, L, r_masked=r, lm=args.lm)
                    keep_blk = torch.from_numpy(np.transpose(keep_bkl, (0, 2, 1))).to(device).float()

                batch_m = override_test_batch_with_keepmask(batch, keep_blk, device=device)

                samples, X_Tilde_BKL, eval_mask_BKL, X_Tilde_mask_BKL, tp = model.evaluate(
                    batch_m, n_samples=args.nsample
                )

                mae, mse, rmse = masked_metrics_from_samples(samples, X_Tilde_BKL, eval_mask_BKL)
                all_mae.append(mae)
                all_mse.append(mse)
                all_rmse.append(rmse)
                all_r_obs.append(float(keep_blk.mean().item()))

                if args.save_test_arrays and split_name == "test":
                    saved["samples"].append(samples.detach().cpu())
                    saved["X_Tilde"].append(X_Tilde_BKL.detach().cpu())
                    saved["eval_mask"].append(eval_mask_BKL.detach().cpu())
                    saved["cond_mask"].append(keep_blk.permute(0, 2, 1).detach().cpu())  # (B,K,L)

            mean_mae = float(np.mean(all_mae))
            mean_mse = float(np.mean(all_mse))
            mean_rmse = float(np.mean(all_rmse))
            mean_r_obs = float(np.mean(all_r_obs))

            print(f"{split_name}    r={r:.2f}  MAE={mean_mae:.6f}\tMSE={mean_mse:.6f}\tRMSE={mean_rmse:.6f}\tr_obs={mean_r_obs:.2f}")

            row = f"{split_name}\t{r:.2f}\t{mean_r_obs:.2f}\t{mean_mae:.6f}\t{mean_mse:.6f}\t{mean_rmse:.6f}\n"
            _append_row(args.out_txt, header, row)
            rows.append((r, mean_r_obs, mean_mae, mean_mse, mean_rmse))

            if args.save_test_arrays and split_name == "test":
                samples = torch.cat(saved["samples"], dim=0).numpy()     # (N,ns,K,L)
                gt = torch.cat(saved["X_Tilde"], dim=0).numpy()          # (N,K,L)
                evalmask = torch.cat(saved["eval_mask"], dim=0).numpy()  # (N,K,L)
                condmask = torch.cat(saved["cond_mask"], dim=0).numpy()  # (N,K,L)

                np.save(os.path.join(args.save_dir, f"mtsci_beijing_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_samples.npy"), samples)
                np.save(os.path.join(args.save_dir, f"mtsci_beijing_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_gt.npy"), gt)
                np.save(os.path.join(args.save_dir, f"mtsci_beijing_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_evalmask.npy"), evalmask)
                np.save(os.path.join(args.save_dir, f"mtsci_beijing_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_condmask.npy"), condmask)

        return rows

    eval_one_loader("val", val_loader_clean, Nva)
    eval_one_loader("test", test_loader_clean, Nte)

    if args.save_test_arrays:
        meta_out = {
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_data_dir": args.shared_data_dir,
            "lm": float(args.lm),
            "nsample": int(args.nsample),
        }
        with open(os.path.join(args.save_dir, f"metadata_seed{args.seed}.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
