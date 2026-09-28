#!/usr/bin/env python3

import os, sys, time, argparse, random, re
from typing import List, Tuple, Dict, Set

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
# GAIT dataset prep 
# -------------------------
USER_RE = re.compile(r"_ID(\d+)_", re.IGNORECASE)


def list_gait_files(data_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".csv"):
            files.append(os.path.join(data_dir, fn))
    if not files:
        raise ValueError(f"No .csv files found in {data_dir}")
    return sorted(files)


def user_id_from_name(path: str) -> str:
    m = USER_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse user ID from filename: {os.path.basename(path)}")
    return m.group(1)


def read_gait_csv(path: str) -> np.ndarray:
    # automatic-ou-gaitdata: csv with two header rows
    df = pd.read_csv(path, skiprows=2, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad array shape from {path}: {arr.shape}")
    return arr


def windows_from_files(file_list: List[str], seq_len: int) -> np.ndarray:
    windows = []
    for p in file_list:
        arr = read_gait_csv(p)  # (T,K)
        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))
    if not windows:
        raise ValueError("No windows formed (seq_len may be too large or files too short).")
    return np.concatenate(windows, axis=0).astype(np.float32)  # (N,L,K)


def split_users(files: List[str], seed: int, train_ratio=0.7, val_ratio=0.15):
    by_user: Dict[str, List[str]] = {}
    for p in files:
        uid = user_id_from_name(p)
        by_user.setdefault(uid, []).append(p)

    users = sorted(by_user.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_u = users[:n_train]
    val_u = users[n_train:n_train + n_val]
    test_u = users[n_train + n_val:]

    def gather(uids):
        out = []
        for u in uids:
            out.extend(by_user[u])
        return out

    return gather(train_u), gather(val_u), gather(test_u), (train_u, val_u, test_u)


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = np.isnan(x)
        if nanmask.any():
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd


def make_gait_mtsci_pickles(data_dir: str, out_dir: str, seq_len: int, seed: int) -> Tuple[str, int]:
    os.makedirs(out_dir, exist_ok=True)

    files = list_gait_files(data_dir)
    tr_files, va_files, te_files, (tr_users, va_users, te_users) = split_users(files, seed=seed)

    train_raw = windows_from_files(tr_files, seq_len)
    val_raw   = windows_from_files(va_files, seq_len)
    test_raw  = windows_from_files(te_files, seq_len)

    train_w, val_w, test_w, mu, sd = standardize_by_train(train_raw, val_raw, test_raw)

    if len(train_w) == 0 or len(val_w) == 0 or len(test_w) == 0:
        raise ValueError("One of the splits has 0 windows. Check seq_len or dataset length.")

    K = int(train_w.shape[-1])
    print(f"[data] #users train/val/test={len(tr_users)}/{len(va_users)}/{len(te_users)}  "
          f"#windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={seq_len}")

    # MTSCI dataloader expects (T,K) arrays in pickles, not (N,L,K)
    pk.dump(train_w.reshape(-1, K), open(os.path.join(out_dir, "train_set.pkl"), "wb"))
    pk.dump(val_w.reshape(-1, K),   open(os.path.join(out_dir, "val_set.pkl"), "wb"))
    pk.dump(test_w.reshape(-1, K),  open(os.path.join(out_dir, "test_set.pkl"), "wb"))
    pk.dump((mu.astype(np.float32), sd.astype(np.float32)), open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    # helpful metadata
    pk.dump(
        {
            "seed": int(seed),
            "seq_len": int(seq_len),
            "users_train": tr_users,
            "users_val": va_users,
            "users_test": te_users,
            "n_windows_train": int(len(train_w)),
            "n_windows_val": int(len(val_w)),
            "n_windows_test": int(len(test_w)),
            "K": int(K),
        },
        open(os.path.join(out_dir, "metadata.pkl"), "wb"),
    )

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
    # In MTSCI dataloader, test batches behave like: (X, mask, X_Tilde, gt_mask, indicating_mask)
    X, mask, X_Tilde, gt_mask, indicating = batch
    B, L, K = X_Tilde.shape

    keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_masked=r_masked, lm=lm)  # (B,K,L)
    keep_BLK = np.transpose(keep_BKL, (0, 2, 1))  # (B,L,K)
    keep_BLK = torch.from_numpy(keep_BLK).to(device=device)

    indicating_new = (1.0 - keep_BLK) * gt_mask.to(device=device)
    X_new = X_Tilde.to(device=device) * (1.0 - indicating_new)
    mask_new = gt_mask.to(device=device) * (1.0 - indicating_new)

    return (X_new, mask_new, X_Tilde.to(device=device), gt_mask.to(device=device), indicating_new)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="automatic-ou-gaitdata directory containing per-trial CSVs")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--work_dir", type=str, default="datasets/gait_mtsci_userwise")
    ap.add_argument("--out_txt", type=str, default="runs/mtsci_gait_metrics.txt")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_mtsci_gait")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    
    dataset_dir, K = make_gait_mtsci_pickles(args.data_dir, args.work_dir, args.seq_len, args.seed)

    
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
        saved = {"samples": [], "X_Tilde": [], "eval_mask": [], "tp": []}

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
            samples = torch.cat(saved["samples"], dim=0).numpy()  # (N,ns,K,L)
            gt = torch.cat(saved["X_Tilde"], dim=0).numpy()       # (N,K,L)
            evalmask = torch.cat(saved["eval_mask"], dim=0).numpy()

            np.save(os.path.join(args.save_dir, f"mtsci_gait_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_samples.npy"), samples)
            np.save(os.path.join(args.save_dir, f"mtsci_gait_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_gt.npy"), gt)
            np.save(os.path.join(args.save_dir, f"mtsci_gait_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_evalmask.npy"), evalmask)

    print("Done.")


if __name__ == "__main__":
    main()
