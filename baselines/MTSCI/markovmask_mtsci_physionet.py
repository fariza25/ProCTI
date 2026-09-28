#!/usr/bin/env python3

import os, sys, time, argparse, random
from typing import List, Tuple, Set

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


def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    pid_col = "Patient_ID"
    if pid_col not in df.columns:
        raise ValueError("Patient_ID column not found in CSV.")

    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce")
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)

    # harmless aliases used in some pipelines
    for _alias in ["patient_id", "PatientID", "patient", "RecordID", "id", "UID", "stay_id"]:
        if _alias not in df.columns:
            df[_alias] = df[pid_col]
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
    seq_len: int
) -> np.ndarray:
    patient_ids = set(int(x) for x in patient_ids)
    windows = []

    for pid, g in df.groupby("Patient_ID", sort=False):
        pid = int(pid)
        if pid not in patient_ids:
            continue

        g = g.sort_values(time_col, kind="mergesort")
        X = g[feat_cols].astype(np.float32).ffill().bfill()
        arr = X.to_numpy()  # (T,K_raw)

        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue

        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))

    if not windows:
        raise ValueError("No windows formed for this split (check seq_len or patient IDs).")
    return np.concatenate(windows, axis=0).astype(np.float32)  # (N,L,K_raw)


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    """
    - compute TRAIN nanmean/nanstd
    - drop features that are non-finite or ~zero-std in TRAIN
    - fill remaining NaNs using TRAIN mean
    - z-score with TRAIN mean/std
    """
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All features are non-finite / zero-std after TRAIN stats. Check CSV.")

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


def make_physionet_mtsci_pickles(csv_path: str, out_dir: str, seq_len: int, seed: int) -> Tuple[str, int]:
    os.makedirs(out_dir, exist_ok=True)

    df = load_physionet_df(csv_path)
    time_col = get_time_col(df)
    feat_cols = get_feature_cols(df)

    train_ids, val_ids, test_ids = split_patients(df, seed=seed)

    train_raw = make_windows_for_patients(df, train_ids, feat_cols, time_col, seq_len)  # (N,L,K_raw)
    val_raw   = make_windows_for_patients(df, val_ids,   feat_cols, time_col, seq_len)
    test_raw  = make_windows_for_patients(df, test_ids,  feat_cols, time_col, seq_len)

    train_w, val_w, test_w, mu, sd, valid = standardize_by_train(train_raw, val_raw, test_raw)

    K = int(train_w.shape[-1])
    print(f"[data] patients train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)} "
          f" windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K}  L={seq_len}")


    pk.dump(train_w.reshape(-1, K), open(os.path.join(out_dir, "train_set.pkl"), "wb"))
    pk.dump(val_w.reshape(-1, K),   open(os.path.join(out_dir, "val_set.pkl"), "wb"))
    pk.dump(test_w.reshape(-1, K),  open(os.path.join(out_dir, "test_set.pkl"), "wb"))
    pk.dump((mu.astype(np.float32), sd.astype(np.float32)), open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    pk.dump(
        {
            "seed": int(seed),
            "seq_len": int(seq_len),
            "time_col": str(time_col),
            "feat_cols_raw": feat_cols,
            "valid_feature_mask": valid.astype(bool),
            "patients_train": sorted(list(train_ids)),
            "patients_val": sorted(list(val_ids)),
            "patients_test": sorted(list(test_ids)),
            "n_windows_train": int(len(train_w)),
            "n_windows_val": int(len(val_w)),
            "n_windows_test": int(len(test_w)),
        },
        open(os.path.join(out_dir, "metadata.pkl"), "wb")
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


def override_test_batch_with_markov(batch, r_masked: float, lm: float, device: torch.device):

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
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--work_dir", type=str, default="datasets/physionet_mtsci_patientwise")
    ap.add_argument("--out_txt", type=str, default="runs/mtsci_physionet_metrics.txt")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_mtsci_physionet")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

   
    dataset_dir, K = make_physionet_mtsci_pickles(args.csv, args.work_dir, args.seq_len, args.seed)

    
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

            np.save(os.path.join(args.save_dir, f"mtsci_physionet_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_samples.npy"), samples)
            np.save(os.path.join(args.save_dir, f"mtsci_physionet_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_gt.npy"), gt)
            np.save(os.path.join(args.save_dir, f"mtsci_physionet_test_r{r:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}_evalmask.npy"), evalmask)

    print("Done.")


if __name__ == "__main__":
    main()
