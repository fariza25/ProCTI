import os
import re
import time
import random
import argparse
from types import SimpleNamespace
from typing import List, Tuple, Dict, Optional

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
# GAIT data pipeline
# -----------------------------
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
        raise ValueError(f"Could not parse user id from filename: {path}")
    return str(int(m.group(1)))


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


def read_gait_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=None, skiprows=2)
    arr = df.to_numpy(dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr.astype(np.float32)


def windows_from_files(file_list: List[str], seq_len: int) -> np.ndarray:
    windows = []
    for p in file_list:
        arr = read_gait_csv(p)
        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))
    if not windows:
        raise ValueError("No windows formed (seq_len may be too large or files too short).")
    return np.concatenate(windows, axis=0).astype(np.float32)  # (N,L,K)


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


def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int, invert: bool) -> np.ndarray:
    path = os.path.join(mask_dir, f"gait_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must have shape (N,L,K), got {keep.shape}")
    if invert or os.environ.get("GAIT_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep
    return keep

def align_shared_keepmask_to_dataset(
    keep_NLK: np.ndarray,
    n_data: int,
    split_name: str,
    r_m: float,
    mode: str = "trim_or_repeat",
) -> np.ndarray:
    """
    Align shared keepmask length to the number of windows in the current split.

    Exact matching is preferred for final fair-comparison runs. If a previously
    generated maskbank has a different number of windows, trim_or_repeat keeps
    this script runnable by trimming extra masks or repeating masks cyclically.
    """
    n_mask = int(keep_NLK.shape[0])
    if n_mask == n_data:
        return keep_NLK.astype(np.float32)

    msg = (
        f"[warning] Shared maskbank/window mismatch for split={split_name}, "
        f"r={r_m:.2f}: dataset has {n_data} windows but maskbank has {n_mask} masks."
    )

    if mode == "error":
        raise ValueError(
            msg + " Regenerate the gait maskbank using the exact same data_dir, "
            "seq_len, seed, user split logic, and windowing order as this script."
        )

    if mode != "trim_or_repeat":
        raise ValueError(f"Unknown shared mask alignment mode: {mode}")

    if n_mask <= 0:
        raise ValueError(f"Shared maskbank for split={split_name}, r={r_m:.2f} is empty.")

    if n_mask > n_data:
        print(msg + " Trimming maskbank to dataset length.")
        return keep_NLK[:n_data].astype(np.float32)

    reps = int(np.ceil(n_data / n_mask))
    print(msg + " Repeating masks cyclically to cover all windows.")
    return np.tile(keep_NLK, (reps, 1, 1))[:n_data].astype(np.float32)


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
               shared_mismatch_mode: str):
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
            keep_NLK = load_shared_keepmask(shared_dir, split_name, r_m, L, seed, invert_shared)
            keep_NLK = align_shared_keepmask_to_dataset(
                keep_NLK,
                n_data=len(loader.dataset),
                split_name=split_name,
                r_m=r_m,
                mode=shared_mismatch_mode,
            )

        for xb in loader:
            xb = xb.to(device).float()
            B = xb.shape[0]

            if keep_NLK is not None:
                keep_slice = keep_NLK[offset:offset + B]
                if keep_slice.shape[0] != B:
                    raise RuntimeError(
                        f"Internal mask slicing error for split={split_name}, r={r_m:.2f}: "
                        f"requested B={B}, got {keep_slice.shape[0]}, offset={offset}, "
                        f"maskbank size={keep_NLK.shape[0]}."
                    )
                keep_BLK = torch.from_numpy(keep_slice).to(device).float()
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
    ap.add_argument("--data_dir", type=str, required=True, help="Folder with gait CSV files (automatic-ou-gaitdata)")
    ap.add_argument("--seq_len", type=int, default=64)

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
    ap.add_argument(
        "--shared_mismatch_mode",
        type=str,
        default="trim_or_repeat",
        choices=["trim_or_repeat", "error"],
        help=(
            "What to do if the shared maskbank length differs from the current split. "
            "Use 'error' for strict final fair-comparison runs; use 'trim_or_repeat' "
            "to keep the script runnable."
        ),
    )

    # output
    ap.add_argument("--out_txt", type=str, default="itransformer_gait_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    ratios = parse_ratios(args.eval_masked_ratios)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        raise ValueError("--use_shared_evalmask requires --shared_evalmask_dir")

    # ---- split by users, then window within each split
    files = list_gait_files(args.data_dir)
    train_files, val_files, test_files, (train_u, val_u, test_u) = split_users(files, seed=args.seed)
    print(f"[split] users train/val/test = {len(train_u)}/{len(val_u)}/{len(test_u)}")
    print(f"[split] files train/val/test = {len(train_files)}/{len(val_files)}/{len(test_files)}")

    train_w = windows_from_files(train_files, args.seq_len)
    val_w = windows_from_files(val_files, args.seq_len)
    test_w = windows_from_files(test_files, args.seq_len)

    train_w, val_w, test_w, mu, sd = standardize_by_train(train_w, val_w, test_w)

    K = int(train_w.shape[-1])
    L = int(args.seq_len)
    print(f"[data] windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  L={L} K={K}")

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
                shared_mismatch_mode=args.shared_mismatch_mode
            )
            for (sp, r_m, r_obs, mae, mse, rmse) in rows:
                f.write(f"{sp}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{sp}] r_masked={r_m:.2f} r_obs={r_obs:.2f}  MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()

