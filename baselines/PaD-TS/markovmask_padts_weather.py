#!/usr/bin/env python3

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s: str):
    return [float(x) for x in s.split(",")]



def markov_keep_mask(B, K, L, r_masked, lm, device):
    r_keep = 1 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1 - r_keep) / max(r_keep, 1e-8)

    mask = np.ones((B, K, L), dtype=np.float32)

    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                mask[b, k, t] = state
                if np.random.rand() < (p_m if state == 0 else p_u):
                    state = 1 - state

    return torch.from_numpy(mask).to(device)


# -------------------------
# dataset utils
# -------------------------
class WindowDataset(Dataset):
    def __init__(self, windows_nlk: np.ndarray):
        self.x = windows_nlk.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


def load_weather_windows(csv: str, seq_len: int):
    df = pd.read_csv(csv)

    
    tcol = "date" if "date" in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol)

    num_cols = df.select_dtypes(include=[np.number]).columns
    X = df[num_cols].values.astype(np.float32)  # (T,K)

    T, K = X.shape
    N = T // seq_len
    X = X[: N * seq_len]
    windows = X.reshape(N, seq_len, K)  # (N,L,K)
    return windows


def chrono_split_701515(windows):
    N = len(windows)
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)
    train = windows[:n_train]
    val = windows[n_train : n_train + n_val]
    test = windows[n_train + n_val :]
    return train, val, test


def standardize_like_brits(train, val, test):
    # standardize using train statistics; fill NaNs with 0 after scaling
    mu = np.nanmean(train.reshape(-1, train.shape[-1]), axis=0)
    sd = np.nanstd(train.reshape(-1, train.shape[-1]), axis=0)
    sd = np.maximum(sd, 1e-6)

    def z(x):
        x = (x - mu) / sd
        x[np.isnan(x)] = 0.0
        return x

    return z(train), z(val), z(test)


# -------------------------
# shared maskbank loader (optional)
# -------------------------
def load_maskbank(maskbank_path: str):

    obj = np.load(maskbank_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item()
    return obj


def get_keepmask_from_bank(bank, split: str, ratio: float, idx: int, device):
    """
    Returns keepmask shaped (K,L) or (1,K,L) for a single window idx.
    """
    # Try several common structures
    key1 = (split, float(ratio))
    key2 = f"{split}_{ratio:.2f}"
    key3 = f"{split}_r{ratio:.2f}"

    arr = None
    if isinstance(bank, dict):
        if key1 in bank:
            arr = bank[key1]
        elif key2 in bank:
            arr = bank[key2]
        elif key3 in bank:
            arr = bank[key3]
        elif split in bank:
            
            sub = bank[split]
            if isinstance(sub, dict):
                if float(ratio) in sub:
                    arr = sub[float(ratio)]
                elif f"{ratio:.2f}" in sub:
                    arr = sub[f"{ratio:.2f}"]
    else:
        arr = None

    if arr is None:
        raise KeyError(
            f"Could not find masks for split={split}, ratio={ratio} in maskbank. "
            f"Show me np.load(...).item().keys() and I'll align it."
        )

    
    m = arr[idx]
    m = np.asarray(m)

    if m.ndim == 2:
        
        pass
    elif m.ndim == 3:
        
        m = m[0]
    else:
        raise ValueError(f"Unexpected mask shape from bank: {m.shape}")

    
    if m.shape[0] != m.shape[-1] and m.shape[0] < m.shape[1]:
        
        keep_kl = m
    else:
        keep_kl = m  

    keep = torch.from_numpy(keep_kl.astype(np.float32)).to(device)
    return keep  # (K,L)


# -------------------------
# PaD-TS imports
# -------------------------
def import_padts(padts_root: str):
    sys.path.insert(0, padts_root)

    from resample import UniformSampler, Batch_Same_Sampler
    from Model import PaD_TS
    from diffmodel_init import create_gaussian_diffusion
    from training import Trainer

    return PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler


# -------------------------
# diffusion imputation (projection / RePaint-style)
# -------------------------
@torch.no_grad()
def diffusion_impute_project(
    model,
    diffusion,
    x0_blK: torch.Tensor,       # (B,L,K) normalized
    keep_blK: torch.Tensor,     # (B,L,K) 1=observed,0=missing
):
    """
    Unconditional diffusion sampling, but at each reverse step overwrite observed
    entries with the corresponding noisy observation at that timestep.

    Returns xhat0_blK: (B,L,K)
    """
    device = x0_blK.device
    B, L, K = x0_blK.shape

    # start from pure noise at time T
    x_t = torch.randn(B, L, K, device=device)

    eps_obs = torch.randn_like(x0_blK)

    T = diffusion.num_timesteps

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        out = diffusion.p_sample(model, x_t, t_tensor, clip_denoised=True)
        x_prev = out["sample"]  # (B,L,K)

        if t > 0:
            t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
            x_obs_prev = diffusion.q_sample(x0_blK, t_prev, noise=eps_obs)
        else:
            x_obs_prev = x0_blK

        x_t = keep_blK * x_obs_prev + (1.0 - keep_blK) * x_prev

    return x_t


def compute_metrics(diff, evalmask):
    denom = evalmask.sum().item()
    denom = max(denom, 1.0)
    mae = diff.abs().sum().item() / denom
    mse = (diff ** 2).sum().item() / denom
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


@torch.no_grad()
def eval_split(
    model,
    diffusion,
    data_windows_nlk: np.ndarray,
    ratios,
    lm,
    device,
    batch_size,
    use_maskbank=False,
    maskbank=None,
    split_name="val",
):
    """
    Evaluate on a numpy array windows (N,L,K).
    """
    ds = WindowDataset(data_windows_nlk)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    results = []

    for r in ratios:
        sum_abs = 0.0
        sum_sq = 0.0
        sum_den = 0.0

        global_idx = 0

        for x in loader:
            x = x.to(device)  # (B,L,K)
            B, L, K = x.shape

            if use_maskbank:
                # build keepmask per-sample from bank
                keep_list = []
                for b in range(B):
                    keep_kl = get_keepmask_from_bank(maskbank, split_name, r, global_idx + b, device)  # (K,L)
                    keep_list.append(keep_kl.permute(1, 0))  # -> (L,K)
                keep = torch.stack(keep_list, dim=0)  # (B,L,K)
            else:
                keep = markov_keep_mask(B, K, L, r, lm, device).permute(0, 2, 1)  # (B,L,K)

            # impute with projection sampling
            xhat = diffusion_impute_project(model, diffusion, x, keep)

            evalmask = 1.0 - keep
            diff = (xhat - x) * evalmask

            sum_abs += diff.abs().sum().item()
            sum_sq += (diff ** 2).sum().item()
            sum_den += evalmask.sum().item()

            global_idx += B

        denom = max(sum_den, 1.0)
        mae = sum_abs / denom
        mse = sum_sq / denom
        rmse = float(np.sqrt(mse))

        results.append((r, mae, mse, rmse))

    return results


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--padts_root", required=True, help="Path to cloned wmd3i/PaD-TS repo")
    ap.add_argument("--csv", required=True, help="Weather CSV (with date column like BRITS weather)")

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")

    # PaD-TS-ish hyperparams
    ap.add_argument("--train_steps", type=int, default=5000, help="PaD-TS trainer steps (not epochs)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--schedule_sampler", type=str, default="batch", choices=["batch", "uniform"])
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--mmd_alpha", type=float, default=0.0005)

    # Model config (keep defaults similar to energy/stock configs but adjustable)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--n_encoder", type=int, default=1)
    ap.add_argument("--n_decoder", type=int, default=3)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--feature_last", action="store_true", default=True)

    # Diffusion config
    ap.add_argument("--diffusion_steps", type=int, default=250)
    ap.add_argument("--noise_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--loss", type=str, default="MSE_MMD", choices=["MSE", "MSE_MMD"])
    ap.add_argument("--predict_xstart", action="store_true", default=True)
    ap.add_argument("--rescale_timesteps", action="store_true", default=False)

    # evaluation
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_ratios", default="0.10,0.30,0.50,0.70")

    # optional shared maskbank
    ap.add_argument("--use_maskbank", action="store_true")
    ap.add_argument("--maskbank_path", default="")

    ap.add_argument("--out_txt", default="padts_weather_metrics.txt")
    ap.add_argument("--save_dir", default="./OUTPUT/padts_weather/")

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load and split data 
    windows = load_weather_windows(args.csv, args.seq_len)
    train, val, test = chrono_split_701515(windows)
    train, val, test = standardize_like_brits(train, val, test)

    N, L, K = train.shape[0], train.shape[1], train.shape[2]
    print(f"[Weather] windows={len(windows)} train/val/test={len(train)}/{len(val)}/{len(test)}  L={L} K={K}")

    # PaD-TS imports
    PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler = import_padts(args.padts_root)

    # Model + diffusion
    model = PaD_TS(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        feature_last=args.feature_last,
        mlp_ratio=args.mlp_ratio,
        input_shape=(args.seq_len, K),
    )

    diffusion = create_gaussian_diffusion(
        predict_xstart=args.predict_xstart,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        loss=args.loss,
        rescale_timesteps=args.rescale_timesteps,
    )

    # Dataloader for PaD-TS Trainer
    train_loader = DataLoader(
        WindowDataset(train),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    # schedule sampler
    if args.schedule_sampler == "batch":
        schedule_sampler = Batch_Same_Sampler(diffusion)
    else:
        schedule_sampler = UniformSampler(diffusion)

    # Trainer (faithful)
    os.makedirs(args.save_dir, exist_ok=True)
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.train_steps,
        log_interval=10,
        save_interval=args.save_interval,
        save_dir=args.save_dir.rstrip("/") + "/",
        schedule_sampler=schedule_sampler,
        mmd_alpha=args.mmd_alpha,
    )

    print("======Training======")
    trainer.train()
    print("======Training done======")

    ratios = parse_ratios(args.eval_ratios)

    maskbank = None
    if args.use_maskbank:
        if not args.maskbank_path:
            raise ValueError("--use_maskbank requires --maskbank_path")
        maskbank = load_maskbank(args.maskbank_path)
        print(f"[maskbank] loaded: {args.maskbank_path}")

    print("======Evaluation (Markov segment-wise)======")
    model.eval()
    model.to(device)

    val_res = eval_split(
        model, diffusion, val, ratios, args.lm, device, args.batch,
        use_maskbank=args.use_maskbank, maskbank=maskbank, split_name="val",
    )
    test_res = eval_split(
        model, diffusion, test, ratios, args.lm, device, args.batch,
        use_maskbank=args.use_maskbank, maskbank=maskbank, split_name="test",
    )

    # write metrics
    header_needed = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if header_needed:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for r, mae, mse, rmse in val_res:
            f.write(f"val\t{r:.2f}\t{1-r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        for r, mae, mse, rmse in test_res:
            f.write(f"test\t{r:.2f}\t{1-r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[saved] {args.out_txt}")


if __name__ == "__main__":
    main()
