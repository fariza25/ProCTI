#!/usr/bin/env python3


import os, argparse, random
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from csdi_main_model import CSDI_base
from csdi_utils import train as csdi_train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Markov keep-mask from MASKED ratio
# -----------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device):
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    p_u = p_m * r_keep / (1.0 - r_keep)
    p = [p_m, p_u]  # state 0=masked, 1=unmasked

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
# Weather preprocessing
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
        raise ValueError("No numeric columns found.")
    X = df[num_cols].astype(np.float32).to_numpy()  # (T,K)

    T, K = X.shape
    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}")
    X = X[: n_win * seq_len].reshape(n_win, seq_len, K)
    return X.astype(np.float32), num_cols, time_col


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
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd), sd, 1.0)
    sd = np.maximum(sd, eps)

    def fill_and_z(x):
        x = x.copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd


# -----------------------------
# Dataset
# -----------------------------
class WeatherCSDIDataset(Dataset):
    """
    windows_NLK: float32 standardized
    mode:
      - 'train': gt_mask placeholder; model uses get_randmask() because target_strategy='random'
      - 'eval' : gt_mask is keep-mask (1=keep,0=masked), either Markov-sampled or loaded from maskbank
    """
    def __init__(
        self,
        windows_NLK: np.ndarray,
        mode: str,
        r_eval_masked: float,
        lm: float,
        precomputed_keepmask_NLK: Optional[np.ndarray] = None,
    ):
        assert mode in ("train", "eval")
        self.x = windows_NLK.astype(np.float32)
        self.mode = mode
        self.r_eval_masked = float(r_eval_masked)
        self.lm = float(lm)
        if precomputed_keepmask_NLK is not None:
            assert precomputed_keepmask_NLK.shape == self.x.shape, (precomputed_keepmask_NLK.shape, self.x.shape)
            self.keep = precomputed_keepmask_NLK.astype(np.float32)
        else:
            self.keep = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_LK = torch.from_numpy(self.x[idx])  # (L,K)
        L, K = x_LK.shape

        observed_data = x_LK
        observed_mask = torch.ones((L, K), dtype=torch.float32)
        timepoints = torch.arange(L, dtype=torch.float32)

        if self.mode == "eval":
            if self.keep is not None:
                gt_mask = torch.from_numpy(self.keep[idx]).float()  # (L,K) keep-mask
            else:
                keep_KL = markov_keep_mask_from_masked_ratio(
                    B=1, K=K, L=L, r_masked=self.r_eval_masked, lm=self.lm, device=torch.device("cpu")
                )[0]  # (K,L)
                gt_mask = keep_KL.transpose(0, 1).contiguous().float()  # (L,K)
        else:
            gt_mask = observed_mask.clone()

        return {
            "observed_data": observed_data,     # (L,K)
            "observed_mask": observed_mask,     # (L,K)
            "timepoints": timepoints,           # (L,)
            "gt_mask": gt_mask,                 # (L,K) keep-mask
            "hist_mask": observed_mask,         # (L,K)
            "cut_length": torch.tensor(0, dtype=torch.long),
        }


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Stack to (B,L,K)
    observed_data = torch.stack([s["observed_data"] for s in samples], dim=0)
    observed_mask = torch.stack([s["observed_mask"] for s in samples], dim=0)
    timepoints = torch.stack([s["timepoints"] for s in samples], dim=0)  # (B,L)
    gt_mask = torch.stack([s["gt_mask"] for s in samples], dim=0)
    hist_mask = torch.stack([s["hist_mask"] for s in samples], dim=0)
    cut_length = torch.stack([s["cut_length"] for s in samples], dim=0)

    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "timepoints": timepoints,
        "gt_mask": gt_mask,
        "hist_mask": hist_mask,
        "cut_length": cut_length,
    }


# -----------------------------
# CSDI specialization
# -----------------------------
class CSDI_Weather(CSDI_base):
    def __init__(self, config, device, target_dim: int):
        super().__init__(target_dim=target_dim, config=config, device=device)
        self.train_pointwise_mask_ratio = float(config["model"].get("train_pointwise_mask_ratio", 0.3))

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()     # (B,L,K)
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp   = batch["timepoints"].to(self.device).float()        # (B,L)
        gt_mask       = batch["gt_mask"].to(self.device).float()
        cut_length    = batch["cut_length"].to(self.device).long()
        hist_mask     = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask       = gt_mask.permute(0, 2, 1)
        hist_mask     = hist_mask.permute(0, 2, 1)

        return observed_data, observed_mask, observed_tp, gt_mask, hist_mask, cut_length

    def get_randmask(self, observed_mask):
        keep_prob = 1.0 - self.train_pointwise_mask_ratio
        bern = torch.rand_like(observed_mask)
        return (bern < keep_prob).float() * observed_mask


def build_config(args):
    return {
        "model": {
            "timeemb": args.timeemb,
            "featureemb": args.featureemb,
            "is_unconditional": False,
            "target_strategy": "random",
            "train_pointwise_mask_ratio": args.train_pointwise_mask_ratio,
        },
        "diffusion": {
            "layers": args.layers,
            "channels": args.channels,
            "nheads": args.nheads,
            "diffusion_embedding_dim": args.diff_emb_dim,
            "num_steps": args.num_steps,
            "schedule": args.schedule,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "is_linear": args.is_linear,
        },
        "train": {
            "epochs": args.epochs,
            "itr_per_epoch": args.itr_per_epoch,
            "lr": args.lr,
        },
    }


@torch.no_grad()
def eval_loader_mae_mse_rmse_sharedmask(model: CSDI_base, loader: DataLoader, nsample: int, device: torch.device):
    """
    Compute *global* MAE/MSE/RMSE over all evaluated (masked) points.
    This avoids batch-size / mask-count weighting artifacts from per-batch averaging.

    evalmask = 1 - keepmask (keepmask is batch['gt_mask']).
    """
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0

    for batch in loader:
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        keep_BLK = batch["gt_mask"].to(device).float()          # (B,L,K) keep
        evalmask_BKL = (1.0 - keep_BLK).permute(0, 2, 1)        # (B,K,L) eval (masked)

        # If observed_mask exists, ensure we don't score truly-unobserved positions
        if "observed_mask" in batch:
            obs_BLK = batch["observed_mask"].to(device).float()     # (B,L,K)
            evalmask_BKL = evalmask_BKL * obs_BLK.permute(0, 2, 1)

        diff = (pred - observed_data) * evalmask_BKL

        total_abs += diff.abs().sum().item()
        total_sq  += (diff ** 2).sum().item()
        total_count += evalmask_BKL.sum().item()

    total_count = max(total_count, 1.0)
    mse = total_sq / total_count
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_count
    return float(mae), float(mse), float(rmse)


@torch.no_grad()
def collect_test_arrays_sharedmask(model: CSDI_base, loader: DataLoader, nsample: int, device: torch.device):
    gt_all, imp_all, cond_all, targ_all = [], [], [], []

    for batch in tqdm(loader, desc="collect", leave=False):
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        keep_BLK = batch["gt_mask"].to(device).float()          # (B,L,K)
        cond_BKL = keep_BLK.permute(0, 2, 1)                    # (B,K,L)
        eval_BKL = 1.0 - cond_BKL                               # (B,K,L)

        imputed_full = cond_BKL * observed_data + (1.0 - cond_BKL) * pred  # (B,K,L)

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_full.permute(0, 2, 1).contiguous()
        cond_BLK = keep_BLK.contiguous()
        targ_BLK = (1.0 - keep_BLK).contiguous()

        gt_all.append(gt_BLK.cpu().numpy())
        imp_all.append(imp_BLK.cpu().numpy())
        cond_all.append(cond_BLK.cpu().numpy())
        targ_all.append(targ_BLK.cpu().numpy())

    return (
        np.concatenate(gt_all, axis=0),
        np.concatenate(imp_all, axis=0),
        np.concatenate(cond_all, axis=0),
        np.concatenate(targ_all, axis=0),
    )


def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int) -> np.ndarray:
    path = os.path.join(mask_dir, f"weather_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    return np.load(path).astype(np.float32)

#def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int, invert: bool = False) -> np.ndarray:
#    path = os.path.join(mask_dir, f"weather_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
#    if not os.path.exists(path):
#        raise FileNotFoundError(f"Shared keepmask not found: {path}")
#    keep = np.load(path).astype(np.float32)
#    if invert or os.environ.get("WEATHER_MASKBANK_INVERT", "0") == "1":
#        keep = 1.0 - keep
#    return keep

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, default="/data/weather.csv")
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--itr_per_epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--valid_epoch_interval", type=int, default=20)

    ap.add_argument("--train_pointwise_mask_ratio", type=float, default=0.15)

    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--nsample", type=int, default=50)

    ap.add_argument("--timeemb", type=int, default=128)
    ap.add_argument("--featureemb", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--diff_emb_dim", type=int, default=128)

    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--schedule", type=str, default="quad", choices=["quad", "linear"])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=0.2)
    ap.add_argument("--is_linear", action="store_true")

    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="")
    #ap.add_argument("--invert_shared_keepmask", action="store_true")
    ap.add_argument("--out_txt", type=str, default=None)
    ap.add_argument("--run_name", type=str, default="csdi_weather_sharedmask")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_weather_maskbank")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_test_arrays_dir", type=str, default="saved_test_arrays_csdi_sharedmask")

    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    out_txt = args.out_txt or os.path.join(args.save_dir, f"{args.run_name}_metrics.txt")

    windows, feat_cols, tcol = load_weather_windows(args.csv, args.seq_len, args.time_col)
    train_w, val_w, test_w = chrono_split_windows(windows)
    train_w, val_w, test_w, mu, sd = standardize_by_train(train_w, val_w, test_w)

    K = train_w.shape[-1]
    print(f"[data] time_col={tcol}  features={K}  seq_len={args.seq_len}")
    print(f"[split] windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}")

    config = build_config(args)
    model = CSDI_Weather(config=config, device=device, target_dim=K).to(device)

    train_loader = DataLoader(
        WeatherCSDIDataset(train_w, mode="train", r_eval_masked=0.1, lm=args.lm),
        batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )
    val_loader_for_trainloop = DataLoader(
        WeatherCSDIDataset(val_w, mode="train", r_eval_masked=0.1, lm=args.lm),
        batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )

    print("[train] training CSDI ...")
    csdi_train(
        model=model,
        config=config["train"],
        train_loader=train_loader,
        valid_loader=val_loader_for_trainloop,
        valid_epoch_interval=args.valid_epoch_interval,
        foldername=args.save_dir,
    )

    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    write_header = not os.path.exists(out_txt)
    with open(out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tMAE\tMSE\tRMSE\n")

        for split_name, split_w in [("val", val_w), ("test", test_w)]:
            for r_m in eval_masked:
                pre_keep = None
                if args.use_shared_evalmask:
                    pre_keep = load_shared_keepmask(args.shared_evalmask_dir, split=split_name, r_m=r_m, seq_len=args.seq_len, seed=args.seed)

                eval_loader = DataLoader(
                    WeatherCSDIDataset(split_w, mode="eval", r_eval_masked=r_m, lm=args.lm, precomputed_keepmask_NLK=pre_keep),
                    batch_size=args.batch, shuffle=False, drop_last=False,
                    num_workers=args.num_workers, collate_fn=collate_batch,
                )

                if args.save_test_arrays and split_name == "test":
                    os.makedirs(args.save_test_arrays_dir, exist_ok=True)
                    gt, imputed, condmask, targetmask = collect_test_arrays_sharedmask(
                        model=model, loader=eval_loader, nsample=args.nsample, device=device
                    )
                    tag = f"csdi_test_r{r_m:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}"
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_gt.npy"), gt)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_imputed.npy"), imputed)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_condmask.npy"), condmask)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_targetmask.npy"), targetmask)

                    np.savez(
                        os.path.join(args.save_test_arrays_dir, f"metadata_seed{args.seed}_L{args.seq_len}.npz"),
                        mu=mu, sd=sd, feature_cols=np.array(feat_cols, dtype=object),
                        seq_len=args.seq_len, time_col=tcol, lm=args.lm,
                        train_pointwise_mask_ratio=args.train_pointwise_mask_ratio,
                        shared_evalmask=bool(args.use_shared_evalmask),
                    )
                    print(f"[save] Wrote test arrays to: {args.save_test_arrays_dir}  (r_masked={r_m:.2f})")

                mae, mse, rmse = eval_loader_mae_mse_rmse_sharedmask(model, eval_loader, nsample=args.nsample, device=device)
                f.write(f"{split_name}\t{r_m:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{split_name}] r_masked={r_m:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")

    print(f"Done. Metrics appended to: {out_txt}")


if __name__ == "__main__":
    main()

