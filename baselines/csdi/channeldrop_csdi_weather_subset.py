#!/usr/bin/env python3


import os
import json
import argparse
import random
import re
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Weather feature subset
# -----------------------------
WEATHER_FEATURE_ORDER = [
    "p", "T", "Tpot", "Tdew", "rh", "VPmax", "VPact", "VPdef", "sh", "H2OC",
    "rho", "wv", "max. wv", "wd", "rain", "raining", "SWDR", "PAR", "max. PAR", "Tlog", "OT"
]
WEATHER_SUBSET_FEATURES = ["p", "T", "H2OC", "wv", "max. wv", "wd", "rain", "raining", "PAR", "OT"]
WEATHER_SUBSET_INDICES = [WEATHER_FEATURE_ORDER.index(f) for f in WEATHER_SUBSET_FEATURES]


def subset_weather_windows(windows_NLK: np.ndarray) -> np.ndarray:
    """Keep only the selected low-correlation weather features."""
    windows_NLK = windows_NLK.astype(np.float32)
    max_idx = max(WEATHER_SUBSET_INDICES)
    if windows_NLK.ndim != 3:
        raise ValueError(f"Expected windows with shape (N,L,K), got {windows_NLK.shape}")
    if windows_NLK.shape[-1] <= max_idx:
        raise ValueError(
            f"Weather windows have K={windows_NLK.shape[-1]}, but selected indices require index {max_idx}. "
            "Check WEATHER_FEATURE_ORDER against the preprocessing order."
        )
    return windows_NLK[:, :, WEATHER_SUBSET_INDICES].astype(np.float32)


def _read_maskbank_metadata(path: str):
    obj = np.load(path, allow_pickle=True)
    if "eval_mask" not in obj:
        raise KeyError(f"{path} missing eval_mask")
    full_eval_mask = obj["eval_mask"].astype(np.float32)
    if full_eval_mask.ndim != 3:
        raise ValueError(f"Expected eval_mask with shape (N,L,K), got {full_eval_mask.shape}")
    metadata = {}
    if "metadata_json" in obj:
        try:
            metadata = json.loads(str(obj["metadata_json"]))
        except Exception:
            metadata = {}
    return full_eval_mask, metadata


def _parse_seed_and_drop(path: str, metadata: dict):
    base = os.path.basename(path)
    seed_match = re.search(r"seed(\d+)", base)
    drop_match = re.search(r"drop(\d+)", base)
    seed = int(seed_match.group(1)) if seed_match else int(metadata.get("seed", 1))
    n_drop = int(drop_match.group(1)) if drop_match else int(str(metadata.get("protocol", "drop1")).replace("drop", ""))
    return seed, n_drop


def make_subset_weather_channeldrop_mask(path: str):
    """
    Generate a channel-drop mask over the selected feature subset only.

    The original maskbank was produced for all weather features. Since this experiment uses only
    WEATHER_SUBSET_FEATURES, we regenerate the drop1/drop2 mask over K=len(WEATHER_SUBSET_FEATURES)
    using the seed and protocol encoded in the maskbank filename. This keeps masks identical across
    all models while ensuring drop1/drop2 applies only to the selected features.
    """
    full_eval_mask, metadata = _read_maskbank_metadata(path)
    N, L, _ = full_eval_mask.shape
    K = len(WEATHER_SUBSET_FEATURES)
    seed, n_drop = _parse_seed_and_drop(path, metadata)
    if n_drop > K:
        raise ValueError(f"Cannot drop {n_drop} channels when selected subset has only K={K} features")
    rng = np.random.default_rng(seed)
    eval_mask = np.zeros((N, L, K), dtype=np.float32)
    dropped_channels = np.full((N, n_drop), -1, dtype=np.int64)
    for i in range(N):
        chosen = rng.choice(K, size=n_drop, replace=False)
        dropped_channels[i] = chosen
        eval_mask[i, :, chosen] = 1.0
    metadata = dict(metadata)
    metadata["feature_subset"] = list(WEATHER_SUBSET_FEATURES)
    metadata["feature_subset_indices_from_full_weather"] = [int(i) for i in WEATHER_SUBSET_INDICES]
    metadata["subset_mask_generated_in_script"] = True
    return eval_mask.astype(np.float32), dropped_channels.astype(np.int64), metadata

from csdi_main_model import CSDI_base
from csdi_utils import train as csdi_train


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Asset / maskbank loading
# --------------------------------------------------

def load_window_assets(asset_dir: str, seq_len: int):
    train_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_test_windows.npy")).astype(np.float32)
    train_w = subset_weather_windows(train_w)
    val_w = subset_weather_windows(val_w)
    test_w = subset_weather_windows(test_w)
    print(f"[features] using {WEATHER_SUBSET_FEATURES} from indices {WEATHER_SUBSET_INDICES}")
    return train_w, val_w, test_w
def load_maskbank_npz(path: str):
    return make_subset_weather_channeldrop_mask(path)
# --------------------------------------------------
# Dataset
# --------------------------------------------------

class WeatherCSDIDataset(Dataset):
    """
    windows_NLK: standardized float32, shape (N,L,K)

    mode:
      - train: gt_mask is placeholder (all ones), model uses get_randmask()
      - eval : gt_mask is keep-mask derived from precomputed eval-mask
    """

    def __init__(
        self,
        windows_NLK: np.ndarray,
        mode: str,
        precomputed_evalmask_NLK: Optional[np.ndarray] = None,
    ):
        assert mode in ("train", "eval")
        self.x = windows_NLK.astype(np.float32)
        self.mode = mode

        if self.mode == "eval":
            assert precomputed_evalmask_NLK is not None, "eval mode requires precomputed_evalmask_NLK"
            assert precomputed_evalmask_NLK.shape == self.x.shape, (
                precomputed_evalmask_NLK.shape, self.x.shape
            )
            self.evalmask = precomputed_evalmask_NLK.astype(np.float32)
        else:
            self.evalmask = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_LK = torch.from_numpy(self.x[idx])  # (L,K)
        L, K = x_LK.shape

        observed_data = x_LK
        observed_mask = torch.ones((L, K), dtype=torch.float32)
        timepoints = torch.arange(L, dtype=torch.float32)

        if self.mode == "eval":
            eval_mask = torch.from_numpy(self.evalmask[idx]).float()  # (L,K), 1=masked
            gt_mask = 1.0 - eval_mask                                # keep-mask, 1=keep
        else:
            gt_mask = observed_mask.clone()

        return {
            "observed_data": observed_data,   # (L,K)
            "observed_mask": observed_mask,   # (L,K)
            "timepoints": timepoints,         # (L,)
            "gt_mask": gt_mask,               # (L,K) keep-mask
            "hist_mask": observed_mask,       # (L,K)
            "cut_length": torch.tensor(0, dtype=torch.long),
        }


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    observed_data = torch.stack([s["observed_data"] for s in samples], dim=0)  # (B,L,K)
    observed_mask = torch.stack([s["observed_mask"] for s in samples], dim=0)  # (B,L,K)
    timepoints = torch.stack([s["timepoints"] for s in samples], dim=0)        # (B,L)
    gt_mask = torch.stack([s["gt_mask"] for s in samples], dim=0)              # (B,L,K)
    hist_mask = torch.stack([s["hist_mask"] for s in samples], dim=0)          # (B,L,K)
    cut_length = torch.stack([s["cut_length"] for s in samples], dim=0)

    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "timepoints": timepoints,
        "gt_mask": gt_mask,
        "hist_mask": hist_mask,
        "cut_length": cut_length,
    }


# --------------------------------------------------
# CSDI specialization
# --------------------------------------------------

class CSDI_Weather(CSDI_base):
    def __init__(self, config, device, target_dim: int):
        super().__init__(target_dim=target_dim, config=config, device=device)
        self.train_pointwise_mask_ratio = float(config["model"].get("train_pointwise_mask_ratio", 0.15))

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()   # (B,L,K)
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()        # (B,L)
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        hist_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        hist_mask = hist_mask.permute(0, 2, 1)

        return observed_data, observed_mask, observed_tp, gt_mask, hist_mask, cut_length

    def get_randmask(self, observed_mask):
        keep_prob = 1.0 - self.train_pointwise_mask_ratio
        bern = torch.rand_like(observed_mask)
        return (bern < keep_prob).float() * observed_mask


# --------------------------------------------------
# Config
# --------------------------------------------------

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


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

@torch.no_grad()
def eval_loader_mae_mse_rmse(model: CSDI_base, loader: DataLoader, nsample: int, device: torch.device):
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0

    for batch in loader:
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        keep_BLK = batch["gt_mask"].to(device).float()      # (B,L,K)
        evalmask_BKL = (1.0 - keep_BLK).permute(0, 2, 1)    # (B,K,L)

        if "observed_mask" in batch:
            obs_BLK = batch["observed_mask"].to(device).float()
            evalmask_BKL = evalmask_BKL * obs_BLK.permute(0, 2, 1)

        diff = (pred - observed_data) * evalmask_BKL

        total_abs += diff.abs().sum().item()
        total_sq += (diff ** 2).sum().item()
        total_count += evalmask_BKL.sum().item()

    total_count = max(total_count, 1.0)
    mse = total_sq / total_count
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_count
    return float(mae), float(mse), float(rmse)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--itr_per_epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--valid_epoch_interval", type=int, default=20)

    ap.add_argument("--train_pointwise_mask_ratio", type=float, default=0.15)

    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")

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
    ap.add_argument("--out_txt", type=str, default="csdi_weather_channel_drop_metrics.txt")
    ap.add_argument("--run_name", type=str, default="csdi_weather_channeldrop")

    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    # -------------------------
    # Load exact saved assets
    # -------------------------
    train_w, val_w, test_w = load_window_assets(args.asset_dir, args.seq_len)
    K = train_w.shape[-1]

    print(f"[data] train={len(train_w)} val={len(val_w)} test={len(test_w)} L={args.seq_len} K={K}")

    # -------------------------
    # Load exact saved maskbanks
    # -------------------------
    val_evalmask, _, val_meta = load_maskbank_npz(args.val_maskbank)
    test_evalmask, _, test_meta = load_maskbank_npz(args.test_maskbank)

    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    print(f"[sanity] val windows={len(val_w)} val mask={len(val_evalmask)}")
    print(f"[sanity] test windows={len(test_w)} test mask={len(test_evalmask)}")

    assert val_evalmask.shape == val_w.shape, f"val mismatch: windows {val_w.shape} vs mask {val_evalmask.shape}"
    assert test_evalmask.shape == test_w.shape, f"test mismatch: windows {test_w.shape} vs mask {test_evalmask.shape}"

    # -------------------------
    # Build model
    # -------------------------
    config = build_config(args)
    model = CSDI_Weather(config=config, device=device, target_dim=K).to(device)

    # -------------------------
    # Loaders
    # -------------------------
    train_loader = DataLoader(
        WeatherCSDIDataset(train_w, mode="train"),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    val_loader_for_trainloop = DataLoader(
        WeatherCSDIDataset(val_w, mode="train"),
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        WeatherCSDIDataset(val_w, mode="eval", precomputed_evalmask_NLK=val_evalmask),
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    test_loader = DataLoader(
        WeatherCSDIDataset(test_w, mode="eval", precomputed_evalmask_NLK=test_evalmask),
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    # -------------------------
    # Train
    # -------------------------
    print("[train] training CSDI ...")
    csdi_train(
        model=model,
        config=config["train"],
        train_loader=train_loader,
        valid_loader=val_loader_for_trainloop,
        valid_epoch_interval=args.valid_epoch_interval,
        foldername=args.save_dir,
    )

    # -------------------------
    # Evaluate
    # -------------------------
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")

        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            mae, mse, rmse = eval_loader_mae_mse_rmse(
                model=model,
                loader=loader,
                nsample=args.nsample,
                device=device,
            )
            f.write(f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
            print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")

    print(f"Done. Metrics appended to: {args.out_txt}")


if __name__ == "__main__":
    main()

