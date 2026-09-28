import os
import json
import argparse
import random
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from csdi_main_model import CSDI_base
from csdi_utils import train as csdi_train

DATASET = "physionet"


TARGET_FEATURES = ["SBP", "O2Sat", "MAP", "Resp", "HR"]
CANONICAL_PHYSIONET_INDEX = {
    "HR": 0,
    "O2Sat": 1,
    "SBP": 3,
    "MAP": 4,
    "Resp": 6,
}

def get_target_feature_indices(num_features: int):
    if num_features == len(TARGET_FEATURES):
        return list(range(len(TARGET_FEATURES)))
    needed = max(CANONICAL_PHYSIONET_INDEX[name] for name in TARGET_FEATURES)
    if num_features <= needed:
        raise ValueError(
            f"Cannot select {TARGET_FEATURES} from tensor with K={num_features}. "
            "Expected either the original PhysioNet channel order or an already-filtered 5-channel tensor."
        )
    return [CANONICAL_PHYSIONET_INDEX[name] for name in TARGET_FEATURES]

def restrict_physionet_windows(windows_NLK: np.ndarray):
    idx = get_target_feature_indices(int(windows_NLK.shape[-1]))
    return windows_NLK[:, :, idx].astype(np.float32), idx

def restrict_physionet_evalmask(evalmask_NLK: np.ndarray, idx):
    return evalmask_NLK[:, :, idx].astype(np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_window_assets(asset_dir: str, seq_len: int):
    train_w = np.load(os.path.join(asset_dir, f"physionet_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"physionet_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"physionet_seq{seq_len}_test_windows.npy")).astype(np.float32)
    return train_w, val_w, test_w


def load_maskbank_npz(path: str):
    obj = np.load(path, allow_pickle=True)
    eval_mask = obj["eval_mask"].astype(np.float32)
    dropped_channels = obj["dropped_channels"].astype(np.int64) if "dropped_channels" in obj else None
    metadata = {}
    if "metadata_json" in obj:
        try:
            metadata = json.loads(str(obj["metadata_json"]))
        except Exception:
            metadata = {}
    return eval_mask, dropped_channels, metadata


class WindowDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray, mode: str, precomputed_evalmask_NLK: Optional[np.ndarray] = None):
        assert mode in ("train", "eval")
        self.x_raw = windows_NLK.astype(np.float32)
        self.obs = np.isfinite(self.x_raw).astype(np.float32)
        self.x = np.nan_to_num(self.x_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.mode = mode
        self.evalmask = None if precomputed_evalmask_NLK is None else precomputed_evalmask_NLK.astype(np.float32)
        if self.mode == "eval":
            assert self.evalmask is not None and self.evalmask.shape == self.x.shape

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_LK = torch.from_numpy(self.x[idx])
        obs_LK = torch.from_numpy(self.obs[idx])
        L, _ = x_LK.shape
        timepoints = torch.arange(L, dtype=torch.float32)

        if self.mode == "eval":
            eval_mask = torch.from_numpy(self.evalmask[idx]).float()
            gt_mask = obs_LK * (1.0 - eval_mask)  # keep-mask within observed positions
        else:
            gt_mask = obs_LK.clone()

        return {
            "observed_data": x_LK,
            "observed_mask": obs_LK,
            "timepoints": timepoints,
            "gt_mask": gt_mask,
            "hist_mask": obs_LK,
            "cut_length": torch.tensor(0, dtype=torch.long),
        }


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "observed_data": torch.stack([s["observed_data"] for s in samples], dim=0),
        "observed_mask": torch.stack([s["observed_mask"] for s in samples], dim=0),
        "timepoints": torch.stack([s["timepoints"] for s in samples], dim=0),
        "gt_mask": torch.stack([s["gt_mask"] for s in samples], dim=0),
        "hist_mask": torch.stack([s["hist_mask"] for s in samples], dim=0),
        "cut_length": torch.stack([s["cut_length"] for s in samples], dim=0),
    }


class CSDI_Dataset(CSDI_base):
    def __init__(self, config, device, target_dim: int):
        super().__init__(target_dim=target_dim, config=config, device=device)
        self.train_pointwise_mask_ratio = float(config["model"].get("train_pointwise_mask_ratio", 0.15))

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()   # (B,L,K)
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
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
def eval_loader_mae_mse_rmse(model: CSDI_base, loader: DataLoader, nsample: int, device: torch.device):
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0

    for batch in loader:
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        keep_BLK = batch["gt_mask"].to(device).float()      # (B,L,K)
        evalmask_BKL = (1.0 - keep_BLK).permute(0, 2, 1)    # (B,K,L)
        obs_BKL = batch["observed_mask"].to(device).float().permute(0, 2, 1)
        evalmask_BKL = evalmask_BKL * obs_BKL

        diff = (pred - observed_data) * evalmask_BKL
        total_abs += diff.abs().sum().item()
        total_sq += (diff ** 2).sum().item()
        total_count += evalmask_BKL.sum().item()

    total_count = max(total_count, 1.0)
    mse = total_sq / total_count
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_count
    return float(mae), float(mse), float(rmse)


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
    ap.add_argument("--out_txt", type=str, default=f"csdi_physionet_channeldrop.txt")
    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    train_w, val_w, test_w = load_window_assets(args.asset_dir, args.seq_len)
    val_evalmask, _, _ = load_maskbank_npz(args.val_maskbank)
    test_evalmask, _, _ = load_maskbank_npz(args.test_maskbank)

    train_w, feat_idx = restrict_physionet_windows(train_w)
    val_w, _ = restrict_physionet_windows(val_w)
    test_w, _ = restrict_physionet_windows(test_w)
    val_evalmask = restrict_physionet_evalmask(val_evalmask, feat_idx)
    test_evalmask = restrict_physionet_evalmask(test_evalmask, feat_idx)
    K = train_w.shape[-1]
    print(f"[data] train={len(train_w)} val={len(val_w)} test={len(test_w)} L={args.seq_len} K={K}")
    print(f"[features] using {TARGET_FEATURES} from indices {feat_idx}")

    assert val_evalmask.shape == val_w.shape
    assert test_evalmask.shape == test_w.shape

    config = build_config(args)
    model = CSDI_Dataset(config=config, device=device, target_dim=K).to(device)

    train_loader = DataLoader(WindowDataset(train_w, mode="train"), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader_train = DataLoader(WindowDataset(val_w, mode="train"), batch_size=args.batch, shuffle=False, drop_last=False, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(WindowDataset(val_w, mode="eval", precomputed_evalmask_NLK=val_evalmask), batch_size=args.batch, shuffle=False, drop_last=False, num_workers=args.num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(WindowDataset(test_w, mode="eval", precomputed_evalmask_NLK=test_evalmask), batch_size=args.batch, shuffle=False, drop_last=False, num_workers=args.num_workers, collate_fn=collate_batch)

    print("[train] training CSDI ...")
    csdi_train(model=model, config=config["train"], train_loader=train_loader, valid_loader=val_loader_train, valid_epoch_interval=args.valid_epoch_interval, foldername=args.save_dir)

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")
        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            mae, mse, rmse = eval_loader_mae_mse_rmse(model, loader, args.nsample, device)
            f.write(f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
            print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")

    print(f"Done. Metrics appended to: {args.out_txt}")


if __name__ == "__main__":
    main()

