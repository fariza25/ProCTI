#!/usr/bin/env python3

import os
import json
import argparse
import random
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from csdi_main_model import CSDI_base
from csdi_utils import train as csdi_train


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Markov keep-mask from MASKED ratio (segment-based)
# -----------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device):
    """
    Returns keepmask (B,K,L) with 1=kept/observed, 0=masked.
    r_masked is missingness ratio (fraction masked).
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / r_keep
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
# Shared Beijing window loading
# -----------------------------
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
            f"Expected all windows to have shape (N,L,K), got "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    if train_w.shape[1:] != val_w.shape[1:] or train_w.shape[1:] != test_w.shape[1:]:
        raise ValueError(
            f"Shared window shape mismatch: "
            f"train={train_w.shape}, val={val_w.shape}, test={test_w.shape}"
        )

    return train_w, val_w, test_w


# -----------------------------
# Shared Beijing maskbank loader
# -----------------------------
def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    """
    Loads:
      {split}_maskbank_seed{seed}.npz
    where each key is "0.10", "0.30", etc and stores evalmask:
      1 = masked
      0 = observed

    Converts to keepmask expected by CSDI eval dataset:
      keepmask = 1 - evalmask
    """
    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared maskbank not found: {path}")

    obj = np.load(path)
    key = f"{r_m:.2f}"
    if key not in obj:
        raise KeyError(f"Ratio key {key} not found in {path}. Available keys: {obj.files}")

    evalmask = obj[key].astype(np.float32)   # 1=masked, 0=observed
    keepmask = 1.0 - evalmask                # 1=observed, 0=masked
    return keepmask


def safe_batch_size(requested: int, n: int) -> int:
    if n <= 0:
        return 1
    return max(1, min(int(requested), int(n)))


# -----------------------------
# Dataset
# -----------------------------
class WindowCSDIDataset(Dataset):
    """
    windows_NLK: float32 pre-windowed data
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
        observed_mask = torch.isfinite(x_LK).float()
        if not bool(observed_mask.all()):
            observed_data = torch.where(torch.isfinite(x_LK), x_LK, torch.zeros_like(x_LK))

        timepoints = torch.arange(L, dtype=torch.float32)

        if self.mode == "eval":
            if self.keep is not None:
                gt_mask = torch.from_numpy(self.keep[idx]).float()  # (L,K) keep-mask
            else:
                keep_KL = markov_keep_mask_from_masked_ratio(
                    B=1, K=K, L=L, r_masked=self.r_eval_masked, lm=self.lm, device=torch.device("cpu")
                )[0]  # (K,L)
                gt_mask = keep_KL.transpose(0, 1).contiguous().float()  # (L,K)

            gt_mask = gt_mask * observed_mask
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
    observed_data = torch.stack([s["observed_data"] for s in samples], dim=0)  # (B,L,K)
    observed_mask = torch.stack([s["observed_mask"] for s in samples], dim=0)
    timepoints = torch.stack([s["timepoints"] for s in samples], dim=0)        # (B,L)
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
class CSDI_Beijing(CSDI_base):
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
    Compute global MAE/MSE/RMSE over all evaluated masked points.
    evalmask = 1 - keepmask, where keepmask is batch['gt_mask'].
    """
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0
    total_keep = 0.0
    total_keep_count = 0.0

    for batch in loader:
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        keep_BLK = batch["gt_mask"].to(device).float()          # (B,L,K)
        evalmask_BKL = (1.0 - keep_BLK).permute(0, 2, 1)        # (B,K,L)

        if "observed_mask" in batch:
            obs_BLK = batch["observed_mask"].to(device).float()     # (B,L,K)
            evalmask_BKL = evalmask_BKL * obs_BLK.permute(0, 2, 1)

        diff = (pred - observed_data) * evalmask_BKL

        total_abs += diff.abs().sum().item()
        total_sq  += (diff ** 2).sum().item()
        denom = evalmask_BKL.sum().item()
        total_count += denom

        total_keep += keep_BLK.mean().item() * denom
        total_keep_count += denom

    total_count = max(total_count, 1.0)
    mse = total_sq / total_count
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_count
    r_obs = float(total_keep / max(total_keep_count, 1.0))
    return float(mae), float(mse), float(rmse), r_obs


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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", type=str, required=True,
                    help="Directory containing train_windows.npy, val_windows.npy, test_windows.npy and maskbanks")
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
    ap.add_argument("--out_txt", type=str, default=None)
    ap.add_argument("--run_name", type=str, default="csdi_beijing_sharedmask")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_test_arrays_dir", type=str, default="saved_test_arrays_csdi_beijing_sharedmask")

    args = ap.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    out_txt = args.out_txt or (os.path.join(args.save_dir, f"{args.run_name}_metrics.txt") if args.save_dir else f"{args.run_name}_metrics.txt")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- shared Beijing windows
    train_w, val_w, test_w = load_shared_windows(args.shared_data_dir)
    K = train_w.shape[-1]
    L = train_w.shape[1]

    if L != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={L}, but args.seq_len={args.seq_len}")

    print(f"[data] Beijing shared windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}  seq_len={args.seq_len}  K={K}")

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

    config = build_config(args)
    model = CSDI_Beijing(config=config, device=device, target_dim=K).to(device)

    train_loader = DataLoader(
        WindowCSDIDataset(train_w, mode="train", r_eval_masked=0.1, lm=args.lm),
        batch_size=safe_batch_size(args.batch, len(train_w)),
        shuffle=True, drop_last=(len(train_w) > 1),
        num_workers=args.num_workers, collate_fn=collate_batch,
    )
    val_loader_for_trainloop = DataLoader(
        WindowCSDIDataset(val_w, mode="train", r_eval_masked=0.1, lm=args.lm),
        batch_size=safe_batch_size(args.batch, len(val_w)),
        shuffle=False, drop_last=False,
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
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split_name, split_w in [("val", val_w), ("test", test_w)]:
            for r_m in eval_masked:
                pre_keep = None
                if args.use_shared_evalmask:
                    pre_keep = load_shared_keepmask(
                        args.shared_evalmask_dir, split=split_name, r_m=r_m, seed=args.seed
                    )

                    if pre_keep.shape != split_w.shape:
                        raise ValueError(
                            f"Shared keepmask shape mismatch for split={split_name}, r={r_m:.2f}: "
                            f"keep={pre_keep.shape}, windows={split_w.shape}"
                        )

                eval_loader = DataLoader(
                    WindowCSDIDataset(
                        split_w, mode="eval", r_eval_masked=r_m, lm=args.lm,
                        precomputed_keepmask_NLK=pre_keep
                    ),
                    batch_size=safe_batch_size(args.batch, len(split_w)),
                    shuffle=False, drop_last=False,
                    num_workers=args.num_workers, collate_fn=collate_batch,
                )

                if args.save_test_arrays and split_name == "test":
                    os.makedirs(args.save_test_arrays_dir, exist_ok=True)
                    gt, imputed, condmask, targetmask = collect_test_arrays_sharedmask(
                        model=model, loader=eval_loader, nsample=args.nsample, device=device
                    )
                    tag = f"csdi_beijing_test_r{r_m:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}"
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_gt.npy"), gt)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_imputed.npy"), imputed)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_condmask.npy"), condmask)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_targetmask.npy"), targetmask)

                    meta_out = {
                        "seq_len": int(args.seq_len),
                        "lm": float(args.lm),
                        "train_pointwise_mask_ratio": float(args.train_pointwise_mask_ratio),
                        "shared_evalmask": bool(args.use_shared_evalmask),
                        "shared_data_dir": args.shared_data_dir,
                    }
                    with open(os.path.join(args.save_test_arrays_dir, f"metadata_seed{args.seed}_L{args.seq_len}.json"), "w") as jf:
                        json.dump(meta_out, jf, indent=2)

                    print(f"[save] Wrote test arrays to: {args.save_test_arrays_dir}  (r_masked={r_m:.2f})")

                mae, mse, rmse, r_obs = eval_loader_mae_mse_rmse_sharedmask(
                    model, eval_loader, nsample=args.nsample, device=device
                )
                f.write(f"{split_name}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                print(f"[{split_name}] r_masked={r_m:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  r_obs={r_obs:.2f}")

    print(f"Done. Metrics appended to: {out_txt}")


if __name__ == "__main__":
    main()
