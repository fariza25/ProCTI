#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import re
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DATASET = "weather"
PREFIX = "weather"
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH = 64


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


def import_diffusionts(diffusionts_root: str):
    diffusionts_root = os.path.abspath(diffusionts_root)
    if diffusionts_root not in sys.path:
        sys.path.insert(0, diffusionts_root)
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
    from engine.solver import Trainer
    from engine.logger import Logger
    return Diffusion_TS, Trainer, Logger


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
class WindowDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


@torch.no_grad()
def evaluate_loader(
    model,
    loader: DataLoader,
    evalmask_NLK: np.ndarray,
    device: torch.device,
    coef: float,
    step_size: float,
    sampling_steps: int,
    save_arrays: bool = False,
    out_prefix: str = None,
    split_name: str = "test",
):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_den = 0.0
    total_keep = 0.0
    total_obs = 0.0
    offset = 0

    gt_list = []
    imputed_list = []
    evalmask_list = []
    keepmask_list = []

    for xb in loader:
        xb = xb.to(device).float()
        B = xb.shape[0]

        this_evalmask = torch.from_numpy(evalmask_NLK[offset:offset+B]).to(device).float()
        offset += B

        obsb = torch.isfinite(xb).float()
        keepb = obsb * (1.0 - this_evalmask)

        sample = model.fast_sample_infill(
            shape=xb.shape,
            target=xb * keepb,
            partial_mask=keepb.bool(),
            model_kwargs={"coef": coef, "learning_rate": step_size},
            sampling_timesteps=sampling_steps
        )

        evalmask = (1.0 - keepb) * obsb
        diff = (sample - xb) * evalmask

        den = float(evalmask.sum().item())
        if den < 1.0:
            continue

        total_abs += float(diff.abs().sum().item())
        total_sq += float((diff ** 2).sum().item())
        total_den += den
        total_keep += float(keepb.sum().item())
        total_obs += float(obsb.sum().item())

        if save_arrays:
            gt_list.append(xb.detach().cpu().numpy())
            imputed_list.append(sample.detach().cpu().numpy())
            evalmask_list.append(evalmask.detach().cpu().numpy())
            keepmask_list.append(keepb.detach().cpu().numpy())

    total_den = max(total_den, 1.0)
    mae = total_abs / total_den
    mse = total_sq / total_den
    rmse = float(np.sqrt(mse))
    r_obs = total_keep / max(total_obs, 1.0)

    if save_arrays:
        if out_prefix is None:
            out_prefix = f"diffusionts_{DATASET}"
        gt = np.concatenate(gt_list, axis=0)
        imputed = np.concatenate(imputed_list, axis=0)
        evalmask_arr = np.concatenate(evalmask_list, axis=0)
        keepmask_arr = np.concatenate(keepmask_list, axis=0)

        np.save(f"{out_prefix}_{split_name}_gt.npy", gt)
        np.save(f"{out_prefix}_{split_name}_imputed.npy", imputed)
        np.save(f"{out_prefix}_{split_name}_evalmask.npy", evalmask_arr)
        np.save(f"{out_prefix}_{split_name}_keepmask.npy", keepmask_arr)

        print(f"[saved] {out_prefix}_{split_name}_gt.npy")
        print(f"[saved] {out_prefix}_{split_name}_imputed.npy")
        print(f"[saved] {out_prefix}_{split_name}_evalmask.npy")
        print(f"[saved] {out_prefix}_{split_name}_keepmask.npy")

    return float(mae), float(mse), float(rmse), float(r_obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusionts_root", type=str, required=True)
    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--coef", type=float, default=1e-2)
    ap.add_argument("--step_size", type=float, default=5e-2)
    ap.add_argument("--sampling_steps", type=int, default=250)
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--log_frequency", type=int, default=100)
    ap.add_argument("--milestone", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", type=str, default="diffusionts_weather")
    ap.add_argument("--output", type=str, default="OUTPUT")
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--out_txt", type=str, default="diffusionts_weather_channeldrop_metrics.tsv")
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--array_out_prefix", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    args.save_dir = os.path.join(args.output, args.name)
    os.makedirs(args.save_dir, exist_ok=True)

    train_w, val_w, test_w = load_window_assets(args.asset_dir, args.seq_len)
    K = int(train_w.shape[-1])
    L = int(args.seq_len)

    val_evalmask, _, _ = load_maskbank_npz(args.val_maskbank)
    test_evalmask, _, _ = load_maskbank_npz(args.test_maskbank)

    assert val_evalmask.shape == val_w.shape, f"val mismatch: {val_evalmask.shape} vs {val_w.shape}"
    assert test_evalmask.shape == test_w.shape, f"test mismatch: {test_evalmask.shape} vs {test_w.shape}"

    print(f"[data] train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={L}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    Diffusion_TS, Trainer, Logger = import_diffusionts(args.diffusionts_root)

    train_bs = safe_batch_size(args.batch, len(train_w))
    eval_bs = safe_batch_size(args.batch, max(len(val_w), len(test_w)))

    train_loader = DataLoader(
        WindowDataset(train_w),
        batch_size=train_bs,
        shuffle=True,
        drop_last=(len(train_w) >= train_bs and train_bs > 1),
    )
    val_loader = DataLoader(WindowDataset(val_w), batch_size=eval_bs, shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowDataset(test_w), batch_size=eval_bs, shuffle=False, drop_last=False)

    model = Diffusion_TS(
        seq_length=L,
        feature_size=K,
        n_layer_enc=4,
        n_layer_dec=4,
        d_model=96,
        timesteps=1000,
        sampling_timesteps=args.sampling_steps,
        loss_type='l1',
        beta_schedule='cosine',
        n_heads=4,
        use_ff=True,
        eta=0.0
    ).to(device)

    config = {
        "solver": {
            "base_lr": args.lr,
            "max_epochs": args.steps,
            "gradient_accumulate_every": 2,
            "save_cycle": 2000,
            "results_folder": os.path.join(args.save_dir, "checkpoints"),
            "ema": {"decay": 0.995, "update_interval": 10},
            "scheduler": {
                "target": "engine.lr_sch.ReduceLROnPlateauWithWarmup",
                "params": {
                    "factor": 0.5,
                    "patience": 1000,
                    "min_lr": args.lr,
                    "threshold": 1e-1,
                    "threshold_mode": "rel",
                    "warmup_lr": 8e-4,
                    "warmup": 500,
                    "verbose": False
                }
            }
        }
    }

    logger = Logger(args)
    trainer = Trainer(config=config, args=args, model=model, dataloader={"dataloader": train_loader, "dataset": None}, logger=logger)

    print(f"[train] Training Diffusion-TS on {DATASET}...")
    t0 = time.time()
    trainer.train()
    print(f"[train] done in {time.time()-t0:.1f}s")

    ema_model = trainer.ema.ema_model
    ema_model.eval()

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")

        # val
        mae, mse, rmse, r_obs = evaluate_loader(
            ema_model,
            val_loader,
            val_evalmask,
            device,
            coef=args.coef,
            step_size=args.step_size,
            sampling_steps=args.sampling_steps,
            save_arrays=False,
            out_prefix=args.array_out_prefix,
            split_name="val",
        )
        f.write(f"val\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print(f"[val] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}")

        # test
        mae, mse, rmse, r_obs = evaluate_loader(
            ema_model,
            test_loader,
            test_evalmask,
            device,
            coef=args.coef,
            step_size=args.step_size,
            sampling_steps=args.sampling_steps,
            save_arrays=args.save_test_arrays,
            out_prefix=args.array_out_prefix,
            split_name="test",
        )
        f.write(f"test\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print(f"[test] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()

