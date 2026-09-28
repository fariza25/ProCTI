#!/usr/bin/env python3

import os
import sys
import argparse
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# reproducibility
# -------------------------------------------------

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


# -------------------------------------------------
# Shared Beijing loading
# -------------------------------------------------

def load_shared_windows(shared_data_dir: str):
    train_path = os.path.join(shared_data_dir, "train_windows.npy")
    val_path = os.path.join(shared_data_dir, "val_windows.npy")
    test_path = os.path.join(shared_data_dir, "test_windows.npy")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing shared window file: {p}")

    train = np.load(train_path).astype(np.float32)
    val = np.load(val_path).astype(np.float32)
    test = np.load(test_path).astype(np.float32)

    if train.ndim != 3 or val.ndim != 3 or test.ndim != 3:
        raise ValueError(
            f"Expected windows shape (N,L,K), got "
            f"train={train.shape}, val={val.shape}, test={test.shape}"
        )

    if train.shape[1:] != val.shape[1:] or train.shape[1:] != test.shape[1:]:
        raise ValueError(
            f"Window shape mismatch: "
            f"train={train.shape}, val={val.shape}, test={test.shape}"
        )

    return train, val, test


def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    """
    Loads shared Beijing evalmask:
      {split}_maskbank_seed{seed}.npz
    key like "0.10"

    Saved mask semantics:
      1 = masked
      0 = observed

    Returns keepmask:
      1 = observed
      0 = masked
    """
    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")

    obj = np.load(path)
    key = f"{r_m:.2f}"
    if key not in obj:
        raise KeyError(f"Ratio key {key} missing from {path}. Available keys: {obj.files}")

    evalmask = obj[key].astype(np.float32)  # (N,L,K), 1=masked
    keep = 1.0 - evalmask                   # (N,L,K), 1=observed
    return keep


# -------------------------------------------------
# Markov keep-mask
# returns (B, K, L), 1=observed/kept, 0=masked
# -------------------------------------------------

def markov_keep_mask(B: int, K: int, L: int, r_masked: float, lm: float, device):
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0

    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)

    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < (p_m if state == 0 else p_u):
                    state = 1 - state
    return torch.from_numpy(out).to(device)


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, windows_nlk: np.ndarray):
        self.x = windows_nlk.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L, K)


# -------------------------------------------------
# PaD-TS imports
# -------------------------------------------------

def import_padts(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)

    from Model import PaD_TS
    from diffmodel_init import create_gaussian_diffusion
    from training import Trainer
    from resample import UniformSampler, Batch_Same_Sampler

    return PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler


# -------------------------------------------------
# diffusion imputation by projection
# -------------------------------------------------

@torch.no_grad()
def diffusion_impute_project(model, diffusion, x0_blk: torch.Tensor, keep_blk: torch.Tensor):
    device = x0_blk.device
    B, L, K = x0_blk.shape

    x_t = torch.randn_like(x0_blk)
    eps_obs = torch.randn_like(x0_blk)

    T = diffusion.num_timesteps

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        out = diffusion.p_sample(model, x_t, t_tensor, clip_denoised=True)
        x_prev = out["sample"]

        if t > 0:
            t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
            x_obs_prev = diffusion.q_sample(x0_blk, t_prev, noise=eps_obs)
        else:
            x_obs_prev = x0_blk

        x_t = keep_blk * x_obs_prev + (1.0 - keep_blk) * x_prev

    return x_t


# -------------------------------------------------
# evaluation
# -------------------------------------------------

@torch.no_grad()
def evaluate_split(
    model,
    diffusion,
    windows_nlk: np.ndarray,
    ratios,
    lm: float,
    device,
    batch_size: int,
    use_shared_evalmask: bool = False,
    shared_evalmask_dir: str = None,
    seed: int = None,
    split_name: str = "test",
):
    ds = WindowDataset(windows_nlk)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    results = []

    shared_masks = {}
    if use_shared_evalmask:
        for r in ratios:
            keep = load_shared_keepmask(shared_evalmask_dir, split_name, r, seed)
            if keep.shape != windows_nlk.shape:
                raise ValueError(
                    f"Shared keepmask shape mismatch for {split_name}, r={r:.2f}: "
                    f"{keep.shape} vs expected {windows_nlk.shape}"
                )
            shared_masks[r] = keep

    for r in ratios:
        sum_abs = 0.0
        sum_sq = 0.0
        sum_den = 0.0
        global_idx = 0

        for x in loader:
            x = x.to(device)
            B, L, K = x.shape

            if use_shared_evalmask:
                keep = torch.from_numpy(shared_masks[r][global_idx:global_idx + B]).to(device)
            else:
                keep = markov_keep_mask(B, K, L, r, lm, device).permute(0, 2, 1)

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
        r_obs = 1.0 - r

        results.append((r, r_obs, mae, mse, rmse))

    return results


@torch.no_grad()
def collect_test_arrays(
    model,
    diffusion,
    windows_nlk: np.ndarray,
    r_m: float,
    lm: float,
    device,
    batch_size: int,
    use_shared_evalmask: bool = False,
    shared_evalmask_dir: str = None,
    seed: int = None,
):
    ds = WindowDataset(windows_nlk)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    gt_all, imp_all, cond_all, eval_all = [], [], [], []

    keep_all = None
    if use_shared_evalmask:
        keep_all = load_shared_keepmask(shared_evalmask_dir, "test", r_m, seed)
        if keep_all.shape != windows_nlk.shape:
            raise ValueError(
                f"Shared keepmask shape mismatch for test, r={r_m:.2f}: "
                f"{keep_all.shape} vs expected {windows_nlk.shape}"
            )

    global_idx = 0
    for x in loader:
        x = x.to(device)
        B, L, K = x.shape

        if use_shared_evalmask:
            keep = torch.from_numpy(keep_all[global_idx:global_idx + B]).to(device)
        else:
            keep = markov_keep_mask(B, K, L, r_m, lm, device).permute(0, 2, 1)

        xhat = diffusion_impute_project(model, diffusion, x, keep)
        evalmask = 1.0 - keep
        imputed_full = keep * x + evalmask * xhat

        gt_all.append(x.detach().cpu().numpy())
        imp_all.append(imputed_full.detach().cpu().numpy())
        cond_all.append(keep.detach().cpu().numpy())
        eval_all.append(evalmask.detach().cpu().numpy())

        global_idx += B

    return (
        np.concatenate(gt_all, axis=0),
        np.concatenate(imp_all, axis=0),
        np.concatenate(cond_all, axis=0),
        np.concatenate(eval_all, axis=0),
    )


# -------------------------------------------------
# main
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--padts_root", type=str, required=True)
    ap.add_argument("--shared_data_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # PaD-TS model args
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--n_encoder", type=int, default=1)
    ap.add_argument("--n_decoder", type=int, default=3)
    ap.add_argument("--feature_last", action="store_true", default=True)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)

    # diffusion args
    ap.add_argument("--diffusion_steps", type=int, default=250)
    ap.add_argument("--noise_schedule", type=str, default="cosine")
    ap.add_argument("--loss", type=str, default="MSE_MMD")
    ap.add_argument("--predict_xstart", action="store_true", default=True)
    ap.add_argument("--rescale_timesteps", action="store_true", default=False)

    # trainer args
    ap.add_argument("--schedule_sampler", type=str, default="batch", choices=["batch", "uniform"])
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--mmd_alpha", type=float, default=0.0005)
    ap.add_argument("--save_dir", type=str, default="OUTPUT/padts_beijing/")

    # evaluation
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # optional shared eval maskbank
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default=None)

    # optional saved arrays
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_test_arrays_dir", type=str, default="saved_test_arrays_padts_beijing")

    # output
    ap.add_argument("--out_txt", type=str, default="padts_beijing_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ratios = parse_ratios(args.eval_ratios)

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    print("Loading Beijing shared dataset...")
    train, val, test = load_shared_windows(args.shared_data_dir)

    if len(train) == 0:
        raise ValueError("No training windows found.")
    if len(val) == 0:
        raise ValueError("No validation windows found.")
    if len(test) == 0:
        raise ValueError("No test windows found.")

    L = train.shape[1]
    K = train.shape[2]
    if L != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={L}, but args.seq_len={args.seq_len}")

    print(f"[Beijing] train/val/test windows = {len(train)}/{len(val)}/{len(test)}")
    print(f"[Beijing] sequence length = {L}, features = {K}")

    meta_path = os.path.join(args.shared_data_dir, "split_meta.json")
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "train_stations" in meta and "val_stations" in meta and "test_stations" in meta:
                print(f"[split] train_stations={meta['train_stations']}")
                print(f"[split] val_stations={meta['val_stations']}")
                print(f"[split] test_stations={meta['test_stations']}")
        except Exception as e:
            print(f"[warn] Could not read split_meta.json: {e}")

    print("Importing PaD-TS...")
    PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler = import_padts(args.padts_root)

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

    train_bs = safe_batch_size(args.batch, len(train))
    train_loader = DataLoader(
        WindowDataset(train),
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[Beijing] train batch size = {train_bs}")
    print(f"[Beijing] train batches = {len(train_loader)}")

    if args.schedule_sampler == "batch":
        schedule_sampler = Batch_Same_Sampler(diffusion)
    else:
        schedule_sampler = UniformSampler(diffusion)

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=train_bs,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.train_steps,
        save_dir=args.save_dir if args.save_dir.endswith("/") else args.save_dir + "/",
        mmd_alpha=args.mmd_alpha,
    )

    print("====== Training ======")
    trainer.train()
    print("====== Training done ======")

    model.eval()
    model.to(device)

    print("====== Evaluation ======")
    val_res = evaluate_split(
        model=model,
        diffusion=diffusion,
        windows_nlk=val,
        ratios=ratios,
        lm=args.lm,
        device=device,
        batch_size=train_bs,
        use_shared_evalmask=args.use_shared_evalmask,
        shared_evalmask_dir=args.shared_evalmask_dir,
        seed=args.seed,
        split_name="val",
    )

    test_res = evaluate_split(
        model=model,
        diffusion=diffusion,
        windows_nlk=test,
        ratios=ratios,
        lm=args.lm,
        device=device,
        batch_size=train_bs,
        use_shared_evalmask=args.use_shared_evalmask,
        shared_evalmask_dir=args.shared_evalmask_dir,
        seed=args.seed,
        split_name="test",
    )

    header_needed = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if header_needed:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for r, r_obs, mae, mse, rmse in val_res:
            f.write(f"val\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        for r, r_obs, mae, mse, rmse in test_res:
            f.write(f"test\t{r:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"[saved] metrics -> {args.out_txt}")

    if args.save_test_arrays:
        os.makedirs(args.save_test_arrays_dir, exist_ok=True)
        for r in ratios:
            gt, imp, cond, evalm = collect_test_arrays(
                model=model,
                diffusion=diffusion,
                windows_nlk=test,
                r_m=r,
                lm=args.lm,
                device=device,
                batch_size=train_bs,
                use_shared_evalmask=args.use_shared_evalmask,
                shared_evalmask_dir=args.shared_evalmask_dir,
                seed=args.seed,
            )

            tag = f"padts_beijing_test_r{r:.2f}_seed{args.seed}_L{args.seq_len}"
            np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_gt.npy"), gt)
            np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_imputed.npy"), imp)
            np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_condmask.npy"), cond)
            np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_evalmask.npy"), evalm)

        meta = {
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_data_dir": args.shared_data_dir,
            "lm": float(args.lm),
        }
        with open(os.path.join(args.save_test_arrays_dir, f"metadata_seed{args.seed}.json"), "w") as f:
            import json
            json.dump(meta, f, indent=2)

        print(f"[saved] test arrays -> {args.save_test_arrays_dir}")


if __name__ == "__main__":
    main()
