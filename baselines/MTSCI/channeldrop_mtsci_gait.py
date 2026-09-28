#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import pickle as pk
import argparse
import importlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam

DATASET = "gait"
DEFAULT_SEQ_LEN = 128
DEFAULT_BATCH = 8


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


def load_window_assets(asset_dir: str, seq_len: int):
    train_w = np.load(os.path.join(asset_dir, f"gait_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"gait_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"gait_seq{seq_len}_test_windows.npy")).astype(np.float32)
    if train_w.ndim != 3 or val_w.ndim != 3 or test_w.ndim != 3:
        raise ValueError(f"Expected (N,L,K) windows, got train={train_w.shape} val={val_w.shape} test={test_w.shape}")
    if train_w.shape[1] != seq_len or val_w.shape[1] != seq_len or test_w.shape[1] != seq_len:
        raise ValueError(f"seq_len mismatch: requested {seq_len}, got train={train_w.shape} val={val_w.shape} test={test_w.shape}")
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


def bootstrap_mtsci_code(mtsci_root: str):
    mtsci_root = os.path.abspath(mtsci_root)
    if mtsci_root not in sys.path:
        sys.path.insert(0, mtsci_root)
    try:
        dataloader_mod = importlib.import_module("dataloader.dataloader")
        models_mod = importlib.import_module("models.model")
    except Exception as e:
        raise ImportError(f"Failed to import MTSCI modules from {mtsci_root}: {e}")
    return dataloader_mod.generate_train_dataloader, dataloader_mod.generate_val_test_dataloader, models_mod.MTSCI


def make_mtsci_pickles_from_windows(asset_dir: str, out_dir: str, prefix: str, seq_len: int):
    os.makedirs(out_dir, exist_ok=True)
    train_w, val_w, test_w = load_window_assets(asset_dir, seq_len)
    Ntr, L, K = train_w.shape
    Nva, Nte = val_w.shape[0], test_w.shape[0]

    train_tk = train_w.reshape(-1, K).astype(np.float32)
    val_tk = val_w.reshape(-1, K).astype(np.float32)
    test_tk = test_w.reshape(-1, K).astype(np.float32)

    mu = np.nanmean(train_tk, axis=0).astype(np.float32)
    sd = np.nanstd(train_tk, axis=0).astype(np.float32)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.maximum(sd, 1e-6).astype(np.float32)

    pk.dump(train_tk, open(os.path.join(out_dir, "train_set.pkl"), "wb"))
    pk.dump(val_tk, open(os.path.join(out_dir, "val_set.pkl"), "wb"))
    pk.dump(test_tk, open(os.path.join(out_dir, "test_set.pkl"), "wb"))
    pk.dump((mu, sd), open(os.path.join(out_dir, "scaler.pkl"), "wb"))
    pk.dump({
        "dataset": prefix,
        "seq_len": int(seq_len),
        "n_windows_train": int(Ntr),
        "n_windows_val": int(Nva),
        "n_windows_test": int(Nte),
        "K": int(K),
        "asset_dir": asset_dir,
    }, open(os.path.join(out_dir, "metadata.pkl"), "wb"))

    print(f"[data] {prefix} windows train/val/test={Ntr}/{Nva}/{Nte} K={K} L={L}")
    return out_dir, K, Ntr, Nva, Nte


def override_test_batch_with_keepmask(batch, keep_blk: torch.Tensor, device: torch.device):
    X, mask, X_Tilde, gt_mask, indicating = batch
    X_Tilde = X_Tilde.to(device)
    gt_mask = gt_mask.to(device)
    keep_blk = keep_blk.to(device).float()
    indicating_new = (1.0 - keep_blk) * gt_mask
    X_new = X_Tilde * (1.0 - indicating_new)
    mask_new = gt_mask * (1.0 - indicating_new)
    return (X_new, mask_new, X_Tilde, gt_mask, indicating_new)


def masked_metrics_from_samples(samples_BnKL: torch.Tensor, X_true_BKL: torch.Tensor, eval_mask_BKL: torch.Tensor):
    pred = samples_BnKL.mean(dim=1)
    m = eval_mask_BKL
    denom = torch.clamp(m.sum(), min=1.0)
    mae = (torch.abs(pred - X_true_BKL) * m).sum() / denom
    mse = (((pred - X_true_BKL) ** 2) * m).sum() / denom
    rmse = torch.sqrt(mse + 1e-12)
    return mae.item(), mse.item(), rmse.item()


def append_row(out_txt: str, row: str):
    out_dir = os.path.dirname(out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = (not os.path.exists(out_txt)) or (os.path.getsize(out_txt) == 0)
    with open(out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")
        f.write(row)


def evaluate_split(model, loader_clean, evalmask_nlk: np.ndarray, protocol: str, device: torch.device, nsample: int):
    model.eval()
    all_mae, all_mse, all_rmse, all_r_obs = [], [], [], []
    offset = 0
    with torch.no_grad():
        for batch in loader_clean:
            batch = tuple(x.to(device) for x in batch)
            B = batch[2].shape[0]
            keep_blk = torch.from_numpy(1.0 - evalmask_nlk[offset:offset+B]).to(device).float()
            offset += B
            batch_m = override_test_batch_with_keepmask(batch, keep_blk, device=device)
            samples, X_Tilde_BKL, eval_mask_BKL, X_Tilde_mask_BKL, tp = model.evaluate(batch_m, n_samples=nsample)
            mae, mse, rmse = masked_metrics_from_samples(samples, X_Tilde_BKL, eval_mask_BKL)
            all_mae.append(mae); all_mse.append(mse); all_rmse.append(rmse)
            all_r_obs.append(float(keep_blk.mean().item()))
    mean_mae = float(np.mean(all_mae)) if all_mae else float("nan")
    mean_mse = float(np.mean(all_mse)) if all_mse else float("nan")
    mean_rmse = float(np.mean(all_rmse)) if all_rmse else float("nan")
    mean_r_obs = float(np.mean(all_r_obs)) if all_r_obs else float("nan")
    return mean_mae, mean_mse, mean_rmse, mean_r_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtsci_root", type=str, required=True)
    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--work_dir", type=str, default="")
    ap.add_argument("--out_txt", type=str, default="mtsci_gait_channeldrop.txt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    gen_train_loader, gen_val_test_loader, MTSCI = bootstrap_mtsci_code(args.mtsci_root)

    train_w, val_w, test_w = load_window_assets(args.asset_dir, args.seq_len)
    K = int(train_w.shape[-1])
    val_evalmask, _, _ = load_maskbank_npz(args.val_maskbank)
    test_evalmask, _, _ = load_maskbank_npz(args.test_maskbank)
    assert val_evalmask.shape == val_w.shape, f"val shape mismatch: {val_evalmask.shape} vs {val_w.shape}"
    assert test_evalmask.shape == test_w.shape, f"test shape mismatch: {test_evalmask.shape} vs {test_w.shape}"
    print(f"[data] train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)} K={K} L={args.seq_len}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    work_dir = args.work_dir or os.path.join(args.asset_dir, f"mtsci_gait_pickles_seed{args.seed}")
    dataset_dir, K2, _, _, _ = make_mtsci_pickles_from_windows(args.asset_dir, work_dir, "gait", args.seq_len)
    assert K2 == K

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

    train_loader = gen_train_loader(
        dataset_dir, seq_len=args.seq_len,
        missing_ratio=0.2, missing_pattern="block",
        batch_size=safe_batch_size(args.batch, len(train_w)), mode="train",
    )
    val_loader_clean = gen_val_test_loader(
        dataset_dir, seq_len=args.seq_len,
        missing_ratio=0.0, missing_pattern="point",
        batch_size=safe_batch_size(args.batch, len(val_w)), mode="val",
    )
    test_loader_clean = gen_val_test_loader(
        dataset_dir, seq_len=args.seq_len,
        missing_ratio=0.0, missing_pattern="point",
        batch_size=safe_batch_size(args.batch, len(test_w)), mode="test",
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
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[epoch {ep:03d}] loss={loss_sum/max(n_batches,1):.6f} time={time.time()-t0:.1f}s")

    for split_name, loader_clean, evalmask in [
        ("val", val_loader_clean, val_evalmask),
        ("test", test_loader_clean, test_evalmask),
    ]:
        mae, mse, rmse, r_obs = evaluate_split(model, loader_clean, evalmask, args.protocol, device, args.nsample)
        append_row(args.out_txt, f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
