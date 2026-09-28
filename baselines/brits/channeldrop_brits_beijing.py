import os
import sys
import json
import time
import random
import argparse
import importlib.util
import types
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATASET = "beijing"
PREFIX = "beijing"
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH = 64


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
    train_w = np.load(os.path.join(asset_dir, f"{PREFIX}_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"{PREFIX}_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"{PREFIX}_seq{seq_len}_test_windows.npy")).astype(np.float32)
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
            assert self.evalmask is not None and self.evalmask.shape == self.x.shape,                 f"eval mask mismatch: evalmask={None if self.evalmask is None else self.evalmask.shape} x={self.x.shape}"

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.obs[idx])


def make_train_keep_mask(obs_BLK: torch.Tensor, masked_ratio: float) -> torch.Tensor:
    if masked_ratio <= 0:
        return obs_BLK.clone()
    rand = torch.rand_like(obs_BLK)
    keep = (rand > masked_ratio).float()
    keep = keep * obs_BLK
    return keep


def _load_module(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_tsl_stub():
    if "tsl" not in sys.modules:
        tsl_pkg = types.ModuleType("tsl")
        tsl_pkg.__path__ = []
        sys.modules["tsl"] = tsl_pkg
    if "tsl.nn" not in sys.modules:
        tsl_nn = types.ModuleType("tsl.nn")
        tsl_nn.__path__ = []
        sys.modules["tsl.nn"] = tsl_nn
    if "tsl.nn.functional" not in sys.modules:
        tsl_f = types.ModuleType("tsl.nn.functional")
        def reverse_tensor(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
            return torch.flip(x, dims=[dim])
        tsl_f.reverse_tensor = reverse_tensor
        sys.modules["tsl.nn.functional"] = tsl_f


def import_BRITS(spin_root: str):
    spin_root = os.path.abspath(spin_root)
    spin_dir = os.path.join(spin_root, "spin")
    baselines_dir = os.path.join(spin_dir, "baselines")
    brits_dir = os.path.join(baselines_dir, "brits")
    layers_path = os.path.join(brits_dir, "layers.py")
    brits_path = os.path.join(brits_dir, "brits.py")

    for p in [spin_root, spin_dir, baselines_dir, brits_dir, layers_path, brits_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing path: {p}")

    if spin_root not in sys.path:
        sys.path.insert(0, spin_root)

    if "spin" not in sys.modules:
        spin_pkg = types.ModuleType("spin")
        spin_pkg.__path__ = [spin_dir]
        sys.modules["spin"] = spin_pkg
    if "spin.baselines" not in sys.modules:
        baselines_pkg = types.ModuleType("spin.baselines")
        baselines_pkg.__path__ = [baselines_dir]
        sys.modules["spin.baselines"] = baselines_pkg
    if "spin.baselines.brits" not in sys.modules:
        brits_pkg = types.ModuleType("spin.baselines.brits")
        brits_pkg.__path__ = [brits_dir]
        sys.modules["spin.baselines.brits"] = brits_pkg

    _ensure_tsl_stub()
    _load_module("spin.baselines.brits.layers", layers_path)
    brits_mod = _load_module("spin.baselines.brits.brits", brits_path)
    if not hasattr(brits_mod, "BRITS"):
        raise AttributeError(f"{brits_path} loaded, but no BRITS class found.")
    print(f"[import] BRITS loaded from: {brits_path}")
    return brits_mod.BRITS


class BRITSWrapper(nn.Module):
    def __init__(self, BritsCls, K: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.K = int(K)
        self.device = device
        self.model = BritsCls(input_size=self.K, n_nodes=1, hidden_size=hidden_size).to(device)

    def forward_impute(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor):
        x_obs = torch.where(keep_BLK > 0, x_BLK, torch.zeros_like(x_BLK))
        x_in = x_obs.unsqueeze(2)
        m_in = keep_BLK.unsqueeze(2)
        imputation, predictions = self.model(x_in, mask=m_in)
        imp_fwd = predictions[0].squeeze(2)
        imp_bwd = predictions[1].squeeze(2)
        imp = imputation.squeeze(2)
        return imp, imp_fwd, imp_bwd

    def train_loss(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor, lambda_consistency: float = 1.0) -> torch.Tensor:
        imp, imp_fwd, imp_bwd = self.forward_impute(x_BLK, keep_BLK)
        target_mask = (1.0 - keep_BLK)
        denom = target_mask.sum().clamp_min(1.0)
        rec = (torch.abs(imp - x_BLK) * target_mask).sum() / denom
        cons = self.model.consistency_loss(imp_fwd.unsqueeze(2), imp_bwd.unsqueeze(2))
        return rec + lambda_consistency * cons

    @torch.no_grad()
    def eval_totals(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor, obs_BLK: torch.Tensor):
        imp, _, _ = self.forward_impute(x_BLK, keep_BLK)
        evalmask = (1.0 - keep_BLK) * obs_BLK
        diff = (imp - x_BLK) * evalmask
        denom = evalmask.sum().clamp_min(1.0)
        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()
        r_obs = float(keep_BLK.sum().item() / obs_BLK.sum().clamp_min(1.0).item())
        return float(sum_abs.item()), float(sum_sq.item()), float(denom.item()), r_obs


@torch.no_grad()
def evaluate_loader(wrapper: BRITSWrapper, loader: DataLoader, evalmask_NLK: np.ndarray, device: torch.device):
    wrapper.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_cnt = 0.0
    total_keep = 0.0
    total_obs = 0.0
    start = 0
    for xb, obsb in loader:
        xb = xb.to(device)
        obsb = obsb.to(device)
        B = xb.shape[0]
        this_evalmask = torch.from_numpy(evalmask_NLK[start:start+B]).to(device).float()
        start += B
        keepb = obsb * (1.0 - this_evalmask)
        s_abs, s_sq, s_cnt, _ = wrapper.eval_totals(xb, keepb, obsb)
        total_abs += s_abs
        total_sq += s_sq
        total_cnt += s_cnt
        total_keep += keepb.sum().item()
        total_obs += obsb.sum().item()
    total_cnt = max(total_cnt, 1.0)
    mse = total_sq / total_cnt
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_cnt
    r_obs = total_keep / max(total_obs, 1.0)
    return float(mae), float(mse), float(rmse), float(r_obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spin_root", type=str, required=True)
    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--train_pointwise_mask_ratio", type=float, default=0.15)
    ap.add_argument("--lambda_consistency", type=float, default=1.0)
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--out_txt", type=str, default=f"brits_{DATASET}_channeldrop.txt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_w, val_w, test_w = load_window_assets(args.asset_dir, args.seq_len)
    K = train_w.shape[-1]
    val_evalmask, _, _ = load_maskbank_npz(args.val_maskbank)
    test_evalmask, _, _ = load_maskbank_npz(args.test_maskbank)

    assert val_evalmask.shape == val_w.shape, f"val mismatch: {val_evalmask.shape} vs {val_w.shape}"
    assert test_evalmask.shape == test_w.shape, f"test mismatch: {test_evalmask.shape} vs {test_w.shape}"

    print(f"[data] train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={args.seq_len}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    BritsCls = import_BRITS(args.spin_root)
    wrapper = BRITSWrapper(BritsCls, K=K, hidden_size=args.hidden_size, device=device).to(device)

    train_bs = safe_batch_size(args.batch, len(train_w))
    eval_bs = safe_batch_size(args.batch, max(len(val_w), len(test_w)))

    train_loader = DataLoader(
        WindowDataset(train_w, mode="train"),
        batch_size=train_bs,
        shuffle=True,
        drop_last=(len(train_w) >= train_bs and train_bs > 1),
    )
    val_loader = DataLoader(
        WindowDataset(val_w, mode="eval", precomputed_evalmask_NLK=val_evalmask),
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_w, mode="eval", precomputed_evalmask_NLK=test_evalmask),
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
    )

    opt = torch.optim.Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-6)

    for ep in range(1, args.epochs + 1):
        wrapper.train()
        losses = []
        skipped = 0
        t0 = time.time()
        for xb, obsb in train_loader:
            xb = xb.to(device)
            obsb = obsb.to(device)
            keepb = make_train_keep_mask(obsb, args.train_pointwise_mask_ratio)
            opt.zero_grad(set_to_none=True)
            loss = wrapper.train_loss(xb, keepb, lambda_consistency=args.lambda_consistency)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[epoch {ep:03d}] loss={mean_loss:.6f}  skipped={skipped}  time={time.time()-t0:.1f}s")

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE")
        for split_name, loader, evalmask in [("val", val_loader, val_evalmask), ("test", test_loader, test_evalmask)]:
            mae, mse, rmse, r_obs = evaluate_loader(wrapper, loader, evalmask, device)
            f.write(f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
            print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}\n")
    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()

