#!/usr/bin/env python3

import os, re, sys, time, random, argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import importlib.util
import types


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
    Returns (B,K,L) keepmask with 1=kept, 0=masked, with segmenty Markov behavior.
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
# Gait data helpers 
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
        raise ValueError(f"Could not parse user ID from filename: {os.path.basename(path)}")
    return m.group(1)


def read_gait_csv(path: str) -> np.ndarray:
    # automatic-ou-gaitdata: csv with two header rows
    df = pd.read_csv(path, skiprows=2, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad array shape from {path}: {arr.shape}")
    return arr


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
    return np.concatenate(windows, axis=0).astype(np.float32)


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


class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)
    def __len__(self):
        return int(self.x.shape[0])
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


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


# ---------------------------------------------------------------------
# BRITS loader (Option B): bypass spin.baselines.__init__.py and stub tsl
# ---------------------------------------------------------------------
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


# -----------------------------
# BRITS wrapper: train + eval (masked-point metrics)
# -----------------------------
class BRITSWrapper(nn.Module):
    def __init__(self, BritsCls, K: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.K = int(K)
        self.device = device
        # single-node multivariate series: n_nodes=1, input_size=K
        self.model = BritsCls(input_size=self.K, n_nodes=1, hidden_size=hidden_size).to(device)

    def forward_impute(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor):
        x_obs = torch.where(keep_BLK > 0, x_BLK, torch.zeros_like(x_BLK))
        x_in = x_obs.unsqueeze(2)     # (B,L,1,K)
        m_in = keep_BLK.unsqueeze(2)  # (B,L,1,K)

        imputation, predictions = self.model(x_in, mask=m_in)
        imp_fwd = predictions[0].squeeze(2)
        imp_bwd = predictions[1].squeeze(2)
        imp = imputation.squeeze(2)
        return imp, imp_fwd, imp_bwd

    def train_loss(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor, lambda_consistency: float = 1.0) -> torch.Tensor:
        imp, imp_fwd, imp_bwd = self.forward_impute(x_BLK, keep_BLK)
        target_mask = 1.0 - keep_BLK
        denom = target_mask.sum().clamp_min(1.0)
        rec = (torch.abs(imp - x_BLK) * target_mask).sum() / denom
        cons = self.model.consistency_loss(imp_fwd.unsqueeze(2), imp_bwd.unsqueeze(2))
        return rec + lambda_consistency * cons

    @torch.no_grad()
    def eval_totals(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor):
        imp, _, _ = self.forward_impute(x_BLK, keep_BLK)
        evalmask = 1.0 - keep_BLK
        diff = (imp - x_BLK) * evalmask
        denom = evalmask.sum().clamp_min(1.0)
        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()
        r_obs = float(keep_BLK.mean().item())
        return float(sum_abs.item()), float(sum_sq.item()), float(denom.item()), r_obs

    @torch.no_grad()
    def impute_and_collect(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor):
        imp, _, _ = self.forward_impute(x_BLK, keep_BLK)
        imputed_full = keep_BLK * x_BLK + (1.0 - keep_BLK) * imp
        evalmask = 1.0 - keep_BLK
        return (
            x_BLK.detach().cpu().numpy(),
            imputed_full.detach().cpu().numpy(),
            keep_BLK.detach().cpu().numpy(),
            evalmask.detach().cpu().numpy(),
        )


def main():
    ap = argparse.ArgumentParser()

    # repo
    ap.add_argument("--spin_root", type=str, required=True)

    # data
    ap.add_argument("--data_dir", type=str, required=True,
                    help="automatic-ou-gaitdata directory containing per-trial CSVs")
    ap.add_argument("--seq_len", type=int, default=64)

    # training
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")

    # shared masks
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_gait_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # saving/logging
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_brits_gait")
    ap.add_argument("--out_txt", type=str, default="brits_gait_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    BritsCls = import_BRITS(args.spin_root)

    # ----- split by user -----
    files = list_gait_files(args.data_dir)
    tr_files, va_files, te_files, (tr_users, va_users, te_users) = split_users(files, seed=args.seed)

    train_raw = windows_from_files(tr_files, args.seq_len)
    val_raw   = windows_from_files(va_files, args.seq_len)
    test_raw  = windows_from_files(te_files, args.seq_len)

    train_w, val_w, test_w, mu, sd = standardize_by_train(train_raw, val_raw, test_raw)

    if len(train_w) == 0 or len(val_w) == 0 or len(test_w) == 0:
        raise ValueError("One of the splits has 0 windows. Check seq_len or dataset length.")

    K = int(train_w.shape[-1])
    print(f"[data] #users train/val/test={len(tr_users)}/{len(va_users)}/{len(te_users)}  #windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={args.seq_len}")

    train_loader = DataLoader(WindowedDataset(train_w), batch_size=safe_batch_size(args.batch, len(train_w)), shuffle=True, drop_last=True)
    val_loader   = DataLoader(WindowedDataset(val_w),   batch_size=safe_batch_size(args.batch, len(val_w)),   shuffle=False, drop_last=False)
    test_loader  = DataLoader(WindowedDataset(test_w),  batch_size=safe_batch_size(args.batch, len(test_w)),  shuffle=False, drop_last=False)

    # ----- model -----
    wrapper = BRITSWrapper(BritsCls=BritsCls, K=K, hidden_size=args.hidden_size, device=device).to(device)
    opt = torch.optim.Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-6)

    # ----- train -----
    wrapper.train()
    for ep in range(1, args.epochs + 1):
        losses = []
        t0 = time.time()
        for xb in train_loader:
            xb = xb.to(device).float()
            keep_BKL = markov_keep_mask_from_masked_ratio(
                B=xb.shape[0], K=K, L=args.seq_len, r_masked=args.r_train_masked, lm=args.lm, device=device
            )
            keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

            opt.zero_grad(set_to_none=True)
            loss = wrapper.train_loss(xb, keep_BLK, lambda_consistency=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())

        if ep == 1 or ep % 10 == 0:
            print(f"[train] epoch={ep:03d} loss={float(np.mean(losses)):.6f}  time={time.time()-t0:.1f}s")

    # ----- eval -----
    wrapper.eval()
    ratios = parse_ratios(args.eval_masked_ratios)

    def eval_split(loader: DataLoader, split_name: str, n_expected: int):
        rows = []
        for r_m in ratios:
            total_abs = 0.0
            total_sq = 0.0
            total_den = 0.0
            total_robs = 0.0
            total_batches = 0

            keep_NLK = None
            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir, split=split_name, r_m=r_m,
                    seq_len=args.seq_len, seed=args.seed, invert=args.invert_shared_keepmask
                )
                if int(keep_NLK.shape[0]) != int(n_expected):
                    raise ValueError(f"Keepmask N mismatch for split={split_name} r={r_m:.2f}: keepN={keep_NLK.shape[0]} windows={n_expected}")

            offset = 0
            for xb in loader:
                xb = xb.to(device).float()
                B = xb.shape[0]
                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device).float()
                    offset += B
                else:
                    keep_BKL = markov_keep_mask_from_masked_ratio(B, K, args.seq_len, r_m, args.lm, device)
                    keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

                sum_abs, sum_sq, denom, r_obs = wrapper.eval_totals(xb, keep_BLK)
                total_abs += sum_abs
                total_sq += sum_sq
                total_den += denom
                total_robs += r_obs
                total_batches += 1

            total_den = max(total_den, 1.0)
            mae = total_abs / total_den
            mse = total_sq / total_den
            rmse = float(np.sqrt(mse))
            r_obs_mean = total_robs / max(total_batches, 1)

            print(f"[{split_name}] r_masked={r_m:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  r_obs={r_obs_mean:.2f}")
            rows.append((split_name, r_m, float(r_obs_mean), float(mae), float(mse), float(rmse)))
        return rows

    val_rows = eval_split(val_loader, "val", len(val_w))
    test_rows = eval_split(test_loader, "test", len(test_w))

    # ----- save arrays (test) -----
    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)
        ns = 1
        for r_m in ratios:
            gt_all, imp_all, cond_all, eval_all = [], [], [], []

            keep_NLK = None
            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir, split="test", r_m=r_m,
                    seq_len=args.seq_len, seed=args.seed, invert=args.invert_shared_keepmask
                )

            offset = 0
            for xb in test_loader:
                xb = xb.to(device).float()
                B = xb.shape[0]
                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device).float()
                    offset += B
                else:
                    keep_BKL = markov_keep_mask_from_masked_ratio(B, K, args.seq_len, r_m, args.lm, device)
                    keep_BLK = keep_BKL.permute(0, 2, 1).contiguous()

                gt_b, imp_b, cond_b, eval_b = wrapper.impute_and_collect(xb, keep_BLK)
                gt_all.append(gt_b); imp_all.append(imp_b); cond_all.append(cond_b); eval_all.append(eval_b)

            gt_all = np.concatenate(gt_all, axis=0)
            imp_all = np.concatenate(imp_all, axis=0)
            cond_all = np.concatenate(cond_all, axis=0)
            eval_all = np.concatenate(eval_all, axis=0)

            tag = f"brits_gait_test_r{r_m:.2f}_seed{args.seed}_ns{ns}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)
            print(f"[save] {tag}_*.npy  shapes gt={gt_all.shape} imputed={imp_all.shape}")

        np.savez(
            os.path.join(args.save_dir, f"metadata_seed{args.seed}.npz"),
            mu=mu, sd=sd, seq_len=int(args.seq_len),
            shared_evalmask=bool(args.use_shared_evalmask),
            invert_shared_keepmask=bool(args.invert_shared_keepmask),
            users_train=np.array(tr_users, dtype=object),
            users_val=np.array(va_users, dtype=object),
            users_test=np.array(te_users, dtype=object),
        )

    # ----- write metrics -----
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split, r_m, r_obs, mae, mse, rmse in val_rows + test_rows:
            f.write(f"{split}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
