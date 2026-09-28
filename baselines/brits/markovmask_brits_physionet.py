
#!/usr/bin/env python3

import os, sys, time, random, argparse
from typing import Optional, Tuple, List, Dict, Set

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
    Returns (B,K,L) keepmask with 1=kept, 0=masked.
    r_masked is fraction masked (missingness).
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm                        # P(0->1)
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)   # P(1->0) so stationary keep ratio = r_keep
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
# PhysioNet helpers 
# -----------------------------
def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    pid_col = "Patient_ID"
    if pid_col not in df.columns:
        raise ValueError(f"{pid_col} column not found in CSV.")

    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce")
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)

    # common aliases (harmless)
    for _alias in ["patient_id", "PatientID", "patient", "RecordID", "id", "UID", "stay_id"]:
        if _alias not in df.columns:
            df[_alias] = df[pid_col]
    return df


def get_time_col(df: pd.DataFrame) -> str:
    if "ICULOS" in df.columns:
        return "ICULOS"
    if "Hour" in df.columns:
        return "Hour"
    raise ValueError("Expected a time column ICULOS or Hour in physionet2019.csv.")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {"Patient_ID"}
    for c in ["SepsisLabel", "Hour", "ICULOS", "HospAdmTime"]:
        if c in df.columns:
            drop_cols.add(c)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found after dropping id/labels/time columns.")
    return feat_cols


def split_patients(df: pd.DataFrame, seed: int, train_ratio=0.7, val_ratio=0.15) -> Tuple[Set[int], Set[int], Set[int]]:
    pids = df["Patient_ID"].dropna().unique().tolist()
    pids = [int(x) for x in pids]
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(pids[:n_train])
    val_p = set(pids[n_train:n_train + n_val])
    test_p = set(pids[n_train + n_val:])
    return train_p, val_p, test_p


def make_windows_for_patients(df: pd.DataFrame, patient_ids: Set[int], feat_cols: List[str], time_col: str, seq_len: int) -> np.ndarray:
    patient_ids = set(int(x) for x in patient_ids)
    windows = []
    # keep group order stable by using sort=False
    for pid, g in df.groupby("Patient_ID", sort=False):
        pid = int(pid)
        if pid not in patient_ids:
            continue
        g = g.sort_values(time_col, kind="mergesort")
        X = g[feat_cols].astype(np.float32).ffill().bfill()
        arr = X.to_numpy()
        T, K = arr.shape
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, K))
    if not windows:
        raise ValueError("No windows formed for this split (check seq_len or patient IDs).")
    return np.concatenate(windows, axis=0).astype(np.float32)


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    """
    - compute TRAIN nanmean/nanstd
    - drop features that are non-finite or ~zero-std in TRAIN
    - fill remaining NaNs using TRAIN mean
    - z-score with TRAIN mean/std
    """
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All features are non-finite / zero-std after TRAIN stats. Check CSV.")

    train = train[:, :, valid]
    val = val[:, :, valid]
    test = test[:, :, valid]
    mu = mu[valid].astype(np.float32)
    sd = np.maximum(sd[valid], eps).astype(np.float32)

    def fill_and_z(x):
        x = x.copy().astype(np.float32)
        nanmask = np.isnan(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = np.take(mu, ks)
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


# -----------------------------
# Maskbank metadata (Option A)
# -----------------------------
def load_maskbank_metadata(mask_dir: str, seq_len: int, seed: int) -> Dict[str, np.ndarray]:
    meta_path = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Maskbank metadata not found: {meta_path}")
    z = np.load(meta_path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def standardize_with_maskbank(train: np.ndarray, val: np.ndarray, test: np.ndarray, meta: Dict[str, np.ndarray]):
    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)
    eps = float(meta["eps"]) if "eps" in meta else 1e-6

    if train.shape[-1] != len(valid):
        raise ValueError(
            f"[maskbank] K_raw mismatch: data K={train.shape[-1]} but valid_feature_mask len={len(valid)}. "
            f"Ensure you used maskbank feature_cols_raw ordering."
        )

    mu = mu_raw[valid]
    sd = np.maximum(sd_raw[valid], eps)

    def fill_and_z(x):
        x = x[:, :, valid].copy().astype(np.float32)
        nanmask = np.isnan(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = np.take(mu, ks)
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization (maskbank scaling).")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu.astype(np.float32), sd.astype(np.float32), valid


def load_shared_keepmask(
    mask_dir: str,
    split: str,
    r_m: float,
    seq_len: int,
    seed: int,
    invert: bool,
    *,
    valid_feature_mask: Optional[np.ndarray] = None,
    expected_K: Optional[int] = None,
) -> np.ndarray:
    """
    physionet_{split}_keepmask_r{r:.2f}_L{seq_len}_seed{seed}.npy  with shape (N,L,K)
    Handles K alignment:
      - if keep.K == expected_K -> ok
      - else if keep.K == len(valid_feature_mask) -> slice keep[..., valid_feature_mask]
      - else raise
    """
    path = os.path.join(mask_dir, f"physionet_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must have shape (N,L,K), got {keep.shape} from {path}")
    if invert or os.environ.get("PHYSIONET_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep

    if expected_K is None:
        return keep
    if int(keep.shape[2]) == int(expected_K):
        return keep

    if valid_feature_mask is not None:
        valid_feature_mask = np.asarray(valid_feature_mask).astype(bool)
        K_full = int(len(valid_feature_mask))
        K_kept = int(valid_feature_mask.sum())
        if int(expected_K) != K_kept:
            raise ValueError(f"Internal mismatch: expected_K={expected_K} but valid_feature_mask.sum()={K_kept}.")
        if int(keep.shape[2]) == K_full:
            keep2 = keep[:, :, valid_feature_mask]
            if int(keep2.shape[2]) != int(expected_K):
                raise ValueError(f"After slicing keepmask K={keep2.shape[2]} != expected_K={expected_K}. File: {path}")
            return keep2

    raise ValueError(
        f"Shared keepmask K mismatch: maskbank K={keep.shape[2]} but model expects K={expected_K}. "
        f"File: {path}. Fix: regenerate maskbank with matching preprocessing, or store masks in full feature space."
    )


# -----------------------------
# Dataset
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)
    def __len__(self):
        return int(self.x.shape[0])
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i])  # (L,K)


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
    """
    Expects:
      {spin_root}/spin/baselines/brits/{layers.py,brits.py}
    """
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
# BRITS wrapper: train + eval
# -----------------------------
class BRITSWrapper(nn.Module):
    def __init__(self, BritsCls, K: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.K = int(K)
        self.device = device
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
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)

    # training
    ap.add_argument("--batch", type=int, default=8)
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
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_physionet_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # saving/logging
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_brits_physionet")
    ap.add_argument("--out_txt", type=str, default="brits_physionet_metrics.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    BritsCls = import_BRITS(args.spin_root)

    # ----- data load + split -----
    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)

    mask_meta = None
    if args.use_shared_evalmask:
        mask_meta = load_maskbank_metadata(args.shared_evalmask_dir, args.seq_len, args.seed)
        feat_cols = [str(c) for c in mask_meta["feature_cols_raw"].tolist()]
        train_ids = set(int(x) for x in mask_meta["patients_train"].tolist())
        val_ids   = set(int(x) for x in mask_meta["patients_val"].tolist())
        test_ids  = set(int(x) for x in mask_meta["patients_test"].tolist())

        # sanity: ensure ids exist in CSV
        df_pids = set(int(x) for x in df["Patient_ID"].dropna().unique().tolist())
        missing = (train_ids | val_ids | test_ids) - df_pids
        if missing:
            raise ValueError(f"Maskbank metadata contains Patient_IDs not found in CSV: {sorted(list(missing))[:10]} (showing up to 10)")
    else:
        feat_cols = get_feature_cols(df)
        train_ids, val_ids, test_ids = split_patients(df, seed=args.seed)

    train_raw = make_windows_for_patients(df, train_ids, feat_cols, time_col, args.seq_len)
    val_raw   = make_windows_for_patients(df, val_ids, feat_cols, time_col, args.seq_len)
    test_raw  = make_windows_for_patients(df, test_ids, feat_cols, time_col, args.seq_len)

    if args.use_shared_evalmask:
        train_w, val_w, test_w, mu, sd, valid = standardize_with_maskbank(train_raw, val_raw, test_raw, mask_meta)
    else:
        train_w, val_w, test_w, mu, sd, valid = standardize_by_train(train_raw, val_raw, test_raw)

    if len(train_w) == 0 or len(val_w) == 0 or len(test_w) == 0:
        raise ValueError("One of the splits has 0 windows. Check seq_len or dataset length.")

    K = int(train_w.shape[-1])
    print(f"[data] patients train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)}  windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K}  L={args.seq_len}")

    train_loader = DataLoader(WindowDataset(train_w), batch_size=safe_batch_size(args.batch, len(train_w)), shuffle=True, drop_last=True)
    val_loader   = DataLoader(WindowDataset(val_w),   batch_size=safe_batch_size(args.batch, len(val_w)),   shuffle=False, drop_last=False)
    test_loader  = DataLoader(WindowDataset(test_w),  batch_size=safe_batch_size(args.batch, len(test_w)),  shuffle=False, drop_last=False)

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
                    args.shared_evalmask_dir, split=split_name, r_m=r_m, seq_len=args.seq_len, seed=args.seed,
                    invert=args.invert_shared_keepmask, valid_feature_mask=valid, expected_K=K
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

    # ----- save arrays -----
    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)
        ns = 1
        for r_m in ratios:
            gt_all, imp_all, cond_all, eval_all = [], [], [], []

            keep_NLK = None
            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir, split="test", r_m=r_m, seq_len=args.seq_len, seed=args.seed,
                    invert=args.invert_shared_keepmask, valid_feature_mask=valid, expected_K=K
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

            tag = f"brits_physionet_test_r{r_m:.2f}_seed{args.seed}_ns{ns}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)
            print(f"[save] {tag}_*.npy  shapes gt={gt_all.shape} imputed={imp_all.shape} cond={cond_all.shape} eval={eval_all.shape}")

        np.savez(
            os.path.join(args.save_dir, f"metadata_seed{args.seed}.npz"),
            mu=mu, sd=sd, seq_len=int(args.seq_len),
            shared_evalmask=bool(args.use_shared_evalmask),
            invert_shared_keepmask=bool(args.invert_shared_keepmask),
            n_patients_train=int(len(train_ids)), n_patients_val=int(len(val_ids)), n_patients_test=int(len(test_ids)),
        )
        print(f"[save] Test arrays saved under: {args.save_dir}")

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
