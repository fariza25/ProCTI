#!/usr/bin/env python3


import os, sys, time, random, argparse, importlib.util, types
from typing import Optional, Tuple, List

import numpy as np


def _to_int_id(x):
    # robust conversion for patient IDs stored as int/float/str like "1.0"
    if x is None:
        raise ValueError("patient id is None")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    s = str(x).strip()
    try:
        return int(s)
    except ValueError:
        return int(float(s))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torchcde
except Exception as e:
    raise RuntimeError("torchcde is required: pip install torchcde") from e


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dynamic loader for codebase
# -----------------------------
def load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def bootstrap_fgti_code(base_dir: str):
    if "models" not in sys.modules:
        sys.modules["models"] = types.ModuleType("models")
    if "layers" not in sys.modules:
        sys.modules["layers"] = types.ModuleType("layers")

    load_module("layers.Embed", os.path.join(base_dir, "Embed.py"))
    load_module("layers.Diff_layers", os.path.join(base_dir, "Diff_layers.py"))
    load_module("models.ts_transformer", os.path.join(base_dir, "ts_transformer.py"))
    load_module("models.diffusion", os.path.join(base_dir, "diffusion.py"))
    main_model = load_module("models.main_model", os.path.join(base_dir, "main_model.py"))
    return main_model


# -----------------------------
# Markov keep-mask (segmenty; stationary keep ratio = 1 - r_masked)
# -----------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device) -> torch.Tensor:
    """
    Returns (B,K,L) keepmask with 1=kept, 0=masked.
    r_masked is fraction masked (missingness).
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
    p_m = 1.0 / lm                        # P(0->1)
    p_u = p_m * (1.0 - r_keep) / r_keep   # P(1->0) so stationary keep ratio = r_keep
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
# HF + Dominant (robust: masked -> 0 before torchcde)
# -----------------------------
@torch.no_grad()
def make_hf_and_dom_Astyle(
    data_BLK: torch.Tensor,
    keepmask_BLK: torch.Tensor,
    flimit: float,
    topf: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data_BLK: (B,L,K)
    keepmask_BLK: (B,L,K) with 1 keep
    returns hf/dom: (B,L,K)
    """
    B, L, K = data_BLK.shape
    x = torch.where(keepmask_BLK > 0, data_BLK, torch.zeros_like(data_BLK))
    coeffs = torchcde.linear_interpolation_coeffs(x)  # (B,L,K)
    maxdataf = coeffs.clone()

    freqs = torch.fft.rfftfreq(L, device=coeffs.device)

    # HF
    pass_f = torch.abs(freqs) > flimit
    hf_list = []
    for j in range(K):
        xj = coeffs[:, :, j]
        xf = torch.fft.rfft(xj, dim=1)
        rx = torch.fft.irfft(xf * pass_f, n=L, dim=1)
        hf_list.append(rx)
    hf = torch.stack(hf_list, dim=2)

    # Dominant
    dom_list = []
    for j in range(K):
        xj = maxdataf[:, :, j]
        xf = torch.fft.rfft(xj, dim=1)
        mag = torch.abs(xf)
        _, idx = torch.topk(mag, k=min(topf, mag.shape[1]), dim=1)
        keep = torch.zeros_like(xf, dtype=torch.bool)
        keep.scatter_(1, idx, True)
        xf2 = torch.where(keep, xf, torch.zeros_like(xf))
        rx = torch.fft.irfft(xf2, n=L, dim=1)
        dom_list.append(rx)
    dom = torch.stack(dom_list, dim=2)

    return hf, dom


# -----------------------------
# Prototype module + alpha
# -----------------------------
class GlobalProtoRegime(nn.Module):
    """
    Single regime embedding per window:
      tokens: [x, keepmask] -> D
      pool over (L*K) -> query
      attend to prototype bank -> ctx
      project -> r (B,K)
    """
    def __init__(self, proto_dim: int, proto_M: int, proto_heads: int, out_dim_K: int):
        super().__init__()
        self.proto_dim = proto_dim
        self.token_proj = nn.Linear(2, proto_dim)
        self.proto_bank = nn.Parameter(torch.randn(proto_M, proto_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=proto_dim, num_heads=proto_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(proto_dim)
        self.ln_kv = nn.LayerNorm(proto_dim)
        self.to_r = nn.Linear(proto_dim, out_dim_K)

    def forward(self, x_BLK: torch.Tensor, keep_mask_BLK: torch.Tensor) -> torch.Tensor:
        B, L, K = x_BLK.shape
        token_in = torch.stack([x_BLK, keep_mask_BLK], dim=-1)  # (B,L,K,2)
        token_emb = self.token_proj(token_in)                  # (B,L,K,D)
        q = token_emb.reshape(B, L * K, self.proto_dim).mean(dim=1, keepdim=True)
        q = self.ln_q(q)
        kv = self.ln_kv(self.proto_bank.unsqueeze(0).expand(B, -1, -1))
        ctx, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        return self.to_r(ctx.squeeze(1))                       # (B,K)


# -----------------------------
# PhysioNet 2019 data helpers (patient-wise split)
# -----------------------------
class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # ---- Normalize Patient_ID to int (sharedmask alignment) ----
    pid_col = "Patient_ID"
    if pid_col not in df.columns:
        raise ValueError(f"{pid_col} column not found in CSV.")
    df[pid_col] = pd.to_numeric(df[pid_col], errors='coerce')
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)
    # Add common aliases so windowing code still works if it expects a different name
    for _alias in ["patient_id", "PatientID", "patient", "RecordID", "id", "UID", "stay_id"]:
        if _alias not in df.columns:
            df[_alias] = df[pid_col]
    # ------------------------------------------------------------

    # normalize Patient_ID to int so it matches maskbank metadata IDs
    pid_col = "Patient_ID"
    df[pid_col] = pd.to_numeric(df[pid_col], errors='coerce')
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Patient_ID" not in df.columns:
        raise ValueError("physionet2019.csv must contain a Patient_ID column.")
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


def split_patients(df: pd.DataFrame, seed: int, train_ratio=0.7, val_ratio=0.15):
    pids = df["Patient_ID"].dropna().unique().tolist()
    pids = list(map(str, pids))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_p = set(pids[:n_train])
    val_p = set(pids[n_train:n_train + n_val])
    test_p = set(pids[n_train + n_val:])
    return train_p, val_p, test_p


def make_windows_for_patients(df, patient_ids_set, feat_cols, time_col, seq_len):
    patient_ids_set = set(int(x) for x in patient_ids_set)
    windows = []
    for pid, g in df.groupby("Patient_ID", sort=False):
        if int(pid) not in patient_ids_set:
            continue
        if time_col:
            g = g.sort_values(time_col, kind="mergesort")   # stable
        X = g[feat_cols].astype(np.float32).ffill().bfill()
        arr = X.to_numpy()
        T = arr.shape[0]
        n_win = T // seq_len
        if n_win <= 0:
            continue
        windows.append(arr[: n_win * seq_len].reshape(n_win, seq_len, arr.shape[1]))
    if not windows:
        raise ValueError("No windows formed for this split.")
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
        raise ValueError("All features are non-finite / zero-std after nanmean/nanstd on TRAIN. Check CSV.")

    train = train[:, :, valid]
    val   = val[:, :, valid]
    test  = test[:, :, valid]
    mu = mu[valid]
    sd = sd[valid]
    sd = np.maximum(sd, eps)

    def fill_and_z(x):
        x = x.copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            # nanmask is (N,L,K): use mu[k] for each NaN at feature index k
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization. Check data.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


# -----------------------------
# Shared maskbank metadata (Option A: maskbank defines feature set + patient split + scaling)
# -----------------------------
def load_maskbank_metadata(mask_dir: str, seq_len: int, seed: int) -> dict:
    meta_path = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Maskbank metadata not found: {meta_path}. "
            f"Make sure you generated the maskbank with make_physionet_maskbank_aligned.py using the same seq_len and seed."
        )
    z = np.load(meta_path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def standardize_with_maskbank(train: np.ndarray, val: np.ndarray, test: np.ndarray, meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)
    eps = float(meta.get("eps", 1e-6))

    if train.shape[-1] != len(valid):
        raise ValueError(
            f"[maskbank] K_raw mismatch: data has K={train.shape[-1]} but maskbank valid_feature_mask has len={len(valid)}. "
            f"This usually means your feature column selection/order differs from the maskbank's feature_cols_raw."
        )

    mu = mu_raw[valid]
    sd = sd_raw[valid]
    sd = np.maximum(sd, eps)

    def fill_and_z(x):
        x = x[:, :, valid].copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = mu[ks]
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization (maskbank scaling).")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid



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
    Expects maskbank files produced by make_physionet_maskbank.py:
      physionet_{split}_keepmask_r{r:.2f}_L{seq_len}_seed{seed}.npy

    This loader handles both:
      - if keep.shape[2] == expected_K: use as-is
      - else if valid_feature_mask is provided and keep.shape[2] == len(valid_feature_mask):
            slice keep[..., valid_feature_mask] to match expected_K
      - otherwise: raise with an actionable error telling you to regenerate maskbank.
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

    K_mask = int(keep.shape[2])
    if K_mask == int(expected_K):
        return keep

    if valid_feature_mask is not None:
        valid_feature_mask = np.asarray(valid_feature_mask).astype(bool)
        K_full = int(len(valid_feature_mask))
        K_kept = int(valid_feature_mask.sum())
        if int(expected_K) != K_kept:
            raise ValueError(
                f"Internal mismatch: expected_K={expected_K} but valid_feature_mask.sum()={K_kept}. "
                f"This suggests your preprocessing differs from how `valid` was computed."
            )
        if K_mask == K_full:
            keep2 = keep[:, :, valid_feature_mask]
            if keep2.shape[2] != expected_K:
                raise ValueError(
                    f"After slicing maskbank with valid_feature_mask, K became {keep2.shape[2]} but expected_K={expected_K}. "
                    f"Mask file: {path}"
                )
            return keep2

    msg = (
        f"Shared keepmask K mismatch for split={split}: maskbank K={K_mask} but model expects K={expected_K}."
        f"Mask file: {path}"
        f"Fix: regenerate the PhysioNet maskbank using the *same* preprocessing/feature filtering as your training script, "
        f"so the saved keepmasks have K={expected_K}. If you have a `valid_feature_mask`, you may also store masks in full "
        f"feature space and let this script slice them (requires maskbank K to equal len(valid_feature_mask))."
    )
    raise ValueError(msg)


# -----------------------------
# Model wrapper (no proto)
# -----------------------------
class CustomFGTIPlusProto(nn.Module):
    def __init__(self, fgti_model: nn.Module, proto: nn.Module, cfg):
        super().__init__()
        self.m = fgti_model
        self.proto = proto
        self.cfg = cfg
        self.device = cfg.device
        self.proto_alpha = nn.Parameter(torch.tensor(float(cfg.proto_alpha_init), dtype=torch.float32))

    def _tp(self, B: int, L: int):
        return torch.arange(L, device=self.device).float().unsqueeze(0).repeat(B, 1)

    def _build_observed_dataf(self, x_BLK: torch.Tensor, keep_mask_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        hf, dom = make_hf_and_dom_Astyle(x_BLK, keep_mask_BLK, self.cfg.flimit, self.cfg.topf)
        r_BK = self.proto(x_BLK, keep_mask_BLK)  # (B,K)
        dom = dom + self.proto_alpha * r_BK.unsqueeze(1).expand(B, L, K)
        # (B,L,2K) as expected by process_data in prior scripts
        return torch.stack([hf, dom], dim=-1).reshape(B, L, 2 * K)

    def forward(self, x_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_mask_BKL = markov_keep_mask_from_masked_ratio(
            B, K, L, r_masked=self.cfg.r_train_masked, lm=self.cfg.lm, device=self.device
        )  # (B,K,L)
        keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)  # (B,L,K)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)

        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)  # (B,K,L)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        loss = self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)
        return loss

    @torch.no_grad()
    def eval_sums(
        self,
        x_BLK: torch.Tensor,
        r_eval_masked: float,
        n_samples: int,
        keep_mask_BLK: Optional[torch.Tensor] = None,
    ):
        """Return (sum_abs, sum_sq, denom, r_obs) over masked positions for a batch."""
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        if keep_mask_BLK is None:
            keep_mask_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_eval_masked, self.cfg.lm, self.device)
            keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)
        else:
            keep_mask_BLK = keep_mask_BLK.to(self.device).float()

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)  # (B,K,L)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        imputed_samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        imputed_median = imputed_samples.median(dim=1).values

        imputed_full = cond_mask * observed_data + (1.0 - cond_mask) * imputed_median
        evalmask = 1.0 - cond_mask

        diff = (imputed_full - observed_data) * evalmask
        denom = float(evalmask.sum().clamp_min(1.0).item())

        sum_abs = float(diff.abs().sum().item())
        sum_sq  = float((diff ** 2).sum().item())
        r_obs = float(cond_mask.mean().item())
        return sum_abs, sum_sq, denom, r_obs

    @torch.no_grad()
    def eval_mae_mse_rmse(
        self,
        x_BLK: torch.Tensor,
        r_eval_masked: float,
        n_samples: int,
        keep_mask_BLK: Optional[torch.Tensor] = None,
    ):
        sum_abs, sum_sq, denom, r_obs = self.eval_sums(
            x_BLK, r_eval_masked=r_eval_masked, n_samples=n_samples, keep_mask_BLK=keep_mask_BLK
        )
        mae = sum_abs / max(denom, 1.0)
        mse = sum_sq / max(denom, 1.0)
        rmse = float(np.sqrt(mse))
        return mae, mse, rmse, r_obs

    @torch.no_grad()
    def impute_and_collect(
        self,
        x_BLK: torch.Tensor,
        r_eval_masked: float,
        n_samples: int,
        keep_mask_BLK: Optional[torch.Tensor] = None,
    ):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        if keep_mask_BLK is None:
            keep_mask_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_eval_masked, self.cfg.lm, self.device)
            keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)
        else:
            keep_mask_BLK = keep_mask_BLK.to(self.device).float()

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)  # (B,K,L)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        imputed_samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        imputed_median = imputed_samples.median(dim=1).values

        imputed_full = cond_mask * observed_data + (1.0 - cond_mask) * imputed_median
        evalmask = 1.0 - cond_mask

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_full.permute(0, 2, 1).contiguous()
        cond_BLK = cond_mask.permute(0, 2, 1).contiguous()
        eval_BLK = evalmask.permute(0, 2, 1).contiguous()

        return gt_BLK.cpu().numpy(), imp_BLK.cpu().numpy(), cond_BLK.cpu().numpy(), eval_BLK.cpu().numpy()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data + code
    ap.add_argument("--csv", type=str, default="/data/physionet2019.csv")
    ap.add_argument("--code_dir", type=str, required=True, help="Directory with Embed.py, Diff_layers.py, ts_transformer.py, diffusion.py, main_model.py")

    # windows/splits
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Use torch.cuda.amp mixed precision (saves VRAM).")
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch*grad_accum).")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    # masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--n_samples", type=int, default=20)

    # model hyperparams expected by code
    ap.add_argument("--diffusion_step_num", type=int, default=50)
    ap.add_argument("--schedule", type=str, default="quad")
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=0.2)

    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--e_layers", type=int, default=4)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--channel", type=int, default=64)
    ap.add_argument("--proj_t", type=int, default=32)
    ap.add_argument("--residual_layers", type=int, default=4)
    ap.add_argument("--timeemb", type=int, default=128)
    ap.add_argument("--featureemb", type=int, default=16)

    # freq features
    ap.add_argument("--flimit", type=float, default=0.3)
    ap.add_argument("--topf", type=int, default=10)

    # proto
    ap.add_argument("--proto_M", type=int, default=32)
    ap.add_argument("--proto_dim", type=int, default=128)
    ap.add_argument("--proto_heads", type=int, default=8)
    ap.add_argument("--proto_alpha_init", type=float, default=0.1)

    # shared masks
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_physionet_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    # saving
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_procti_physionet")
    ap.add_argument("--out_txt", type=str, default="procti_physionet_sharedmask.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = device

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # bootstrap code
    main_model = bootstrap_fgti_code(args.code_dir)

    # ---------- data: patient-wise split ----------
    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)
    feat_cols = get_feature_cols(df)
    # ---- Option A: if using shared maskbank, enforce its split + feature order + scaling ----
    mask_meta = None
    if args.use_shared_evalmask:
        mask_meta = load_maskbank_metadata(args.shared_evalmask_dir, seq_len=args.seq_len, seed=args.seed)

        # enforce exact raw feature column order used by maskbank
        feat_cols = [str(c) for c in mask_meta["feature_cols_raw"].tolist()]

        # enforce exact patient split used by maskbank (so window order + counts match masks)
        if mask_meta is None:
            # fallback: derive split from data (shouldn't happen in sharedmask runs)
            train_ids, val_ids, test_ids = split_patients(df, seed=args.seed)
        else:
            train_ids = [_to_int_id(x) for x in mask_meta["patients_train"].tolist()]
            val_ids = [_to_int_id(x) for x in mask_meta["patients_val"].tolist()]
            test_ids = [_to_int_id(x) for x in mask_meta["patients_test"].tolist()]

            df_pids = set(int(x) for x in df["Patient_ID"].dropna().unique().tolist())
            missing = (set(train_ids) | set(val_ids) | set(test_ids)) - df_pids
            if missing:
                raise ValueError(f"Maskbank metadata contains Patient_IDs not found in CSV: {sorted(list(missing))[:10]} (showing up to 10)")
        train_w = make_windows_for_patients(df, train_ids, feat_cols, time_col, args.seq_len)
        val_w   = make_windows_for_patients(df, val_ids, feat_cols, time_col, args.seq_len)
        test_w  = make_windows_for_patients(df, test_ids, feat_cols, time_col, args.seq_len)

        if mask_meta is not None:
            train_w, val_w, test_w, mu, sd, valid = standardize_with_maskbank(train_w, val_w, test_w, mask_meta)
        else:
            train_w, val_w, test_w, mu, sd, valid = standardize_by_train(train_w, val_w, test_w)
        K = train_w.shape[-1]
        args.enc_in = K
        args.c_out = K
        args.missing_rate = 0.0

        print(f"[data] #patients train/val/test={len(set(train_ids))}/{len(set(val_ids))}/{len(set(test_ids))}  "
            f"#windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={args.seq_len}")
        print(f"[prep] kept {int(valid.sum())} / {int(len(valid))} numeric features after dropping all-NaN/zero-std in TRAIN.")

        train_loader = DataLoader(WindowedDataset(train_w), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_loader   = DataLoader(WindowedDataset(val_w),   batch_size=args.batch, shuffle=False, drop_last=False)
        test_loader  = DataLoader(WindowedDataset(test_w),  batch_size=args.batch, shuffle=False, drop_last=False)

        # build model
        base = main_model.FGTI(args).to(device)
        proto = GlobalProtoRegime(args.proto_dim, args.proto_M, args.proto_heads, out_dim_K=K).to(device)
        wrapper = CustomFGTIPlusProto(base, proto, args).to(device)

        opt = torch.optim.Adam(
            list(base.parameters()) + list(proto.parameters()) + [wrapper.proto_alpha],
            lr=args.lr,
            weight_decay=1e-6,
        )

        # ---------- train ----------
        use_amp = bool(args.amp) and (device.type == "cuda")
        if use_amp:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
        else:
            scaler = None
            # autocast is still referenced below via torch.cuda.amp.autocast, but disabled.

        for ep in range(1, args.epochs + 1):
            base.train(); proto.train(); wrapper.train()
            losses = []
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            for step, xb in enumerate(train_loader):
                # Forward
                if use_amp:
                    with autocast():
                        loss = wrapper(xb) / max(1, args.grad_accum)
                else:
                    loss = wrapper(xb) / max(1, args.grad_accum)

                # Backward
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step (with gradient accumulation)
                if (step + 1) % max(1, args.grad_accum) == 0:
                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                losses.append(float(loss.item()) * max(1, args.grad_accum))

            if ep == 1 or ep % 10 == 0:
                print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.6f}  alpha={float(wrapper.proto_alpha.item()):.6f}  time={time.time()-t0:.1f}s")

        # ---------- eval ----------
        eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
        base.eval(); proto.eval(); wrapper.eval()

        def eval_split(loader, split_name: str):
            rows = []
            for r_m in eval_masked:
                robs = []
                sum_abs_tot, sum_sq_tot, denom_tot = 0.0, 0.0, 0.0
                keep_NLK = None
                if args.use_shared_evalmask:
                    keep_NLK = load_shared_keepmask(
                        args.shared_evalmask_dir, split=split_name, r_m=r_m,
                        seq_len=args.seq_len, seed=args.seed, invert=args.invert_shared_keepmask,
                        valid_feature_mask=valid, expected_K=K
                    )
                offset = 0
                for xb in loader:
                    B = xb.shape[0]
                    keep_BLK = None
                    if keep_NLK is not None:
                        keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                        offset += B

                    sum_abs, sum_sq, denom, r_obs = wrapper.eval_sums(
                        xb, r_eval_masked=r_m, n_samples=args.n_samples, keep_mask_BLK=keep_BLK
                    )
                    sum_abs_tot += sum_abs
                    sum_sq_tot  += sum_sq
                    denom_tot   += denom
                    robs.append(r_obs)

                denom_tot = max(denom_tot, 1.0)
                mae = sum_abs_tot / denom_tot
                mse = sum_sq_tot / denom_tot
                rmse = float(np.sqrt(mse))
                rows.append((split_name, r_m, float(np.mean(robs)), float(mae), float(mse), float(rmse)))
                print(f"[{split_name}] r_masked={r_m:.2f}  MAE={rows[-1][3]:.6f}  MSE={rows[-1][4]:.6f}  RMSE={rows[-1][5]:.6f}  r_obs={rows[-1][2]:.2f}")
            return rows

        val_rows = eval_split(val_loader, "val")
        test_rows = eval_split(test_loader, "test")

        # ---------- save arrays for test ----------
        if args.save_test_arrays:
            os.makedirs(args.save_dir, exist_ok=True)

            for r_m in eval_masked:
                gt_all, imp_all, cond_all, eval_all = [], [], [], []

                keep_NLK = None
                if args.use_shared_evalmask:
                    keep_NLK = load_shared_keepmask(args.shared_evalmask_dir, split="test", r_m=r_m, seq_len=args.seq_len, seed=args.seed, invert=args.invert_shared_keepmask, valid_feature_mask=valid, expected_K=K)
                offset = 0

                for xb in test_loader:
                    B = xb.shape[0]
                    keep_BLK = None
                    if keep_NLK is not None:
                        keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                        offset += B

                    gt_b, imp_b, cond_b, eval_b = wrapper.impute_and_collect(xb, r_eval_masked=r_m, n_samples=args.n_samples, keep_mask_BLK=keep_BLK)
                    gt_all.append(gt_b); imp_all.append(imp_b); cond_all.append(cond_b); eval_all.append(eval_b)

                gt_all = np.concatenate(gt_all, axis=0)
                imp_all = np.concatenate(imp_all, axis=0)
                cond_all = np.concatenate(cond_all, axis=0)
                eval_all = np.concatenate(eval_all, axis=0)

                tag = f"procti_physionet_test_r{r_m:.2f}_seed{args.seed}_ns{args.n_samples}_L{args.seq_len}"
                np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
                np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
                np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
                np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)
                print(f"[save] {tag}_*.npy  shapes gt={gt_all.shape} imputed={imp_all.shape} cond={cond_all.shape} eval={eval_all.shape}")

            np.savez(
                os.path.join(args.save_dir, f"metadata_seed{args.seed}.npz"),
                mu=mu, sd=sd, seq_len=args.seq_len,
                shared_evalmask=bool(args.use_shared_evalmask),
                invert_shared_keepmask=bool(args.invert_shared_keepmask),
                n_patients_train=len(set(train_ids)), n_patients_val=len(set(val_ids)), n_patients_test=len(set(test_ids)),
                patients_train=np.array(sorted(set(train_ids)), dtype=object),
                patients_val=np.array(sorted(set(val_ids)), dtype=object),
                patients_test=np.array(sorted(set(test_ids)), dtype=object),
            )
            print(f"[save] Test arrays saved under: {args.save_dir}")

        # ---------- write metrics ----------
        write_header = not os.path.exists(args.out_txt)
        with open(args.out_txt, "a") as f:
            if write_header:
                f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
            for split, r_m, r_obs, mae, mse, rmse in val_rows + test_rows:
                f.write(f"{split}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
