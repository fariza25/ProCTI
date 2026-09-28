#!/usr/bin/env python3
"""
Ablations:
  A: no prototype branch
  B: proto bank + fixed alpha, no proto attention
  C: proto bank + proto attention + fixed alpha
  D: frozen proto bank + proto attention + fixed alpha
  E: proto bank + proto attention + learnable alpha

Notes:
- Supports Option-A shared-mask alignment using physionet maskbank metadata.
- Shared keepmask semantics: 1 = kept/observed, 0 = masked.
"""

import os
import sys
import time
import random
import argparse
import importlib.util
import types
from typing import Optional, Tuple, List

import numpy as np
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
# Utilities
# -----------------------------
def _to_int_id(x):
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


def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))


# -----------------------------
# Dynamic loader 
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
# Ablation config
# -----------------------------
def configure_ablation(args):
    ablation = args.ablation.upper()
    if ablation == "FULL":
        ablation = "E"

    if ablation not in {"A", "B", "C", "D", "E"}:
        raise ValueError("Choose ablation from A,B,C,D,E")

    args.variant_label = ablation

    args.no_proto = False
    args.use_proto_attn = True
    args.alpha_learnable = True
    args.freeze_proto_bank = False

    if ablation == "A":
        args.no_proto = True
    elif ablation == "B":
        args.use_proto_attn = False
        args.alpha_learnable = False
    elif ablation == "C":
        args.use_proto_attn = True
        args.alpha_learnable = False
    elif ablation == "D":
        args.use_proto_attn = True
        args.alpha_learnable = False
        args.freeze_proto_bank = True
    elif ablation == "E":
        args.use_proto_attn = True
        args.alpha_learnable = True

    return args


# -----------------------------
# Markov keep-mask
# -----------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device) -> torch.Tensor:
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

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
# HF + DOM features
# -----------------------------
@torch.no_grad()
def make_hf_and_dom_Astyle(data_BLK: torch.Tensor, keepmask_BLK: torch.Tensor, flimit: float, topf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, K = data_BLK.shape
    x = torch.where(keepmask_BLK > 0, data_BLK, torch.zeros_like(data_BLK))
    coeffs = torchcde.linear_interpolation_coeffs(x)
    maxdataf = coeffs.clone()

    freqs = torch.fft.rfftfreq(L, device=coeffs.device)

    pass_f = torch.abs(freqs) > flimit
    hf_list = []
    for j in range(K):
        xj = coeffs[:, :, j]
        xf = torch.fft.rfft(xj, dim=1)
        rx = torch.fft.irfft(xf * pass_f, n=L, dim=1)
        hf_list.append(rx)
    hf = torch.stack(hf_list, dim=2)

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
# Prototype module
# -----------------------------
class GlobalProtoRegime(nn.Module):
    def __init__(self, proto_dim: int, proto_M: int, proto_heads: int, out_dim_K: int, use_attn: bool = True):
        super().__init__()
        self.proto_dim = proto_dim
        self.use_attn = use_attn
        self.token_proj = nn.Linear(2, proto_dim)
        self.proto_bank = nn.Parameter(torch.randn(proto_M, proto_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=proto_dim, num_heads=proto_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(proto_dim)
        self.ln_kv = nn.LayerNorm(proto_dim)
        self.to_r = nn.Linear(proto_dim, out_dim_K)

    def freeze_bank_(self):
        self.proto_bank.requires_grad_(False)

    def forward(self, x_BLK: torch.Tensor, keep_mask_BLK: torch.Tensor) -> torch.Tensor:
        B, L, K = x_BLK.shape
        token_in = torch.stack([x_BLK, keep_mask_BLK], dim=-1)
        token_emb = self.token_proj(token_in)
        q = token_emb.reshape(B, L * K, self.proto_dim).mean(dim=1, keepdim=True)
        q = self.ln_q(q)
        kv = self.ln_kv(self.proto_bank.unsqueeze(0).expand(B, -1, -1))

        if self.use_attn:
            ctx, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
            z = ctx.squeeze(1)
        else:
            z = kv.mean(dim=1)

        return self.to_r(z)


# -----------------------------
# Dataset
# -----------------------------
class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


# -----------------------------
# PhysioNet data helpers
# -----------------------------
def load_physionet_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Patient_ID" not in df.columns:
        raise ValueError("physionet2019.csv must contain Patient_ID.")

    df["Patient_ID"] = pd.to_numeric(df["Patient_ID"], errors="coerce")
    df = df.dropna(subset=["Patient_ID"]).copy()
    df["Patient_ID"] = df["Patient_ID"].astype(int)

    for alias in ["patient_id", "PatientID", "patient", "RecordID", "id", "UID", "stay_id"]:
        if alias not in df.columns:
            df[alias] = df["Patient_ID"]

    return df


def get_time_col(df: pd.DataFrame) -> str:
    if "ICULOS" in df.columns:
        return "ICULOS"
    if "Hour" in df.columns:
        return "Hour"
    raise ValueError("Expected ICULOS or Hour in physionet CSV.")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    drop_cols = {"Patient_ID"}
    for c in ["SepsisLabel", "Hour", "ICULOS", "HospAdmTime"]:
        if c in df.columns:
            drop_cols.add(c)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in drop_cols]
    if len(feat_cols) == 0:
        raise ValueError("No numeric feature columns found.")
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
        g = g.sort_values(time_col, kind="mergesort")
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
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > eps)
    if valid.sum() == 0:
        raise ValueError("All features dropped by train standardization.")

    train = train[:, :, valid]
    val = val[:, :, valid]
    test = test[:, :, valid]
    mu = mu[valid]
    sd = np.maximum(sd[valid], eps)

    def fill_and_z(x):
        x = x.copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


# -----------------------------
# Shared maskbank metadata / keepmask
# -----------------------------
def load_maskbank_metadata(mask_dir: str, seq_len: int, seed: int):
    path = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return path, {k: z[k] for k in z.files}
    fallback = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed1.npz")
    if os.path.exists(fallback):
        print(f"[maskbank] missing {os.path.basename(path)}; falling back to {os.path.basename(fallback)}")
        z = np.load(fallback, allow_pickle=True)
        return fallback, {k: z[k] for k in z.files}
    raise FileNotFoundError(path)


def standardize_with_maskbank(train: np.ndarray, val: np.ndarray, test: np.ndarray, meta: dict):
    valid = meta["valid_feature_mask"].astype(bool)
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)
    eps = float(meta.get("eps", 1e-6))

    if train.shape[-1] != len(valid):
        raise ValueError(
            f"K_raw mismatch: data has {train.shape[-1]} but maskbank valid_feature_mask has len={len(valid)}"
        )

    mu = mu_raw[valid]
    sd = np.maximum(sd_raw[valid], eps)

    def fill_and_z(x):
        x = x[:, :, valid].copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            ks = np.where(nanmask)[2]
            x[nanmask] = mu[ks]
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization (maskbank).")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu, sd, valid


def load_shared_keepmask(
    mask_dir: str,
    split: str,
    r_m: float,
    seq_len: int,
    seed: int,
    invert: bool,
    valid_feature_mask: Optional[np.ndarray] = None,
    expected_K: Optional[int] = None,
) -> np.ndarray:
    path = os.path.join(mask_dir, f"physionet_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if os.path.exists(path):
        keep = np.load(path).astype(np.float32)
    else:
        fallback = os.path.join(mask_dir, f"physionet_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed1.npy")
        if not os.path.exists(fallback):
            raise FileNotFoundError(path)
        print(f"[shared-mask] missing {os.path.basename(path)}; falling back to {os.path.basename(fallback)}")
        keep = np.load(fallback).astype(np.float32)

    if invert or os.environ.get("PHYSIONET_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep

    if expected_K is None:
        return keep

    K_mask = keep.shape[2]
    if K_mask == expected_K:
        return keep

    if valid_feature_mask is not None:
        valid_feature_mask = np.asarray(valid_feature_mask).astype(bool)
        if K_mask == len(valid_feature_mask):
            keep = keep[:, :, valid_feature_mask]
            if keep.shape[2] == expected_K:
                return keep

    raise ValueError(
        f"Shared keepmask K mismatch: maskbank K={K_mask}, expected K={expected_K}."
    )


# -----------------------------
# Proto wrapper
# -----------------------------
class CustomFGTIPlusProto(nn.Module):
    def __init__(self, fgti_model: nn.Module, proto: nn.Module, cfg):
        super().__init__()
        self.m = fgti_model
        self.proto = proto
        self.cfg = cfg
        self.device = cfg.device

        alpha0 = torch.tensor(float(cfg.proto_alpha_init), dtype=torch.float32)
        if cfg.alpha_learnable:
            self.proto_alpha = nn.Parameter(alpha0)
        else:
            self.register_buffer("proto_alpha", alpha0)

    def current_alpha(self):
        return float(self.proto_alpha.detach().item())

    def _tp(self, B: int, L: int):
        return torch.arange(L, device=self.device).float().unsqueeze(0).repeat(B, 1)

    def _build_observed_dataf(self, x_BLK: torch.Tensor, keep_mask_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        hf, dom = make_hf_and_dom_Astyle(x_BLK, keep_mask_BLK, self.cfg.flimit, self.cfg.topf)
        r_BK = self.proto(x_BLK, keep_mask_BLK)
        dom = dom + self.proto_alpha * r_BK.unsqueeze(1).expand(B, L, K)
        return torch.stack([hf, dom], dim=-1).reshape(B, L, 2 * K)

    def forward(self, x_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_mask_BKL = markov_keep_mask_from_masked_ratio(
            B, K, L, r_masked=self.cfg.r_train_masked, lm=self.cfg.lm, device=self.device
        )
        keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        loss = self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)
        return loss

    @torch.no_grad()
    def eval_totals(self, x_BLK: torch.Tensor, n_samples: int, keep_mask_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        pred = samples.median(dim=1).values

        evalmask = 1.0 - cond_mask
        diff = (pred - observed_data) * evalmask
        denom = evalmask.sum().clamp_min(1.0)

        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()
        return float(sum_abs.item()), float(sum_sq.item()), float(denom.item())

    @torch.no_grad()
    def impute_and_collect(self, x_BLK, n_samples, keep_mask_BLK):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        pred = samples.median(dim=1).values

        imputed_full = cond_mask * observed_data + (1.0 - cond_mask) * pred
        evalmask = 1.0 - cond_mask

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_full.permute(0, 2, 1).contiguous()
        cond_BLK = cond_mask.permute(0, 2, 1).contiguous()
        eval_BLK = evalmask.permute(0, 2, 1).contiguous()

        return gt_BLK.cpu().numpy(), imp_BLK.cpu().numpy(), cond_BLK.cpu().numpy(), eval_BLK.cpu().numpy()


# -----------------------------
# A
# -----------------------------
class BaseFGTIAdapter:
    def __init__(self, base_model: nn.Module, cfg):
        self.m = base_model
        self.cfg = cfg
        self.device = cfg.device

    def train(self):
        self.m.train()

    def eval(self):
        self.m.eval()

    def current_alpha(self):
        return float("nan")

    def __call__(self, x):
        return self.forward(x)

    def _tp(self, B, L):
        return torch.arange(L, device=self.device).float().unsqueeze(0).repeat(B, 1)

    def _build_observed_dataf(self, x_BLK, keep_mask_BLK):
        B, L, K = x_BLK.shape
        hf, dom = make_hf_and_dom_Astyle(x_BLK, keep_mask_BLK, self.cfg.flimit, self.cfg.topf)
        return torch.stack([hf, dom], dim=-1).reshape(B, L, 2 * K)

    def forward(self, x_BLK):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()

        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_mask_BKL = markov_keep_mask_from_masked_ratio(
            B, K, L, r_masked=self.cfg.r_train_masked, lm=self.cfg.lm, device=self.device
        )
        keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        loss = self.m.calc_loss(
            observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info
        )
        return loss

    @torch.no_grad()
    def eval_totals(self, x_BLK: torch.Tensor, n_samples: int, keep_mask_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        pred = samples.median(dim=1).values

        evalmask = 1.0 - cond_mask
        diff = (pred - observed_data) * evalmask
        denom = evalmask.sum().clamp_min(1.0)

        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()
        return float(sum_abs.item()), float(sum_sq.item()), float(denom.item())

    @torch.no_grad()
    def impute_and_collect(self, x_BLK, n_samples, keep_mask_BLK):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        pred = samples.median(dim=1).values

        imputed_full = cond_mask * observed_data + (1.0 - cond_mask) * pred
        evalmask = 1.0 - cond_mask

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_full.permute(0, 2, 1).contiguous()
        cond_BLK = cond_mask.permute(0, 2, 1).contiguous()
        eval_BLK = evalmask.permute(0, 2, 1).contiguous()

        return gt_BLK.cpu().numpy(), imp_BLK.cpu().numpy(), cond_BLK.cpu().numpy(), eval_BLK.cpu().numpy()


# -----------------------------
# Train / eval helpers
# -----------------------------
def train_one_epoch(model, loader, opt):
    model.train()
    losses = []
    for xb in loader:
        xb = xb.to(model.device if hasattr(model, "device") else next(model.m.parameters()).device)
        opt.zero_grad(set_to_none=True)
        loss = model(xb)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_global(model, loader, keep_NLK: np.ndarray, n_samples: int):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_den = 0.0
    total_keep = 0.0
    total_pts = 0.0

    bs = loader.batch_size
    device = model.device if hasattr(model, "device") else next(model.m.parameters()).device

    for i, xb in enumerate(loader):
        xb = xb.to(device)
        B, L, K = xb.shape
        keep_batch = torch.from_numpy(keep_NLK[i * bs:i * bs + B]).to(device).float()

        sum_abs, sum_sq, denom = model.eval_totals(xb, n_samples, keep_batch)
        total_abs += sum_abs
        total_sq += sum_sq
        total_den += denom
        total_keep += float(keep_batch.sum().item())
        total_pts += float(B * L * K)

    total_den = max(total_den, 1.0)
    mae = total_abs / total_den
    mse = total_sq / total_den
    rmse = float(np.sqrt(mse))
    r_actual_masked = 1.0 - (total_keep / max(total_pts, 1.0))
    return mae, mse, rmse, r_actual_masked


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ablation", type=str, required=True, help="One of: A,B,C,D,E,full")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--code_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--n_samples", type=int, default=20)

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

    ap.add_argument("--flimit", type=float, default=0.3)
    ap.add_argument("--topf", type=int, default=10)

    ap.add_argument("--proto_M", type=int, default=32)
    ap.add_argument("--proto_dim", type=int, default=128)
    ap.add_argument("--proto_heads", type=int, default=8)
    ap.add_argument("--proto_alpha_init", type=float, default=0.1)

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, required=True)
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_procti_physionet_ablation")
    ap.add_argument("--out_txt", type=str, default="physionet_ablation_metrics.txt")

    args = ap.parse_args()
    args = configure_ablation(args)
    set_seed(args.seed)
    args.device = torch.device(args.device)
    device = args.device

    main_model = bootstrap_fgti_code(args.code_dir)

    # ---------- data ----------
    df = load_physionet_df(args.csv)
    time_col = get_time_col(df)
    feat_cols = get_feature_cols(df)

    mask_meta_path, mask_meta = load_maskbank_metadata(args.shared_evalmask_dir, seq_len=args.seq_len, seed=args.seed)

    feat_cols = [str(c) for c in mask_meta["feature_cols_raw"].tolist()]
    train_ids = [_to_int_id(x) for x in mask_meta["patients_train"].tolist()]
    val_ids = [_to_int_id(x) for x in mask_meta["patients_val"].tolist()]
    test_ids = [_to_int_id(x) for x in mask_meta["patients_test"].tolist()]

    df_pids = set(int(x) for x in df["Patient_ID"].dropna().unique().tolist())
    missing = (set(train_ids) | set(val_ids) | set(test_ids)) - df_pids
    if missing:
        raise ValueError(f"Maskbank metadata contains Patient_IDs not found in CSV: {sorted(list(missing))[:10]}")

    train_w = make_windows_for_patients(df, train_ids, feat_cols, time_col, args.seq_len)
    val_w = make_windows_for_patients(df, val_ids, feat_cols, time_col, args.seq_len)
    test_w = make_windows_for_patients(df, test_ids, feat_cols, time_col, args.seq_len)

    train_w, val_w, test_w, mu, sd, valid = standardize_with_maskbank(train_w, val_w, test_w, mask_meta)

    K = train_w.shape[-1]
    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    print(f"[maskbank] {mask_meta_path}")
    print(f"[data] #patients train/val/test={len(set(train_ids))}/{len(set(val_ids))}/{len(set(test_ids))} "
          f"#windows train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)} K={K} L={args.seq_len}")
    if args.no_proto:
        print(f"[ablation] variant={args.variant_label} (A)")
    else:
        print(f"[ablation] variant={args.variant_label} use_proto_attn={args.use_proto_attn} "
              f"alpha_learnable={args.alpha_learnable} freeze_proto_bank={args.freeze_proto_bank}")

    train_loader = DataLoader(WindowedDataset(train_w), batch_size=safe_batch_size(args.batch, len(train_w)), shuffle=True, drop_last=True)
    val_loader = DataLoader(WindowedDataset(val_w), batch_size=safe_batch_size(args.batch, len(val_w)), shuffle=False)
    test_loader = DataLoader(WindowedDataset(test_w), batch_size=safe_batch_size(args.batch, len(test_w)), shuffle=False)

    base = main_model.FGTI(args).to(device)

    if args.no_proto:
        proto = None
        wrapper = BaseFGTIAdapter(base, args)
        opt = torch.optim.Adam(base.parameters(), lr=args.lr, weight_decay=1e-6)
    else:
        proto = GlobalProtoRegime(
            args.proto_dim, args.proto_M, args.proto_heads, out_dim_K=K, use_attn=args.use_proto_attn
        ).to(device)
        if args.freeze_proto_bank:
            proto.freeze_bank_()
        wrapper = CustomFGTIPlusProto(base, proto, args).to(device)

        params = list(base.parameters())
        params += [p for p in proto.parameters() if p.requires_grad]
        if isinstance(getattr(wrapper, "proto_alpha", None), nn.Parameter):
            params += [wrapper.proto_alpha]
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-6)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(wrapper, train_loader, opt)
        if ep == 1 or ep % 10 == 0:
            alpha_val = wrapper.current_alpha()
            if np.isnan(alpha_val):
                print(f"[epoch {ep:03d}] loss={loss:.6f}  time={time.time()-t0:.1f}s")
            else:
                print(f"[epoch {ep:03d}] loss={loss:.6f}  alpha={alpha_val:.6f}  time={time.time()-t0:.1f}s")

    ratios = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]

    def run_split(split: str, loader: DataLoader, n_expected: int):
        rows = []
        for r in ratios:
            keep = load_shared_keepmask(
                args.shared_evalmask_dir,
                split,
                r,
                args.seq_len,
                args.seed,
                args.invert_shared_keepmask,
                valid_feature_mask=valid,
                expected_K=K,
            )
            if keep.shape[0] != n_expected:
                raise ValueError(f"Keepmask N mismatch: split={split} r={r:.2f} keepN={keep.shape[0]} windows={n_expected}")

            mae, mse, rmse, r_actual = eval_global(wrapper, loader, keep, args.n_samples)
            print(f"[{split}] ablation={args.variant_label} r_masked={r:.2f} "
                  f"MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_actual={r_actual:.2f}")
            rows.append((args.variant_label, split, r, r_actual, mae, mse, rmse))
        return rows

    val_rows = run_split("val", val_loader, len(val_w))
    test_rows = run_split("test", test_loader, len(test_w))

    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)
        for r in ratios:
            keep = load_shared_keepmask(
                args.shared_evalmask_dir,
                "test",
                r,
                args.seq_len,
                args.seed,
                args.invert_shared_keepmask,
                valid_feature_mask=valid,
                expected_K=K,
            )
            gt_all, imp_all, cond_all, eval_all = [], [], [], []
            offset = 0
            bs = test_loader.batch_size
            for xb in test_loader:
                B = xb.shape[0]
                keep_batch = torch.from_numpy(keep[offset:offset + B]).to(device).float()
                offset += B
                gt_b, imp_b, cond_b, eval_b = wrapper.impute_and_collect(xb, args.n_samples, keep_batch)
                gt_all.append(gt_b)
                imp_all.append(imp_b)
                cond_all.append(cond_b)
                eval_all.append(eval_b)

            gt_all = np.concatenate(gt_all, axis=0)
            imp_all = np.concatenate(imp_all, axis=0)
            cond_all = np.concatenate(cond_all, axis=0)
            eval_all = np.concatenate(eval_all, axis=0)

            tag = f"procti_physionet_{args.variant_label}_test_r{r:.2f}_seed{args.seed}_ns{args.n_samples}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)

        print(f"[save] Test arrays saved under: {args.save_dir}")

    need_header = not os.path.exists(args.out_txt) or os.path.getsize(args.out_txt) == 0
    with open(args.out_txt, "a") as f:
        if need_header:
            f.write("ablation\tsplit\tr_masked\tr_actual\tMAE\tMSE\tRMSE\n")
        for abl, split, r, r_actual, mae, mse, rmse in val_rows + test_rows:
            f.write(f"{abl}\t{split}\t{r:.2f}\t{r_actual:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
