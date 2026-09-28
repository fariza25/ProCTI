#!/usr/bin/env python3
"""
Ablations:
  A: no prototype branch
  B: proto bank + fixed alpha, no proto attention
  C: proto bank + proto attention + fixed alpha
  D: frozen proto bank + proto attention + fixed alpha
  E: proto bank + proto attention + learnable alpha

Expected shared_data_dir contents:
  train_windows.npy
  val_windows.npy
  test_windows.npy
  val_maskbank_seed{seed}.npz
  test_maskbank_seed{seed}.npz

Maskbank semantics:
  1 = masked
  0 = observed

Internal keepmask semantics:
  1 = observed
  0 = masked
"""

import os
import sys
import time
import json
import random
import argparse
import importlib.util
import types
from typing import Optional, Tuple

import numpy as np
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
# Markov keep-mask for training
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
# HF / DOM features
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
# Dataset wrapper
# -----------------------------
class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


def safe_batch_size(requested: int, n: int) -> int:
    if n <= 0:
        return 1
    return max(1, min(int(requested), int(n)))


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
# Shared maskbank loader
# -----------------------------
def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")
    if os.path.exists(path):
        obj = np.load(path)
    else:
        fallback = os.path.join(mask_dir, f"{split}_maskbank_seed1.npz")
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"Shared maskbank not found: {path}")
        print(f"[shared-mask] missing {os.path.basename(path)}; falling back to {os.path.basename(fallback)}")
        obj = np.load(fallback)

    key = f"{r_m:.2f}"
    if key not in obj:
        raise KeyError(f"Ratio key {key} not found in {path}. Available keys: {obj.files}")

    evalmask = obj[key].astype(np.float32)   # 1=masked, 0=observed
    keepmask = 1.0 - evalmask                # 1=observed, 0=masked
    return keepmask


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
        return self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)

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
# Proto wrapper (B-E)
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
        return self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)

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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", type=str, required=True)
    ap.add_argument("--code_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--n_samples", type=int, default=20)

    ap.add_argument("--diffusion_step_num", type=int, default=50)
    ap.add_argument("--schedule", type=str, default="quad")
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=0.2)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--e_layers", type=int, default=4)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--channel", type=int, default=128)
    ap.add_argument("--proj_t", type=int, default=128)
    ap.add_argument("--residual_layers", type=int, default=4)
    ap.add_argument("--timeemb", type=int, default=128)
    ap.add_argument("--featureemb", type=int, default=16)

    ap.add_argument("--flimit", type=float, default=0.3)
    ap.add_argument("--topf", type=int, default=10)

    ap.add_argument("--proto_M", type=int, default=32)
    ap.add_argument("--proto_dim", type=int, default=128)
    ap.add_argument("--proto_heads", type=int, default=8)
    ap.add_argument("--proto_alpha_init", type=float, default=0.1)

    ap.add_argument("--ablation", type=str, required=True, help="One of: A,B,C,D,E,full")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_procti_beijing_ablation")
    ap.add_argument("--out_txt", type=str, default="beijing_ablation_metrics.txt")

    args = ap.parse_args()
    args = configure_ablation(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = device

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    main_model = bootstrap_fgti_code(args.code_dir)

    train_w, val_w, test_w = load_shared_windows(args.shared_data_dir)

    L = train_w.shape[1]
    K = train_w.shape[2]
    if L != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={L}, but args.seq_len={args.seq_len}")

    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    print(f"[data] Beijing shared windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={L}")
    if args.no_proto:
        print(f"[ablation] variant={args.variant_label} (A)")
    else:
        print(
            f"[ablation] variant={args.variant_label} "
            f"use_proto_attn={args.use_proto_attn} "
            f"alpha_learnable={args.alpha_learnable} "
            f"freeze_proto_bank={args.freeze_proto_bank}"
        )

    meta_path = os.path.join(args.shared_data_dir, "split_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "train_stations" in meta:
                print(f"[split] train_stations={meta['train_stations']}")
            if "val_stations" in meta:
                print(f"[split] val_stations={meta['val_stations']}")
            if "test_stations" in meta:
                print(f"[split] test_stations={meta['test_stations']}")
        except Exception as e:
            print(f"[warn] Could not read split_meta.json: {e}")

    train_loader = DataLoader(
        WindowedDataset(train_w),
        batch_size=safe_batch_size(args.batch, len(train_w)),
        shuffle=True,
        drop_last=(len(train_w) > 1),
    )
    val_loader = DataLoader(
        WindowedDataset(val_w),
        batch_size=safe_batch_size(args.batch, len(val_w)),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        WindowedDataset(test_w),
        batch_size=safe_batch_size(args.batch, len(test_w)),
        shuffle=False,
        drop_last=False,
    )

    base = main_model.FGTI(args).to(device)

    if args.no_proto:
        proto = None
        wrapper = BaseFGTIAdapter(base, args)
        opt = torch.optim.Adam(base.parameters(), lr=args.lr, weight_decay=1e-6)
    else:
        proto = GlobalProtoRegime(
            args.proto_dim,
            args.proto_M,
            args.proto_heads,
            out_dim_K=K,
            use_attn=args.use_proto_attn,
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
        wrapper.train()
        losses = []
        t0 = time.time()
        for xb in train_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = wrapper(xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if ep == 1 or ep % 10 == 0:
            alpha_val = wrapper.current_alpha()
            if np.isnan(alpha_val):
                print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.6f}  time={time.time()-t0:.1f}s")
            else:
                print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.6f}  alpha={alpha_val:.6f}  time={time.time()-t0:.1f}s")

    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    wrapper.eval()

    def eval_split(loader, split_name: str):
        rows = []
        for r_m in eval_masked:
            total_abs = 0.0
            total_sq = 0.0
            total_count = 0.0
            total_keep = 0.0
            total_keep_count = 0.0

            keep_NLK = None
            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir,
                    split=split_name,
                    r_m=r_m,
                    seed=args.seed,
                )
                expected_n = len(loader.dataset)
                if keep_NLK.shape[0] != expected_n:
                    raise ValueError(
                        f"Mask/window count mismatch for split={split_name}, r={r_m:.2f}: "
                        f"mask has N={keep_NLK.shape[0]}, dataset has N={expected_n}"
                    )

            offset = 0
            for xb in loader:
                B = xb.shape[0]
                keep_BLK = None
                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                    offset += B

                sum_abs, sum_sq, denom = wrapper.eval_totals(xb, args.n_samples, keep_BLK)
                total_abs += sum_abs
                total_sq += sum_sq
                total_count += denom

                if keep_BLK is not None:
                    total_keep += float(keep_BLK.sum().item())
                    total_keep_count += float(np.prod(keep_BLK.shape))

            total_count = max(total_count, 1.0)
            mse_g = total_sq / total_count
            rmse_g = float(np.sqrt(mse_g))
            mae_g = total_abs / total_count

            if total_keep_count > 0:
                r_actual = 1.0 - (total_keep / total_keep_count)
            else:
                r_actual = float(r_m)

            rows.append((split_name, r_m, r_actual, float(mae_g), float(mse_g), float(rmse_g)))
            print(
                f"[{split_name}] ablation={args.variant_label} "
                f"r_masked={r_m:.2f}  MAE={rows[-1][3]:.6f}  "
                f"MSE={rows[-1][4]:.6f}  RMSE={rows[-1][5]:.6f}  r_actual={rows[-1][2]:.2f}"
            )
        return rows

    val_rows = eval_split(val_loader, "val")
    test_rows = eval_split(test_loader, "test")

    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)

        for r_m in eval_masked:
            gt_all, imp_all, cond_all, eval_all = [], [], [], []

            keep_NLK = None
            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir,
                    split="test",
                    r_m=r_m,
                    seed=args.seed,
                )

            offset = 0
            for xb in test_loader:
                B = xb.shape[0]
                keep_BLK = None
                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                    offset += B

                gt_b, imp_b, cond_b, eval_b = wrapper.impute_and_collect(
                    xb,
                    n_samples=args.n_samples,
                    keep_mask_BLK=keep_BLK,
                )
                gt_all.append(gt_b)
                imp_all.append(imp_b)
                cond_all.append(cond_b)
                eval_all.append(eval_b)

            gt_all = np.concatenate(gt_all, axis=0)
            imp_all = np.concatenate(imp_all, axis=0)
            cond_all = np.concatenate(cond_all, axis=0)
            eval_all = np.concatenate(eval_all, axis=0)

            tag = f"procti_beijing_{args.variant_label}_test_r{r_m:.2f}_seed{args.seed}_ns{args.n_samples}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)
            print(f"[save] {tag}_*.npy  shapes gt={gt_all.shape} imputed={imp_all.shape} cond={cond_all.shape} eval={eval_all.shape}")

    write_header = not os.path.exists(args.out_txt) or os.path.getsize(args.out_txt) == 0
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("ablation\tsplit\tr_masked\tr_actual\tMAE\tMSE\tRMSE\n")
        for split, r_m, r_actual, mae, mse, rmse in val_rows + test_rows:
            f.write(f"{args.variant_label}\t{split}\t{r_m:.2f}\t{r_actual:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
