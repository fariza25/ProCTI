#!/usr/bin/env python3

"""
ProCTI CHANNEL DROP evaluation.

Changes:
- loads precomputed weather window assets from --asset_dir
- loads precomputed channel-drop maskbanks (.npz)
- evaluates one protocol per run: drop1 or drop2
- uses exact same masked positions across models

Expected files in --asset_dir:
    weather_seq{L}_train_windows.npy
    weather_seq{L}_val_windows.npy
    weather_seq{L}_test_windows.npy

Expected maskbanks:
    weather_seq{L}_val_drop1_seed{S}.npz
    weather_seq{L}_test_drop1_seed{S}.npz
or drop2 variants
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
# Markov keep-mask for training
# -----------------------------
def markov_keep_mask_from_masked_ratio(
    B: int,
    K: int,
    L: int,
    r_masked: float,
    lm: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns (B,K,L) keepmask with 1=kept, 0=masked.
    r_masked is the missingness ratio.
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / r_keep
    p = [p_m, p_u]  # state 0=masked, 1=keep

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
# HF + Dominant frequency features
# -----------------------------
@torch.no_grad()
def make_hf_and_dom_Astyle(
    data_BLK: torch.Tensor,
    keepmask_BLK: torch.Tensor,
    flimit: float,
    topf: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data_BLK: (B,L,K)
    keepmask_BLK: (B,L,K), 1=keep
    returns:
      hf:  (B,L,K)
      dom: (B,L,K)
    """
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
# Prototype module + alpha
# -----------------------------
class GlobalProtoRegime(nn.Module):
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
        token_in = torch.stack([x_BLK, keep_mask_BLK], dim=-1)     # (B,L,K,2)
        token_emb = self.token_proj(token_in)                      # (B,L,K,D)
        q = token_emb.reshape(B, L * K, self.proto_dim).mean(dim=1, keepdim=True)
        q = self.ln_q(q)
        kv = self.ln_kv(self.proto_bank.unsqueeze(0).expand(B, -1, -1))
        ctx, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        return self.to_r(ctx.squeeze(1))                           # (B,K)


# -----------------------------
# Data helpers
# -----------------------------
class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])  # (L,K)


def load_assets(asset_dir: str, seq_len: int):
    train_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_test_windows.npy")).astype(np.float32)
    return train_w, val_w, test_w


def load_channel_drop_maskbank(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    return obj["eval_mask"].astype(np.float32)  # (N,L,K), 1=masked


# -----------------------------
# wrapper
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
        """
        Build observed_dataf with proto injection into dominant-frequency branch.
        """
        B, L, K = x_BLK.shape
        hf, dom = make_hf_and_dom_Astyle(x_BLK, keep_mask_BLK, self.cfg.flimit, self.cfg.topf)
        r_BK = self.proto(x_BLK, keep_mask_BLK)  # (B,K)
        dom = dom + self.proto_alpha * r_BK.unsqueeze(1).expand(B, L, K)
        return torch.stack([hf, dom], dim=-1).reshape(B, L, 2 * K)

    def forward(self, x_BLK: torch.Tensor):
        """
        Training step: Markov masking exactly as in the uploaded script.
        """
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

        cond_mask = keep_mask_BLK.permute(0, 2, 1)  # (B,K,L)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        loss = self.m.calc_loss(
            observed_data * observed_mask_kL,
            observed_dataf,
            cond_mask,
            observed_mask_kL,
            side_info,
        )
        return loss

    @torch.no_grad()
    def eval_mae_mse_rmse(
        self,
        x_BLK: torch.Tensor,
        n_samples: int,
        keep_mask_BLK: torch.Tensor,
        return_totals: bool = False,
    ):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_mask_BLK = keep_mask_BLK.to(self.device).float()

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)  # (B,K,L)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        imputed_samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        imputed_median = imputed_samples.median(dim=1).values  # (B,K,L)

        evalmask = 1.0 - cond_mask
        diff = (imputed_median - observed_data) * evalmask
        denom = evalmask.sum().clamp_min(1.0)

        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()

        mse = sum_sq / denom
        rmse = torch.sqrt(mse)
        mae = sum_abs / denom
        r_obs = float(cond_mask.mean().item())

        if return_totals:
            return (
                mae.item(),
                mse.item(),
                rmse.item(),
                r_obs,
                float(sum_abs.item()),
                float(sum_sq.item()),
                float(denom.item()),
            )
        return mae.item(), mse.item(), rmse.item(), r_obs

    @torch.no_grad()
    def impute_and_collect(
        self,
        x_BLK: torch.Tensor,
        n_samples: int,
        keep_mask_BLK: torch.Tensor,
    ):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_mask_BLK = keep_mask_BLK.to(self.device).float()

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        imputed_samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        imputed_median = imputed_samples.median(dim=1).values

        imputed_full = cond_mask * observed_data + (1.0 - cond_mask) * imputed_median
        evalmask = 1.0 - cond_mask

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_full.permute(0, 2, 1).contiguous()
        cond_BLK = cond_mask.permute(0, 2, 1).contiguous()
        eval_BLK = evalmask.permute(0, 2, 1).contiguous()

        return (
            gt_BLK.cpu().numpy(),
            imp_BLK.cpu().numpy(),
            cond_BLK.cpu().numpy(),
            eval_BLK.cpu().numpy(),
        )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--code_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    # training masking
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)

    # eval
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--n_samples", type=int, default=20)

    # model hyperparams expected by code
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

    # frequency features
    ap.add_argument("--flimit", type=float, default=0.3)
    ap.add_argument("--topf", type=int, default=10)

    # proto
    ap.add_argument("--proto_M", type=int, default=32)
    ap.add_argument("--proto_dim", type=int, default=128)
    ap.add_argument("--proto_heads", type=int, default=8)
    ap.add_argument("--proto_alpha_init", type=float, default=0.1)

    # saving
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_procti_channeldrop")
    ap.add_argument("--out_txt", type=str, default="procti_weather_channeldrop.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = device

    main_model = bootstrap_fgti_code(args.code_dir)

    # exact saved assets
    train_w, val_w, test_w = load_assets(args.asset_dir, args.seq_len)
    K = train_w.shape[-1]
    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    val_evalmask = load_channel_drop_maskbank(args.val_maskbank)
    test_evalmask = load_channel_drop_maskbank(args.test_maskbank)

    assert val_evalmask.shape == val_w.shape, f"val mismatch: {val_evalmask.shape} vs {val_w.shape}"
    assert test_evalmask.shape == test_w.shape, f"test mismatch: {test_evalmask.shape} vs {test_w.shape}"

    print(f"[data] train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={args.seq_len}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    train_loader = DataLoader(WindowedDataset(train_w), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(WindowedDataset(val_w), batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowedDataset(test_w), batch_size=args.batch, shuffle=False, drop_last=False)

    # build model
    base = main_model.FGTI(args).to(device)
    proto = GlobalProtoRegime(args.proto_dim, args.proto_M, args.proto_heads, out_dim_K=K).to(device)
    wrapper = CustomFGTIPlusProto(base, proto, args).to(device)

    opt = torch.optim.Adam(
        list(base.parameters()) + list(proto.parameters()) + [wrapper.proto_alpha],
        lr=args.lr,
        weight_decay=1e-6,
    )

    # train
    for ep in range(1, args.epochs + 1):
        base.train()
        proto.train()
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
            print(
                f"[epoch {ep:03d}] loss={float(np.mean(losses)):.6f} "
                f"alpha={float(wrapper.proto_alpha.item()):.6f} "
                f"time={time.time()-t0:.1f}s"
            )

    base.eval()
    proto.eval()
    wrapper.eval()

    def eval_split(loader, split_name: str, evalmask_NLK: np.ndarray):
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0.0
        total_keep = 0.0
        total_keep_count = 0.0

        offset = 0
        for xb in loader:
            B = xb.shape[0]
            eval_BLK = torch.from_numpy(evalmask_NLK[offset:offset + B]).to(device).float()  # (B,L,K), 1=masked
            keep_BLK = 1.0 - eval_BLK                                                        # (B,L,K), 1=keep
            offset += B

            mae, mse, rmse, r_obs, sum_abs, sum_sq, denom = wrapper.eval_mae_mse_rmse(
                xb,
                n_samples=args.n_samples,
                keep_mask_BLK=keep_BLK,
                return_totals=True,
            )

            total_abs += sum_abs
            total_sq += sum_sq
            total_count += denom

            total_keep += r_obs * denom
            total_keep_count += denom

        total_count = max(total_count, 1.0)
        mse_g = total_sq / total_count
        rmse_g = float(np.sqrt(mse_g))
        mae_g = total_abs / total_count
        r_obs_g = float(total_keep / max(total_keep_count, 1.0))

        print(
            f"[{split_name}] protocol={args.protocol} "
            f"MAE={mae_g:.6f} MSE={mse_g:.6f} RMSE={rmse_g:.6f} r_obs={r_obs_g:.2f}"
        )
        return split_name, args.protocol, r_obs_g, float(mae_g), float(mse_g), float(rmse_g)

    val_row = eval_split(val_loader, "val", val_evalmask)
    test_row = eval_split(test_loader, "test", test_evalmask)

    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)

        gt_all, imp_all, cond_all, eval_all = [], [], [], []
        offset = 0
        for xb in test_loader:
            B = xb.shape[0]
            eval_BLK = torch.from_numpy(test_evalmask[offset:offset + B]).to(device).float()
            keep_BLK = 1.0 - eval_BLK
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

        tag = f"procti_test_{args.protocol}_seed{args.seed}_ns{args.n_samples}_L{args.seq_len}"
        np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
        np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
        np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
        np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)
        print(f"[save] wrote test arrays to {args.save_dir} with tag {tag}")

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tr_observed\tMAE\tMSE\tRMSE\n")
        for split, protocol, r_obs, mae, mse, rmse in [val_row, test_row]:
            f.write(f"{split}\t{protocol}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
