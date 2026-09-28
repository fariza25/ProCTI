#!/usr/bin/env python3
"""
Ablations follow the reduced table:
  A: no prototype branch
  B: proto bank + fixed alpha, no proto attention
  C: proto bank + proto attention + fixed alpha
  D: frozen proto bank + proto attention + fixed alpha
  E: proto bank + proto attention + learnable alpha

The script preserves the stock row-split / Option-A shared-mask alignment pipeline.
"""

import os, sys, argparse, random, importlib.util, types
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import torchcde
except Exception as e:
    raise RuntimeError("torchcde is required: pip install torchcde") from e


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


def configure_ablation(args):
    ablation = args.ablation.upper()
    if ablation == 'FULL':
        ablation = 'E'
    if ablation not in {'A', 'B', 'C', 'D', 'E'}:
        raise ValueError(f"Unsupported ablation '{args.ablation}'. Choose from A,B,C,D,E,full")

    args.use_proto_attn = True
    args.alpha_learnable = True
    args.freeze_proto_bank = False
    args.no_proto = False
    args.variant_label = ablation

    if ablation == 'A':
        args.use_proto_attn = False
        args.alpha_learnable = False
        args.freeze_proto_bank = False
        args.no_proto = True
    elif ablation == 'B':
        args.use_proto_attn = False
        args.alpha_learnable = False
        args.freeze_proto_bank = False
    elif ablation == 'C':
        args.use_proto_attn = True
        args.alpha_learnable = False
        args.freeze_proto_bank = False
    elif ablation == 'D':
        args.use_proto_attn = True
        args.alpha_learnable = False
        args.freeze_proto_bank = True
    elif ablation == 'E':
        args.use_proto_attn = True
        args.alpha_learnable = True
        args.freeze_proto_bank = False
    return args


def load_maskbank_meta(shared_dir: str, seq_len: int, seed: int):
    path = os.path.join(shared_dir, f"stock_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if os.path.exists(path):
        return path, np.load(path, allow_pickle=True)
    fallback = os.path.join(shared_dir, f"stock_maskbank_metadata_L{seq_len}_seed1.npz")
    if os.path.exists(fallback):
        print(f"[maskbank] missing {os.path.basename(path)}; falling back to {os.path.basename(fallback)}")
        return fallback, np.load(fallback, allow_pickle=True)
    raise FileNotFoundError(path)


def load_shared_keepmask(shared_dir: str, split: str, r_masked: float, seq_len: int, seed: int) -> np.ndarray:
    path = os.path.join(shared_dir, f"stock_{split}_keepmask_r{r_masked:.2f}_L{seq_len}_seed{seed}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    fallback = os.path.join(shared_dir, f"stock_{split}_keepmask_r{r_masked:.2f}_L{seq_len}_seed1.npy")
    if os.path.exists(fallback):
        print(f"[shared-mask] missing {os.path.basename(path)}; falling back to {os.path.basename(fallback)}")
        return np.load(fallback).astype(np.float32)
    raise FileNotFoundError(path)


def load_stock_csv_ordered(csv_path: str, feature_cols_raw: List[str]) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df.columns = df.columns.astype(str)
    cols = [str(c) for c in list(feature_cols_raw)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing {len(missing)} columns required by maskbank metadata. Example: {missing[:5]}")
    X = df[cols].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32)


def split_rows_then_window(X_TK: np.ndarray, seq_len: int, train_end: int, val_end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xtr = X_TK[:train_end]
    Xva = X_TK[train_end:val_end]
    Xte = X_TK[val_end:]
    def to_windows(seg: np.ndarray) -> np.ndarray:
        T, K = seg.shape
        n = T // seq_len
        if n <= 0:
            return np.zeros((0, seq_len, K), dtype=np.float32)
        return seg[:n * seq_len].reshape(n, seq_len, K).astype(np.float32)
    return to_windows(Xtr), to_windows(Xva), to_windows(Xte)


def optionA_standardize_and_keep(train_raw: np.ndarray, val_raw: np.ndarray, test_raw: np.ndarray, meta, eps: float = 1e-6):
    valid = meta['valid_feature_mask'].astype(bool)
    mu_raw = meta['train_mu_raw'].astype(np.float32)
    sd_raw = meta['train_sd_raw'].astype(np.float32)
    mu = mu_raw[valid]
    sd = sd_raw[valid]
    sd = np.where(sd > eps, sd, 1.0).astype(np.float32)

    def transform(w: np.ndarray) -> np.ndarray:
        w = w[:, :, valid].astype(np.float32)
        nan = ~np.isfinite(w)
        if nan.any():
            w[nan] = np.take(mu, np.where(nan)[2])
        w = (w - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(w).all():
            raise ValueError('Non-finite after standardization.')
        return w

    return transform(train_raw), transform(val_raw), transform(test_raw)


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


@torch.no_grad()
def observed_hf_dom(x_BLK: torch.Tensor, keep_BLK: torch.Tensor, flimit: float, topf: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, K = x_BLK.shape
    x = torch.where(keep_BLK > 0, x_BLK, torch.zeros_like(x_BLK))
    coeffs = torchcde.linear_interpolation_coeffs(x)
    maxdataf = coeffs.clone()
    freqs = torch.fft.rfftfreq(L, device=coeffs.device)

    pass_f = torch.abs(freqs) > flimit
    hf = []
    for j in range(K):
        xf = torch.fft.rfft(coeffs[:, :, j], dim=1)
        rx = torch.fft.irfft(xf * pass_f, n=L, dim=1)
        hf.append(rx)
    hf = torch.stack(hf, dim=2)

    dom = []
    for j in range(K):
        xf = torch.fft.rfft(maxdataf[:, :, j], dim=1)
        mag = torch.abs(xf)
        _, idx = torch.topk(mag, k=min(topf, mag.shape[1]), dim=1)
        keepf = torch.zeros_like(xf, dtype=torch.bool)
        keepf.scatter_(1, idx, True)
        xf2 = torch.where(keepf, xf, torch.zeros_like(xf))
        rx = torch.fft.irfft(xf2, n=L, dim=1)
        dom.append(rx)
    dom = torch.stack(dom, dim=2)
    return hf, dom


class WindowDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)
    def __len__(self):
        return int(self.x.shape[0])
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i])


class GlobalProtoRegime(nn.Module):
    def __init__(self, proto_dim: int, proto_M: int, proto_heads: int, out_K: int, use_attn: bool = True):
        super().__init__()
        D = int(proto_dim)
        self.use_attn = use_attn
        self.token_proj = nn.Linear(2, D)
        self.proto_bank = nn.Parameter(torch.randn(int(proto_M), D) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=int(proto_heads), batch_first=True)
        self.ln_q = nn.LayerNorm(D)
        self.ln_kv = nn.LayerNorm(D)
        self.to_r = nn.Linear(D, int(out_K))

    def freeze_bank_(self):
        self.proto_bank.requires_grad_(False)

    def forward(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor) -> torch.Tensor:
        B, L, K = x_BLK.shape
        tok = torch.stack([x_BLK, keep_BLK], dim=-1)
        emb = self.token_proj(tok)
        q = emb.reshape(B, L * K, -1).mean(dim=1, keepdim=True)
        q = self.ln_q(q)
        kv = self.ln_kv(self.proto_bank.unsqueeze(0).expand(B, -1, -1))
        if self.use_attn:
            ctx, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
            z = ctx.squeeze(1)
        else:
            z = kv.mean(dim=1)
        return self.to_r(z)



class BaseFGTIWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, cfg):
        super().__init__()
        self.m = base_model
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def _tp(self, B: int, L: int) -> torch.Tensor:
        return torch.arange(L, device=self.device).float().unsqueeze(0).repeat(B, 1)

    def _dataf(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor) -> torch.Tensor:
        hf, dom = observed_hf_dom(x_BLK, keep_BLK, self.cfg.flimit, self.cfg.topf)
        return torch.cat([hf, dom], dim=2)

    def forward(self, x_BLK: torch.Tensor) -> torch.Tensor:
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, self.cfg.r_train_masked, self.cfg.lm, self.device)
        keep_BLK = keep_BKL.permute(0, 2, 1)

        dataf = self._dataf(x_BLK, keep_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, dataf, observed_mask, observed_tp, observed_mask
        )
        cond_mask = keep_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        return self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)

    @torch.no_grad()
    def eval_totals(self, x_BLK: torch.Tensor, n_samples: int, keep_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        dataf = self._dataf(x_BLK, keep_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, dataf, observed_mask, observed_tp, observed_mask
        )
        cond_mask = keep_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        pred = samples.median(dim=1).values

        evalmask = 1.0 - cond_mask
        diff = (pred - observed_data) * evalmask
        denom = evalmask.sum().clamp_min(1.0)
        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()
        return float(sum_abs.item()), float(sum_sq.item()), float(denom.item())


class CustomFGTIProto(nn.Module):
    def __init__(self, base_model: nn.Module, proto: nn.Module, cfg):
        super().__init__()
        self.m = base_model
        self.proto = proto
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        alpha = torch.tensor(float(cfg.proto_alpha_init), dtype=torch.float32)
        if cfg.alpha_learnable:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer('alpha', alpha)

    def _tp(self, B: int, L: int) -> torch.Tensor:
        return torch.arange(L, device=self.device).float().unsqueeze(0).repeat(B, 1)

    def _dataf(self, x_BLK: torch.Tensor, keep_BLK: torch.Tensor) -> torch.Tensor:
        hf, dom = observed_hf_dom(x_BLK, keep_BLK, self.cfg.flimit, self.cfg.topf)
        r_BK = self.proto(x_BLK, keep_BLK)
        dom = dom + self.alpha * r_BK.unsqueeze(1)
        return torch.cat([hf, dom], dim=2)

    def forward(self, x_BLK: torch.Tensor) -> torch.Tensor:
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_BKL = markov_keep_mask_from_masked_ratio(B, K, L, self.cfg.r_train_masked, self.cfg.lm, self.device)
        keep_BLK = keep_BKL.permute(0, 2, 1)

        dataf = self._dataf(x_BLK, keep_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, dataf, observed_mask, observed_tp, observed_mask
        )
        cond_mask = keep_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        return self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)

    @torch.no_grad()
    def eval_totals(self, x_BLK: torch.Tensor, n_samples: int, keep_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        dataf = self._dataf(x_BLK, keep_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, dataf, observed_mask, observed_tp, observed_mask
        )
        cond_mask = keep_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        pred = samples.median(dim=1).values

        evalmask = 1.0 - cond_mask
        diff = (pred - observed_data) * evalmask
        denom = evalmask.sum().clamp_min(1.0)
        sum_abs = diff.abs().sum()
        sum_sq = (diff ** 2).sum()
        return float(sum_abs.item()), float(sum_sq.item()), float(denom.item())


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer):
    model.train()
    tot = 0.0
    n = 0
    for x in loader:
        x = x.to(model.device)
        loss = model(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += float(loss.item())
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def eval_global(model: CustomFGTIProto, loader: DataLoader, keep_NLK: np.ndarray, n_samples: int):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_den = 0.0
    total_keep = 0.0
    total_pts = 0.0
    bs = loader.batch_size
    for i, x in enumerate(loader):
        x = x.to(model.device)
        B, L, K = x.shape
        keep_batch = torch.from_numpy(keep_NLK[i * bs:i * bs + B]).to(model.device).float()
        sum_abs, sum_sq, denom = model.eval_totals(x, n_samples, keep_batch)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', type=str, required=True, help='One of: A, B, C, D, E, full')
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--code_dir', type=str, required=True)
    ap.add_argument('--seq_len', type=int, default=48)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)

    ap.add_argument('--r_train_masked', type=float, default=0.15)
    ap.add_argument('--lm', type=float, default=6.0)
    ap.add_argument('--eval_masked_ratios', type=str, default='0.10,0.30,0.50,0.70')
    ap.add_argument('--n_samples', type=int, default=20)

    ap.add_argument('--flimit', type=float, default=0.3)
    ap.add_argument('--topf', type=int, default=10)

    ap.add_argument('--use_shared_evalmask', action='store_true')
    ap.add_argument('--shared_evalmask_dir', type=str, required=True)

    ap.add_argument('--proto_dim', type=int, default=64)
    ap.add_argument('--proto_M', type=int, default=16)
    ap.add_argument('--proto_heads', type=int, default=4)
    ap.add_argument('--proto_alpha_init', type=float, default=0.1)

    ap.add_argument('--out_txt', type=str, default=None)

    ap.add_argument('--diffusion_step_num', type=int, default=50)
    ap.add_argument('--schedule', type=str, default='quad')
    ap.add_argument('--beta_start', type=float, default=1e-4)
    ap.add_argument('--beta_end', type=float, default=0.2)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--e_layers', type=int, default=4)
    ap.add_argument('--nheads', type=int, default=8)
    ap.add_argument('--channel', type=int, default=128)
    ap.add_argument('--proj_t', type=int, default=128)
    ap.add_argument('--residual_layers', type=int, default=4)
    ap.add_argument('--timeemb', type=int, default=128)
    ap.add_argument('--featureemb', type=int, default=16)

    args = ap.parse_args()
    if not args.use_shared_evalmask:
        raise ValueError('Pass --use_shared_evalmask.')

    configure_ablation(args)
    set_seed(args.seed)
    args.device = str(args.device)
    device = torch.device(args.device)

    meta_path, meta = load_maskbank_meta(args.shared_evalmask_dir, args.seq_len, args.seed)
    print(f"[maskbank] {meta_path}")

    X_TK = load_stock_csv_ordered(args.csv, meta['feature_cols_raw'])
    train_raw, val_raw, test_raw = split_rows_then_window(
        X_TK, args.seq_len, int(meta['split_row_train_end']), int(meta['split_row_val_end'])
    )
    train_w, val_w, test_w = optionA_standardize_and_keep(train_raw, val_raw, test_raw, meta)
    print(f"[data] windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}  K_kept={train_w.shape[-1]}  L={args.seq_len}  ablation={args.variant_label}")

    K = int(train_w.shape[-1])
    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    train_loader = DataLoader(WindowDataset(train_w), batch_size=safe_batch_size(args.batch, len(train_w)), shuffle=True)
    val_loader = DataLoader(WindowDataset(val_w), batch_size=safe_batch_size(args.batch, len(val_w)), shuffle=False)
    test_loader = DataLoader(WindowDataset(test_w), batch_size=safe_batch_size(args.batch, len(test_w)), shuffle=False)

    main_model = bootstrap_fgti_code(args.code_dir)
    base = main_model.FGTI(args).to(device)
    if args.no_proto:
        wrapper = BaseFGTIWrapper(base, args).to(device)
        proto = None
    else:
        proto = GlobalProtoRegime(args.proto_dim, args.proto_M, args.proto_heads, out_K=K, use_attn=args.use_proto_attn).to(device)
        if args.freeze_proto_bank:
            proto.freeze_bank_()
        wrapper = CustomFGTIProto(base, proto, args).to(device)

    params = [p for p in wrapper.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(wrapper, train_loader, opt)
        if ep == 1 or ep % 10 == 0:
            if hasattr(wrapper, 'alpha'):
                alpha_val = float(wrapper.alpha.detach().item())
                print(f"[epoch {ep:03d}] loss={loss:.6f}  alpha={alpha_val:.6f}")
            else:
                print(f"[epoch {ep:03d}] loss={loss:.6f}")

    ratios = parse_ratios(args.eval_masked_ratios)

    def run_split(split: str, loader: DataLoader, n_expected: int):
        rows = []
        for r in ratios:
            keep = load_shared_keepmask(args.shared_evalmask_dir, split, r, args.seq_len, args.seed)
            if keep.shape[0] != n_expected:
                raise ValueError(f"Keepmask N mismatch: split={split} r={r:.2f} keepN={keep.shape[0]} windows={n_expected}")
            mae, mse, rmse, r_actual = eval_global(wrapper, loader, keep, args.n_samples)
            print(f"[{split}] ablation={args.variant_label} r_masked={r:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}  r_actual={r_actual:.2f}")
            rows.append((args.variant_label, split, r, r_actual, mae, mse, rmse))
        return rows

    val_rows = run_split('val', val_loader, len(val_w))
    test_rows = run_split('test', test_loader, len(test_w))

    if args.out_txt:
        need_header = not os.path.exists(args.out_txt) or os.path.getsize(args.out_txt) == 0
        with open(args.out_txt, 'a') as f:
            if need_header:
                f.write('ablation\tsplit\tr_masked\tr_actual\tMAE\tMSE\tRMSE\n')
            for abl, split, r, r_actual, mae, mse, rmse in val_rows + test_rows:
                f.write(f"{abl}\t{split}\t{r:.2f}\t{r_actual:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print(f"Done. Results appended to: {args.out_txt}")
    else:
        print('Done.')


if __name__ == '__main__':
    main()


