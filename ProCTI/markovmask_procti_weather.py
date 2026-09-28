
#!/usr/bin/env python3


import os, sys, time, random, argparse, importlib.util, types
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device) -> torch.Tensor:
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / r_keep
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
        token_in = torch.stack([x_BLK, keep_mask_BLK], dim=-1)
        token_emb = self.token_proj(token_in)
        q = token_emb.reshape(B, L * K, self.proto_dim).mean(dim=1, keepdim=True)
        q = self.ln_q(q)
        kv = self.ln_kv(self.proto_bank.unsqueeze(0).expand(B, -1, -1))
        ctx, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        return self.to_r(ctx.squeeze(1))


class WindowedDataset(Dataset):
    def __init__(self, windows_NLK: np.ndarray):
        self.x = windows_NLK.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


def load_weather_windows(csv_path: str, seq_len: int, time_col: Optional[str]):
    df = pd.read_csv(csv_path)
    if time_col is None:
        if "date" in df.columns:
            time_col = "date"
        elif "time" in df.columns:
            time_col = "time"
        else:
            time_col = df.columns[0]

    t = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[~t.isna()].copy()
    df["_t"] = pd.to_datetime(df[time_col])
    df = df.sort_values("_t", kind="mergesort").drop(columns=["_t"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found in weather CSV.")
    X = df[num_cols].astype(np.float32).to_numpy()

    T, K = X.shape
    n_win = T // seq_len
    if n_win <= 0:
        raise ValueError(f"Not enough rows ({T}) for seq_len={seq_len}")
    X = X[: n_win * seq_len].reshape(n_win, seq_len, K)
    return X, num_cols, time_col


def chrono_split_windows(windows_NLK: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(windows_NLK)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = windows_NLK[:n_train]
    val = windows_NLK[n_train:n_train + n_val]
    test = windows_NLK[n_train + n_val:]
    return train, val, test


def standardize_by_train(train: np.ndarray, val: np.ndarray, test: np.ndarray, eps=1e-6):
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd), sd, 1.0)
    sd = np.maximum(sd, eps)

    def fill_and_z(x):
        x = x.copy()
        nanmask = np.isnan(x)
        if nanmask.any():
            x[nanmask] = np.take(mu, np.where(nanmask)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite after standardization.")
        return x.astype(np.float32)

    return fill_and_z(train), fill_and_z(val), fill_and_z(test), mu.astype(np.float32), sd.astype(np.float32)


def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seq_len: int, seed: int, invert: bool) -> np.ndarray:
    path = os.path.join(mask_dir, f"weather_{split}_keepmask_r{r_m:.2f}_L{seq_len}_seed{seed}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared keepmask not found: {path}")
    keep = np.load(path).astype(np.float32)
    if invert or os.environ.get("WEATHER_MASKBANK_INVERT", "0") == "1":
        keep = 1.0 - keep
    return keep


def save_npz_bundle(out_path: str, **arrays):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"[saved] {out_path}")


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

    def _build_observed_dataf(self, x_BLK: torch.Tensor, keep_mask_BLK: torch.Tensor, return_parts: bool = False):
        B, L, K = x_BLK.shape
        hf, dom = make_hf_and_dom_Astyle(x_BLK, keep_mask_BLK, self.cfg.flimit, self.cfg.topf)
        r_BK = self.proto(x_BLK, keep_mask_BLK)
        dom_adj = dom + self.proto_alpha * r_BK.unsqueeze(1).expand(B, L, K)
        observed_dataf = torch.stack([hf, dom_adj], dim=-1).reshape(B, L, 2 * K)
        if return_parts:
            return observed_dataf, dom, dom_adj
        return observed_dataf

    def forward(self, x_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        keep_mask_BKL = markov_keep_mask_from_masked_ratio(B, K, L, self.cfg.r_train_masked, self.cfg.lm, self.device)
        keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)

        observed_dataf_BL2K = self._build_observed_dataf(x_BLK, keep_mask_BLK)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        return self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)

    @torch.no_grad()
    def eval_mae_mse_rmse(self, x_BLK: torch.Tensor, r_eval_masked: float, n_samples: int,
                          keep_mask_BLK: Optional[torch.Tensor] = None, return_totals: bool = False):
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

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        imputed_samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        imputed_median = imputed_samples.median(dim=1).values

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
            return mae.item(), mse.item(), rmse.item(), r_obs, float(sum_abs.item()), float(sum_sq.item()), float(denom.item())
        return mae.item(), mse.item(), rmse.item(), r_obs

    @torch.no_grad()
    def impute_and_collect(self, x_BLK: torch.Tensor, r_eval_masked: float, n_samples: int,
                           keep_mask_BLK: Optional[torch.Tensor] = None):
        B, L, K = x_BLK.shape
        x_BLK = x_BLK.to(self.device).float()
        observed_tp = self._tp(B, L)
        observed_mask = torch.ones((B, L, K), device=self.device)

        if keep_mask_BLK is None:
            keep_mask_BKL = markov_keep_mask_from_masked_ratio(B, K, L, r_eval_masked, self.cfg.lm, self.device)
            keep_mask_BLK = keep_mask_BKL.permute(0, 2, 1)
        else:
            keep_mask_BLK = keep_mask_BLK.to(self.device).float()

        observed_dataf_BL2K, dom_BLK, dom_adj_BLK = self._build_observed_dataf(x_BLK, keep_mask_BLK, return_parts=True)
        observed_data, observed_dataf, observed_mask_kL, observed_tp2, _ = self.m.process_data(
            x_BLK, observed_dataf_BL2K, observed_mask, observed_tp, observed_mask
        )

        cond_mask = keep_mask_BLK.permute(0, 2, 1)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)

        imputed_samples = self.m.impute(observed_data, observed_dataf, cond_mask, side_info, n_samples=n_samples)
        imputed_median = imputed_samples.median(dim=1).values
        evalmask = 1.0 - cond_mask

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_median.permute(0, 2, 1).contiguous()
        cond_BLK = cond_mask.permute(0, 2, 1).contiguous()
        eval_BLK = evalmask.permute(0, 2, 1).contiguous()

        return (
            gt_BLK.cpu().numpy(),
            imp_BLK.cpu().numpy(),
            cond_BLK.cpu().numpy(),
            eval_BLK.cpu().numpy(),
            dom_BLK.cpu().numpy(),
            dom_adj_BLK.cpu().numpy(),
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--code_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
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

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_weather_maskbank")
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_procti_weather")
    ap.add_argument("--out_txt", type=str, default="procti_weather_sharedmask.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = device

    main_model = bootstrap_fgti_code(args.code_dir)

    windows, feat_cols, tcol = load_weather_windows(args.csv, args.seq_len, args.time_col)
    train_w, val_w, test_w = chrono_split_windows(windows)
    train_w, val_w, test_w, mu, sd = standardize_by_train(train_w, val_w, test_w)

    K = train_w.shape[-1]
    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    print(f"[data] windows={len(windows)} train/val/test={len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={args.seq_len}  time_col={tcol}")

    train_loader = DataLoader(WindowedDataset(train_w), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(WindowedDataset(val_w), batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowedDataset(test_w), batch_size=args.batch, shuffle=False, drop_last=False)

    base = main_model.FGTI(args).to(device)
    proto = GlobalProtoRegime(args.proto_dim, args.proto_M, args.proto_heads, out_dim_K=K).to(device)
    wrapper = CustomFGTIPlusProto(base, proto, args).to(device)

    opt = torch.optim.Adam(
        list(base.parameters()) + list(proto.parameters()) + [wrapper.proto_alpha],
        lr=args.lr,
        weight_decay=1e-6,
    )

    for ep in range(1, args.epochs + 1):
        base.train()
        proto.train()
        wrapper.train()
        losses = []
        t0 = time.time()
        for xb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = wrapper(xb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep == 1 or ep % 10 == 0:
            print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.6f}  alpha={float(wrapper.proto_alpha.item()):.6f}  time={time.time()-t0:.1f}s")

    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    base.eval()
    proto.eval()
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
                    args.shared_evalmask_dir, split=split_name, r_m=r_m,
                    seq_len=args.seq_len, seed=args.seed, invert=args.invert_shared_keepmask
                )

            offset = 0
            for xb in loader:
                B = xb.shape[0]
                keep_BLK = None
                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                    offset += B

                mae, mse, rmse, r_obs, sum_abs, sum_sq, denom = wrapper.eval_mae_mse_rmse(
                    xb, r_eval_masked=r_m, n_samples=args.n_samples, keep_mask_BLK=keep_BLK, return_totals=True
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

            rows.append((split_name, r_m, r_obs_g, float(mae_g), float(mse_g), float(rmse_g)))
            print(f"[{split_name}] r_masked={r_m:.2f}  MAE={rows[-1][3]:.6f}  MSE={rows[-1][4]:.6f}  RMSE={rows[-1][5]:.6f}  r_obs={rows[-1][2]:.2f}")
        return rows

    val_rows = eval_split(val_loader, "val")
    test_rows = eval_split(test_loader, "test")

    if args.save_test_arrays:
        os.makedirs(args.save_dir, exist_ok=True)

        for r_m in eval_masked:
            gt_all, imp_all, cond_all, eval_all, dom_all, dom_adj_all = [], [], [], [], [], []

            keep_NLK = None
            if args.use_shared_evalmask:
                keep_NLK = load_shared_keepmask(
                    args.shared_evalmask_dir, split="test", r_m=r_m,
                    seq_len=args.seq_len, seed=args.seed, invert=args.invert_shared_keepmask
                )

            offset = 0
            for xb in test_loader:
                B = xb.shape[0]
                keep_BLK = None
                if keep_NLK is not None:
                    keep_BLK = torch.from_numpy(keep_NLK[offset:offset+B]).to(device)
                    offset += B

                gt_b, imp_b, cond_b, eval_b, dom_b, dom_adj_b = wrapper.impute_and_collect(
                    xb, r_eval_masked=r_m, n_samples=args.n_samples, keep_mask_BLK=keep_BLK
                )
                gt_all.append(gt_b)
                imp_all.append(imp_b)
                cond_all.append(cond_b)
                eval_all.append(eval_b)
                dom_all.append(dom_b)
                dom_adj_all.append(dom_adj_b)

            gt_all = np.concatenate(gt_all, axis=0).astype(np.float32)
            imp_all = np.concatenate(imp_all, axis=0).astype(np.float32)
            cond_all = np.concatenate(cond_all, axis=0).astype(np.float32)
            eval_all = np.concatenate(eval_all, axis=0).astype(np.float32)
            dom_all = np.concatenate(dom_all, axis=0).astype(np.float32)
            dom_adj_all = np.concatenate(dom_adj_all, axis=0).astype(np.float32)

            tag = f"procti_weather_test_r{r_m:.2f}_seed{args.seed}_ns{args.n_samples}_L{args.seq_len}"
            save_npz_bundle(
                os.path.join(args.save_dir, f"{tag}.npz"),
                gt=gt_all,
                imputed=imp_all,
                cond_mask=cond_all,
                eval_mask=eval_all,
                dom=dom_all,
                dom_adj=dom_adj_all,
            )

        save_npz_bundle(
            os.path.join(args.save_dir, f"weather_metadata_seed{args.seed}_L{args.seq_len}.npz"),
            mu=mu,
            sd=sd,
            feature_cols=np.array(feat_cols, dtype=object),
            seq_len=np.array(args.seq_len),
            time_col=np.array(tcol, dtype=object),
            shared_evalmask=np.array(bool(args.use_shared_evalmask)),
            invert_shared_keepmask=np.array(bool(args.invert_shared_keepmask)),
        )
        print(f"[save] Test arrays saved under: {args.save_dir}")

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split, r_m, r_obs, mae, mse, rmse in val_rows + test_rows:
            f.write(f"{split}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
