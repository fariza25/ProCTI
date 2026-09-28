import os
import sys
import time
import json
import random
import argparse
import importlib.util
import types
from typing import Tuple

import numpy as np
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


def markov_keep_mask_from_masked_ratio(B, K, L, r_masked, lm, device):
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
def make_hf_and_dom_Astyle(data_BLK, keepmask_BLK, flimit, topf):
    B, L, K = data_BLK.shape
    x = torch.where(keepmask_BLK > 0, data_BLK, torch.zeros_like(data_BLK))
    coeffs = torchcde.linear_interpolation_coeffs(x)
    maxdataf = coeffs.clone()
    freqs = torch.fft.rfftfreq(L, device=coeffs.device)

    pass_f = torch.abs(freqs) > flimit
    hf_list = []
    dom_list = []

    for j in range(K):
        xj = coeffs[:, :, j]
        xf = torch.fft.rfft(xj, dim=1)

        rx = torch.fft.irfft(xf * pass_f, n=L, dim=1)
        hf_list.append(rx)

        mag = torch.abs(xf)
        _, idx = torch.topk(mag, k=min(topf, mag.shape[1]), dim=1)
        keep = torch.zeros_like(xf, dtype=torch.bool)
        keep.scatter_(1, idx, True)
        xf2 = torch.where(keep, xf, torch.zeros_like(xf))
        rx2 = torch.fft.irfft(xf2, n=L, dim=1)
        dom_list.append(rx2)

    hf = torch.stack(hf_list, dim=2)
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
        return torch.from_numpy(self.x[idx])  # (L,K)


def load_assets(asset_dir: str, seq_len: int):
    train_w = np.load(os.path.join(asset_dir, f"gait_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"gait_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"gait_seq{seq_len}_test_windows.npy")).astype(np.float32)
    return train_w, val_w, test_w


def load_channel_drop_maskbank(path: str):
    obj = np.load(path, allow_pickle=True)
    return obj["eval_mask"].astype(np.float32)


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
        loss = self.m.calc_loss(
            observed_data * observed_mask_kL,
            observed_dataf,
            cond_mask,
            observed_mask_kL,
            side_info,
        )
        return loss

    @torch.no_grad()
    def eval_mae_mse_rmse(self, x_BLK, n_samples, keep_mask_BLK, return_totals=False):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--asset_dir", required=True)
    parser.add_argument("--code_dir", required=True)

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--r_train_masked", type=float, default=0.15)
    parser.add_argument("--lm", type=float, default=6.0)
    parser.add_argument("--n_samples", type=int, default=20)

    parser.add_argument("--diffusion_step_num", type=int, default=50)
    parser.add_argument("--schedule", type=str, default="quad")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--e_layers", type=int, default=4)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--proj_t", type=int, default=128)
    parser.add_argument("--residual_layers", type=int, default=4)
    parser.add_argument("--timeemb", type=int, default=128)
    parser.add_argument("--featureemb", type=int, default=16)

    parser.add_argument("--flimit", type=float, default=0.3)
    parser.add_argument("--topf", type=int, default=10)

    parser.add_argument("--proto_M", type=int, default=32)
    parser.add_argument("--proto_dim", type=int, default=128)
    parser.add_argument("--proto_heads", type=int, default=8)
    parser.add_argument("--proto_alpha_init", type=float, default=0.1)

    parser.add_argument("--val_maskbank", required=True)
    parser.add_argument("--test_maskbank", required=True)
    parser.add_argument("--protocol", default="drop1")

    parser.add_argument("--out_txt", default="procti_gait_channeldrop.txt")

    args = parser.parse_args()
    args.device = torch.device(args.device)

    set_seed(args.seed)

    main_model = bootstrap_fgti_code(args.code_dir)

    train, val, test = load_assets(args.asset_dir, args.seq_len)
    K = train.shape[-1]

    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    val_mask = load_channel_drop_maskbank(args.val_maskbank)
    test_mask = load_channel_drop_maskbank(args.test_maskbank)

    train_loader = DataLoader(WindowedDataset(train), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(WindowedDataset(val), batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowedDataset(test), batch_size=args.batch, shuffle=False, drop_last=False)

    base = main_model.FGTI(args).to(args.device)
    proto = GlobalProtoRegime(args.proto_dim, args.proto_M, args.proto_heads, out_dim_K=K).to(args.device)
    wrapper = CustomFGTIPlusProto(base, proto, args).to(args.device)

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
            xb = xb.to(args.device)
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

    def eval_split(loader, split_name, evalmask_NLK):
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0.0
        total_keep = 0.0
        total_keep_count = 0.0

        offset = 0
        for xb in loader:
            B = xb.shape[0]
            eval_BLK = torch.from_numpy(evalmask_NLK[offset:offset + B]).to(args.device).float()
            keep_BLK = 1.0 - eval_BLK
            offset += B

            mae, mse, rmse, r_obs, sum_abs, sum_sq, denom = wrapper.eval_mae_mse_rmse(
                xb, n_samples=args.n_samples, keep_mask_BLK=keep_BLK, return_totals=True
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

        print(f"[{split_name}] protocol={args.protocol} MAE={mae_g:.6f} MSE={mse_g:.6f} RMSE={rmse_g:.6f} r_obs={r_obs_g:.2f}")
        return split_name, args.protocol, r_obs_g, float(mae_g), float(mse_g), float(rmse_g)

    val_row = eval_split(val_loader, "val", val_mask)
    test_row = eval_split(test_loader, "test", test_mask)

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tr_observed\tMAE\tMSE\tRMSE\n")
        for split, protocol, r_obs, mae, mse, rmse in [val_row, test_row]:
            f.write(f"{split}\t{protocol}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")


if __name__ == "__main__":
    main()

