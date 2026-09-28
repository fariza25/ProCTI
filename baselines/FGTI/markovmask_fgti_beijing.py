#!/usr/bin/env python3

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
    """
    Loads FGTI codebase from a directory containing:
      Embed.py, Diff_layers.py, ts_transformer.py, diffusion.py, main_model.py
    and returns the loaded main_model module.
    """
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
# Markov keep-mask (training-time random masking only)
# -----------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device) -> torch.Tensor:
    """
    Returns (B,K,L) keepmask with 1=kept, 0=masked, with segmenty Markov behavior.
    r_masked is fraction masked (missingness).
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    # state 0 = masked, state 1 = keep
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


# -----------------------------
# HF / DOM construction
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
    keepmask_BLK: (B,L,K) 1=kept
    Returns:
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
# Shared Beijing maskbank loader
# -----------------------------
def load_shared_keepmask(mask_dir: str, split: str, r_m: float, seed: int) -> np.ndarray:
    """
    Loads:
      {split}_maskbank_seed{seed}.npz
    where each key is "0.10", "0.30", etc and stores evalmask:
      1 = masked
      0 = observed

    Converts to keepmask expected by FGTI:
      keepmask = 1 - evalmask
    """
    path = os.path.join(mask_dir, f"{split}_maskbank_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared maskbank not found: {path}")

    obj = np.load(path)
    key = f"{r_m:.2f}"
    if key not in obj:
        raise KeyError(f"Ratio key {key} not found in {path}. Available keys: {obj.files}")

    evalmask = obj[key].astype(np.float32)   # 1=masked, 0=observed
    keepmask = 1.0 - evalmask                # 1=observed, 0=masked
    return keepmask


# -----------------------------
# Model wrapper (no proto)
# -----------------------------
class CustomFGTI(nn.Module):
    def __init__(self, fgti_model: nn.Module, cfg):
        super().__init__()
        self.m = fgti_model
        self.cfg = cfg
        self.device = cfg.device

    def _tp(self, B: int, L: int):
        return torch.arange(L, device=self.device).float().unsqueeze(0).repeat(B, 1)

    def _build_observed_dataf(self, x_BLK: torch.Tensor, keep_mask_BLK: torch.Tensor):
        B, L, K = x_BLK.shape
        hf, dom = make_hf_and_dom_Astyle(x_BLK, keep_mask_BLK, self.cfg.flimit, self.cfg.topf)
        return torch.stack([hf, dom], dim=-1).reshape(B, L, 2 * K)

    def forward(self, x_BLK: torch.Tensor):
        """
        Train step: sample Markov keepmask with r_train_masked, compute loss.
        """
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

        cond_mask = keep_mask_BLK.permute(0, 2, 1)  # (B,K,L)
        side_info = self.m.get_side_info(observed_tp2, cond_mask)
        loss = self.m.calc_loss(observed_data * observed_mask_kL, observed_dataf, cond_mask, observed_mask_kL, side_info)
        return loss

    @torch.no_grad()
    def eval_mae_mse_rmse(
        self,
        x_BLK: torch.Tensor,
        r_eval_masked: float,
        n_samples: int,
        keep_mask_BLK: Optional[torch.Tensor] = None,
        return_totals: bool = False,
    ):
        """
        If return_totals=True, returns
        (mae,mse,rmse,r_obs,sum_abs,sum_sq,denom)
        """
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
                mae.item(), mse.item(), rmse.item(), r_obs,
                float(sum_abs.item()), float(sum_sq.item()), float(denom.item())
            )
        return mae.item(), mse.item(), rmse.item(), r_obs

    @torch.no_grad()
    def impute_and_collect(
        self,
        x_BLK: torch.Tensor,
        r_eval_masked: float,
        n_samples: int,
        keep_mask_BLK: Optional[torch.Tensor] = None
    ):
        """
        Returns numpy arrays in (B,L,K) for:
          gt, imputed_full, condmask(keep), evalmask
        """
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shared_data_dir", type=str, required=True,
                    help="Directory containing train_windows.npy, val_windows.npy, test_windows.npy and maskbanks")
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

    # diffusion/model params
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

    # shared eval masks
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="")

    # save test arrays
    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_dir", type=str, default="saved_test_arrays_fgti_beijing_sharedmask")
    ap.add_argument("--out_txt", type=str, default="fgti_beijing_sharedmask.txt")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.device = device

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        args.shared_evalmask_dir = args.shared_data_dir

    main_model = bootstrap_fgti_code(args.code_dir)

    # ---- shared Beijing windows
    train_w, val_w, test_w = load_shared_windows(args.shared_data_dir)

    L = train_w.shape[1]
    K = train_w.shape[2]

    if L != args.seq_len:
        raise ValueError(f"seq_len mismatch: shared windows have L={L}, but args.seq_len={args.seq_len}")

    args.enc_in = K
    args.c_out = K
    args.missing_rate = 0.0

    print(
        f"[data] Beijing shared windows train/val/test = "
        f"{len(train_w)}/{len(val_w)}/{len(test_w)}  K={K} L={L}"
    )

    meta_path = os.path.join(args.shared_data_dir, "split_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "train_stations" in meta and "val_stations" in meta and "test_stations" in meta:
                print(f"[split] train_stations={meta['train_stations']}")
                print(f"[split] val_stations={meta['val_stations']}")
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
    wrapper = CustomFGTI(base, args).to(device)
    opt = torch.optim.Adam(list(base.parameters()), lr=args.lr, weight_decay=1e-6)

    # ---- training
    for ep in range(1, args.epochs + 1):
        base.train()
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
            print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.6f}  time={time.time()-t0:.1f}s")

    # ---- evaluation
    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    base.eval()
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

                mae, mse, rmse, r_obs, sum_abs, sum_sq, denom = wrapper.eval_mae_mse_rmse(
                    xb, r_eval_masked=r_m, n_samples=args.n_samples,
                    keep_mask_BLK=keep_BLK, return_totals=True
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
            print(
                f"[{split_name}] r_masked={r_m:.2f}  "
                f"MAE={rows[-1][3]:.6f}  MSE={rows[-1][4]:.6f}  "
                f"RMSE={rows[-1][5]:.6f}  r_obs={rows[-1][2]:.2f}"
            )
        return rows

    val_rows = eval_split(val_loader, "val")
    test_rows = eval_split(test_loader, "test")

    # ---- save test arrays
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
                    xb, r_eval_masked=r_m, n_samples=args.n_samples, keep_mask_BLK=keep_BLK
                )
                gt_all.append(gt_b)
                imp_all.append(imp_b)
                cond_all.append(cond_b)
                eval_all.append(eval_b)

            gt_all = np.concatenate(gt_all, axis=0)
            imp_all = np.concatenate(imp_all, axis=0)
            cond_all = np.concatenate(cond_all, axis=0)
            eval_all = np.concatenate(eval_all, axis=0)

            tag = f"fgti_beijing_test_r{r_m:.2f}_seed{args.seed}_ns{args.n_samples}_L{args.seq_len}"
            np.save(os.path.join(args.save_dir, f"{tag}_gt.npy"), gt_all)
            np.save(os.path.join(args.save_dir, f"{tag}_imputed.npy"), imp_all)
            np.save(os.path.join(args.save_dir, f"{tag}_condmask.npy"), cond_all)
            np.save(os.path.join(args.save_dir, f"{tag}_evalmask.npy"), eval_all)
            print(
                f"[save] {tag}_*.npy  shapes "
                f"gt={gt_all.shape} imputed={imp_all.shape} cond={cond_all.shape} eval={eval_all.shape}"
            )

        meta_out = {
            "seq_len": int(args.seq_len),
            "shared_evalmask": bool(args.use_shared_evalmask),
            "shared_data_dir": args.shared_data_dir,
        }
        with open(os.path.join(args.save_dir, f"metadata_seed{args.seed}.json"), "w") as f:
            json.dump(meta_out, f, indent=2)

        print(f"[save] Test arrays saved under: {args.save_dir}")

    # ---- write results
    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")
        for split, r_m, r_obs, mae, mse, rmse in val_rows + test_rows:
            f.write(f"{split}\t{r_m:.2f}\t{r_obs:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")

    print(f"Done. Results appended to: {args.out_txt}")


if __name__ == "__main__":
    main()
