import os, argparse, random
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from csdi_main_model import CSDI_base
from csdi_utils import train as csdi_train


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Markov keep-mask from MASKED ratio (segment-based)
# (Only used if you don't pass --use_shared_evalmask)
# -----------------------------
def markov_keep_mask_from_masked_ratio(B: int, K: int, L: int, r_masked: float, lm: float, device: torch.device):
    """
    Returns keepmask (B,K,L) with 1=kept/observed, 0=masked.
    r_masked is missingness ratio (fraction masked).
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0.0
    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm                           # P(0->1)
    p_u = p_m * (1.0 - r_keep) / r_keep      # P(1->0) so stationary keep ratio = r_keep
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
# Maskbank metadata + keepmask loader
# -----------------------------
def load_maskbank_metadata(mask_dir: str, seq_len: int, seed: int) -> Dict[str, Any]:
    meta_path = os.path.join(mask_dir, f"physionet_maskbank_metadata_L{seq_len}_seed{seed}.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Maskbank metadata not found: {meta_path}")
    z = np.load(meta_path, allow_pickle=True)

    meta = {k: z[k] for k in z.files}
    meta["valid_feature_mask"] = meta["valid_feature_mask"].astype(bool)
    meta["feature_cols_raw"] = [str(x) for x in meta["feature_cols_raw"].tolist()]
    meta["n_features_raw"] = int(meta.get("n_features_raw", len(meta["feature_cols_raw"])))
    meta["n_features_kept"] = int(meta.get("n_features_kept", int(meta["valid_feature_mask"].sum())))
    return meta


def load_shared_keepmask(
    mask_dir: str,
    split: str,
    r_m: float,
    seq_len: int,
    seed: int,
    invert: bool = False,
    *,
    valid_feature_mask=None,
    expected_K=None,
) -> np.ndarray:
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
                f"Internal mismatch: expected_K={expected_K} but valid_feature_mask.sum()={K_kept}."
            )

        if K_mask == K_full:
            keep2 = keep[:, :, valid_feature_mask]
            if keep2.shape[2] != expected_K:
                raise ValueError(
                    f"After slicing maskbank with valid_feature_mask, K became {keep2.shape[2]} but expected_K={expected_K}."
                )
            return keep2

    raise ValueError(
        f"Shared keepmask K mismatch for split={split}: maskbank K={K_mask} but model expects K={expected_K}. "
        f"Regenerate the PhysioNet maskbank with the same preprocessing, or slice using valid_feature_mask."
    )


# -----------------------------
# PhysioNet preprocessing
# -----------------------------
def normalize_patient_id(df: pd.DataFrame, pid_col: str = "Patient_ID") -> pd.DataFrame:
    if pid_col not in df.columns:
        raise ValueError(f"{pid_col} column not found in CSV.")
    df = df.copy()
    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce")
    df = df.dropna(subset=[pid_col])
    df[pid_col] = df[pid_col].astype(int)
    return df


def pick_time_col(df: pd.DataFrame) -> str:
    for c in ("Hour", "ICULOS"):
        if c in df.columns:
            return c
    return ""


def fill_forward_then_zero_per_patient(g: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    gg = g[cols].copy()
    gg = gg.ffill().bfill()
    gg = gg.fillna(0.0)
    return gg


def make_windows_for_patients(
    df: pd.DataFrame,
    patient_ids: List[int],
    feat_cols_kept: List[str],
    time_col: str,
    seq_len: int,
) -> np.ndarray:
    pid_col = "Patient_ID"
    want = set(int(x) for x in patient_ids)

    windows = []
    for pid, g in df.groupby(pid_col, sort=False):
        if int(pid) not in want:
            continue
        if time_col:
            g = g.sort_values(time_col, kind="mergesort")
        gg = fill_forward_then_zero_per_patient(g, feat_cols_kept)
        arr = gg.to_numpy(dtype=np.float32)  # (T,K)
        T = arr.shape[0]
        n_win = T // seq_len
        if n_win <= 0:
            continue
        arr = arr[: n_win * seq_len].reshape(n_win, seq_len, arr.shape[1])
        windows.append(arr)

    if not windows:
        raise ValueError("No windows formed for this split (seq_len too large or split empty/mismatched).")
    return np.concatenate(windows, axis=0).astype(np.float32)


def standardize_with_maskbank_stats(windows: np.ndarray, mu_raw: np.ndarray, sd_raw: np.ndarray, valid_mask: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = mu_raw[valid_mask].astype(np.float32)
    sd = sd_raw[valid_mask].astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0)
    sd = np.maximum(sd, eps)

    # Fill any remaining NaNs (after ffill/bfill) with train mean so they become ~0 after z-score.

    win = windows.astype(np.float32, copy=False)

    win = np.where(np.isfinite(win), win, mu[None, None, :])

    x = (win - mu[None, None, :]) / sd[None, None, :]
    if not np.isfinite(x).all():
        raise ValueError("Non-finite after standardization (check fill/columns).")
    return x.astype(np.float32)


# -----------------------------
# Dataset + collate (same pattern as csdi_gait_sharedmask.py)
# -----------------------------
class PhysioCSDIDataset(Dataset):
    def __init__(
        self,
        windows_NLK: np.ndarray,
        mode: str,
        r_eval_masked: float,
        lm: float,
        precomputed_keepmask_NLK: Optional[np.ndarray] = None,
    ):
        assert mode in ("train", "eval")
        self.x = windows_NLK.astype(np.float32)
        self.mode = mode
        self.r_eval_masked = float(r_eval_masked)
        self.lm = float(lm)
        if precomputed_keepmask_NLK is not None:
            assert precomputed_keepmask_NLK.shape == self.x.shape, (precomputed_keepmask_NLK.shape, self.x.shape)
            self.keep = precomputed_keepmask_NLK.astype(np.float32)
        else:
            self.keep = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_LK = torch.from_numpy(self.x[idx])  # (L,K)
        L, K = x_LK.shape

        observed_data = x_LK
        observed_mask = torch.ones((L, K), dtype=torch.float32)
        timepoints = torch.arange(L, dtype=torch.float32)

        if self.mode == "eval":
            if self.keep is not None:
                gt_mask = torch.from_numpy(self.keep[idx]).float()  # (L,K) keep-mask
            else:
                keep_KL = markov_keep_mask_from_masked_ratio(
                    B=1, K=K, L=L, r_masked=self.r_eval_masked, lm=self.lm, device=torch.device("cpu")
                )[0]  # (K,L)
                gt_mask = keep_KL.transpose(0, 1).contiguous().float()  # (L,K)
        else:
            gt_mask = observed_mask.clone()

        return {
            "observed_data": observed_data,     # (L,K)
            "observed_mask": observed_mask,     # (L,K)
            "timepoints": timepoints,           # (L,)
            "gt_mask": gt_mask,                 # (L,K) keep-mask
            "hist_mask": observed_mask,         # (L,K)
            "cut_length": torch.tensor(0, dtype=torch.long),
        }


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    observed_data = torch.stack([s["observed_data"] for s in samples], dim=0)  # (B,L,K)
    observed_mask = torch.stack([s["observed_mask"] for s in samples], dim=0)
    timepoints = torch.stack([s["timepoints"] for s in samples], dim=0)        # (B,L)
    gt_mask = torch.stack([s["gt_mask"] for s in samples], dim=0)
    hist_mask = torch.stack([s["hist_mask"] for s in samples], dim=0)
    cut_length = torch.stack([s["cut_length"] for s in samples], dim=0)

    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "timepoints": timepoints,
        "gt_mask": gt_mask,
        "hist_mask": hist_mask,
        "cut_length": cut_length,
    }


# -----------------------------
# CSDI specialization (same as gait)
# -----------------------------
class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim: int):
        super().__init__(target_dim=target_dim, config=config, device=device)
        self.train_pointwise_mask_ratio = float(config["model"].get("train_pointwise_mask_ratio", 0.3))

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()     # (B,L,K)
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp   = batch["timepoints"].to(self.device).float()        # (B,L)
        gt_mask       = batch["gt_mask"].to(self.device).float()
        cut_length    = batch["cut_length"].to(self.device).long()
        hist_mask     = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask       = gt_mask.permute(0, 2, 1)
        hist_mask     = hist_mask.permute(0, 2, 1)

        return observed_data, observed_mask, observed_tp, gt_mask, hist_mask, cut_length

    def get_randmask(self, observed_mask):
        keep_prob = 1.0 - self.train_pointwise_mask_ratio
        bern = torch.rand_like(observed_mask)
        return (bern < keep_prob).float() * observed_mask


def build_config(args):
    return {
        "model": {
            "timeemb": args.timeemb,
            "featureemb": args.featureemb,
            "is_unconditional": False,
            "target_strategy": "random",
            "train_pointwise_mask_ratio": args.train_pointwise_mask_ratio,
        },
        "diffusion": {
            "layers": args.layers,
            "channels": args.channels,
            "nheads": args.nheads,
            "diffusion_embedding_dim": args.diff_emb_dim,
            "num_steps": args.num_steps,
            "schedule": args.schedule,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "is_linear": args.is_linear,
        },
        "train": {
            "epochs": args.epochs,
            "itr_per_epoch": args.itr_per_epoch,
            "lr": args.lr,
        },
    }


@torch.no_grad()
def eval_loader_mae_mse_rmse_sharedmask(model: CSDI_base, loader: DataLoader, nsample: int, device: torch.device):
    """Weighted MAE/MSE/RMSE over *all* masked points in the loader.

    This avoids batch-size / last-batch bias from averaging per-batch metrics.
    Mask semantics (Option-A):
      - batch["gt_mask"] is keep-mask in (B,L,K) with 1=observed/kept, 0=masked
      - evalmask = 1 - keepmask  (masked points to score)
    """
    sum_abs = 0.0
    sum_sq = 0.0
    denom = 0.0

    for batch in loader:
        # generate samples
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        # shared evalmask from gt_mask (keep-mask)
        keep_BLK = batch["gt_mask"].to(device).float()          # (B,L,K)
        cond_BKL = keep_BLK.permute(0, 2, 1)                    # (B,K,L)
        eval_BKL = 1.0 - cond_BKL                               # (B,K,L)

        # If an observed_mask exists (e.g., real missingness in raw data), don't score those.
        if "observed_mask" in batch:
            obs_BKL = batch["observed_mask"].to(device).float().permute(0, 2, 1)  # (B,K,L)
            eval_BKL = eval_BKL * obs_BKL

        diff = (pred - observed_data) * eval_BKL
        sum_abs += float(diff.abs().sum().item())
        sum_sq  += float((diff ** 2).sum().item())
        denom   += float(eval_BKL.sum().item())

    denom = max(denom, 1.0)
    mae = sum_abs / denom
    mse = sum_sq / denom
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


@torch.no_grad()
def collect_test_arrays_sharedmask(model: CSDI_base, loader: DataLoader, nsample: int, device: torch.device):
    gt_all, imp_all, cond_all, eval_all = [], [], [], []

    for batch in tqdm(loader, desc="collect", leave=False):
        samples, observed_data, _target_mask_internal, *_ = model.evaluate(batch, nsample)
        pred = samples.median(dim=1).values  # (B,K,L)

        keep_BLK = batch["gt_mask"].to(device).float()          # (B,L,K)
        cond_BKL = keep_BLK.permute(0, 2, 1)                    # (B,K,L)
        eval_BKL = 1.0 - cond_BKL                               # (B,K,L)

        imputed_full = cond_BKL * observed_data + (1.0 - cond_BKL) * pred  # (B,K,L)

        gt_BLK = observed_data.permute(0, 2, 1).contiguous()
        imp_BLK = imputed_full.permute(0, 2, 1).contiguous()
        cond_BLK = keep_BLK.contiguous()
        eval_BLK = (1.0 - keep_BLK).contiguous()

        gt_all.append(gt_BLK.cpu().numpy())
        imp_all.append(imp_BLK.cpu().numpy())
        cond_all.append(cond_BLK.cpu().numpy())
        eval_all.append(eval_BLK.cpu().numpy())

    return (
        np.concatenate(gt_all, axis=0),
        np.concatenate(imp_all, axis=0),
        np.concatenate(cond_all, axis=0),
        np.concatenate(eval_all, axis=0),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True, help="physionet2019.csv")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--itr_per_epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--valid_epoch_interval", type=int, default=20)

    ap.add_argument("--train_pointwise_mask_ratio", type=float, default=0.15)

    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", type=str, default="0.10,0.30,0.50,0.70")
    ap.add_argument("--nsample", type=int, default=50)

    ap.add_argument("--timeemb", type=int, default=128)
    ap.add_argument("--featureemb", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--diff_emb_dim", type=int, default=128)

    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--schedule", type=str, default="quad", choices=["quad", "linear"])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=0.2)
    ap.add_argument("--is_linear", action="store_true")

    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="runs_csdi_physionet_sharedmask")
    ap.add_argument("--out_txt", type=str, default=None)
    ap.add_argument("--run_name", type=str, default="csdi_physionet_sharedmask_optionA")

    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default="shared_physionet_maskbank")
    ap.add_argument("--invert_shared_evalmask", action="store_true", help="invert loaded keepmask (debug)")

    ap.add_argument("--save_test_arrays", action="store_true")
    ap.add_argument("--save_test_arrays_dir", type=str, default="saved_test_arrays_csdi_physionet_sharedmask")

    args = ap.parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    out_txt = args.out_txt or os.path.join(args.save_dir, f"{args.run_name}_metrics.txt")

    # --- load maskbank metadata (Option A truth source) ---
    meta = load_maskbank_metadata(args.shared_evalmask_dir, seq_len=args.seq_len, seed=args.seed)
    valid_mask = meta["valid_feature_mask"]
    feat_cols_raw = meta["feature_cols_raw"]
    feat_cols_kept = [c for c, keep in zip(feat_cols_raw, valid_mask) if keep]
    K_kept = len(feat_cols_kept)

    def _to_int_id(x):
        try:
            return int(x)
        except Exception:
            return int(float(str(x)))

    train_ids = [_to_int_id(x) for x in meta["patients_train"].tolist()]
    val_ids   = [_to_int_id(x) for x in meta["patients_val"].tolist()]
    test_ids  = [_to_int_id(x) for x in meta["patients_test"].tolist()]

    # --- load CSV ---
    df = pd.read_csv(args.csv)
    df = normalize_patient_id(df, "Patient_ID")
    time_col = pick_time_col(df)

    missing = [c for c in feat_cols_raw if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} feature columns expected by maskbank metadata. "
                         f"First few: {missing[:10]}")

    # --- window splits (raw, kept feature space) ---
    train_w = make_windows_for_patients(df, train_ids, feat_cols_kept, time_col, args.seq_len)
    val_w   = make_windows_for_patients(df, val_ids,   feat_cols_kept, time_col, args.seq_len)
    test_w  = make_windows_for_patients(df, test_ids,  feat_cols_kept, time_col, args.seq_len)

    # --- standardize using maskbank stats (raw->kept) ---
    mu_raw = meta["train_mu_raw"].astype(np.float32)
    sd_raw = meta["train_sd_raw"].astype(np.float32)

    train_w = standardize_with_maskbank_stats(train_w, mu_raw, sd_raw, valid_mask)
    val_w   = standardize_with_maskbank_stats(val_w,   mu_raw, sd_raw, valid_mask)
    test_w  = standardize_with_maskbank_stats(test_w,  mu_raw, sd_raw, valid_mask)

    print(f"[data] patients train/val/test = {len(set(train_ids))}/{len(set(val_ids))}/{len(set(test_ids))}   "
          f"time_col={'(none)' if not time_col else time_col}")
    print(f"[split] windows train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)}  seq_len={args.seq_len}  K={K_kept}")

    config = build_config(args)
    model = CSDI_Physio(config=config, device=device, target_dim=K_kept).to(device)

    train_loader = DataLoader(
        PhysioCSDIDataset(train_w, mode="train", r_eval_masked=0.1, lm=args.lm),
        batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )
    val_loader_for_trainloop = DataLoader(
        PhysioCSDIDataset(val_w, mode="train", r_eval_masked=0.1, lm=args.lm),
        batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )

    print("[train] training CSDI ...")
    csdi_train(
        model=model,
        config=config["train"],
        train_loader=train_loader,
        valid_loader=val_loader_for_trainloop,
        valid_epoch_interval=args.valid_epoch_interval,
        foldername=args.save_dir,
    )

    eval_masked = [float(x.strip()) for x in args.eval_masked_ratios.split(",") if x.strip()]
    write_header = not os.path.exists(out_txt)
    with open(out_txt, "a") as f:
        if write_header:
            f.write("split\tr_masked\tMAE\tMSE\tRMSE\n")

        for split_name, split_w in [("val", val_w), ("test", test_w)]:
            for r_m in eval_masked:
                pre_keep = None
                if args.use_shared_evalmask:
                    pre_keep = load_shared_keepmask(
                        args.shared_evalmask_dir,
                        split=split_name,
                        r_m=r_m,
                        seq_len=args.seq_len,
                        seed=args.seed,
                        invert=args.invert_shared_evalmask,
                        valid_feature_mask=valid_mask,
                        expected_K=split_w.shape[2],
                    )

                eval_loader = DataLoader(
                    PhysioCSDIDataset(split_w, mode="eval", r_eval_masked=r_m, lm=args.lm, precomputed_keepmask_NLK=pre_keep),
                    batch_size=args.batch, shuffle=False, drop_last=False,
                    num_workers=args.num_workers, collate_fn=collate_batch,
                )

                if args.save_test_arrays and split_name == "test":
                    os.makedirs(args.save_test_arrays_dir, exist_ok=True)
                    gt, imputed, condmask, evalmask = collect_test_arrays_sharedmask(
                        model=model, loader=eval_loader, nsample=args.nsample, device=device
                    )
                    tag = f"csdi_physionet_test_r{r_m:.2f}_seed{args.seed}_ns{args.nsample}_L{args.seq_len}"
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_gt.npy"), gt)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_imputed.npy"), imputed)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_condmask.npy"), condmask)
                    np.save(os.path.join(args.save_test_arrays_dir, f"{tag}_evalmask.npy"), evalmask)

                    np.savez(
                        os.path.join(args.save_test_arrays_dir, f"metadata_seed{args.seed}_L{args.seq_len}.npz"),
                        seq_len=args.seq_len, lm=args.lm,
                        K_kept=K_kept, feature_cols_kept=np.array(feat_cols_kept, dtype=object),
                        train_pointwise_mask_ratio=args.train_pointwise_mask_ratio,
                        shared_evalmask=bool(args.use_shared_evalmask),
                        maskbank_dir=args.shared_evalmask_dir,
                    )
                    print(f"[save] Wrote test arrays to: {args.save_test_arrays_dir}  (r_masked={r_m:.2f})")

                mae, mse, rmse = eval_loader_mae_mse_rmse_sharedmask(model, eval_loader, nsample=args.nsample, device=device)
                f.write(f"{split_name}\t{r_m:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
                f.flush()
                print(f"[{split_name}] r_masked={r_m:.2f}  MAE={mae:.6f}  MSE={mse:.6f}  RMSE={rmse:.6f}")

    print(f"[done] metrics -> {out_txt}")


if __name__ == "__main__":
    main()


