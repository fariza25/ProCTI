import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

from models.iTransformer import Model as iTransformer


# ------------------------------------------------
# Repro
# ------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ratios(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_batch_size(requested, n):
    if n <= 0:
        return 1
    return max(1, min(int(requested), int(n)))


# ------------------------------------------------
# Markov masking
# ------------------------------------------------

def markov_keep_mask(B, K, L, r_masked, lm, device):
    """
    Returns keep-mask with shape (B,K,L), where 1=observed/kept and 0=masked.
    """
    assert 0.0 < r_masked < 1.0
    assert lm > 0

    r_keep = 1.0 - r_masked

    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)

    out = np.ones((B, K, L), dtype=np.float32)

    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)  # 1=keep, 0=masked
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < (p_m if state == 0 else p_u):
                    state = 1 - state

    return torch.from_numpy(out).to(device=device)


# ------------------------------------------------
# Shared maskbank loading
# ------------------------------------------------

def _candidate_maskbank_names(split, r_masked, seq_len, seed):
    """
    Stock maskbanks have appeared under a few naming conventions across scripts.
    Try the common ones before failing.
    """
    return [
        f"stock_{split}_keepmask_r{r_masked:.2f}_L{seq_len}_seed{seed}.npy",
        f"stock_{split}_keepmask_r{r_masked:.1f}_L{seq_len}_seed{seed}.npy",
        f"stock_{split}_keepmask_r{r_masked:.2f}_seq{seq_len}_seed{seed}.npy",
        f"stock_seq{seq_len}_{split}_keepmask_r{r_masked:.2f}_seed{seed}.npy",
    ]


def load_shared_keepmask(mask_dir, split, r_masked, seq_len, seed, invert=False):
    if mask_dir is None:
        raise ValueError("shared_evalmask_dir is None")

    tried = []
    path = None
    for name in _candidate_maskbank_names(split, r_masked, seq_len, seed):
        p = os.path.join(mask_dir, name)
        tried.append(p)
        if os.path.exists(p):
            path = p
            break

    if path is None:
        tried_msg = "\n  ".join(tried)
        raise FileNotFoundError(
            f"Could not find stock shared keepmask for split={split}, r={r_masked:.2f}, "
            f"L={seq_len}, seed={seed}. Tried:\n  {tried_msg}"
        )

    keep = np.load(path).astype(np.float32)

    if keep.ndim != 3:
        raise ValueError(f"Shared keepmask must be 3D, got shape {keep.shape} from {path}")

    # Expected by this script: (N,L,K). If saved as (N,K,L), convert it.
    if keep.shape[1] == seq_len:
        keep_NLK = keep
    elif keep.shape[2] == seq_len:
        keep_NLK = np.transpose(keep, (0, 2, 1))
        print(f"[maskbank] transposed shared keepmask from (N,K,L) to (N,L,K): {path}")
    else:
        raise ValueError(
            f"Cannot infer shared keepmask layout from shape {keep.shape}; "
            f"expected seq_len={seq_len} on axis 1 or 2. File: {path}"
        )

    if invert or os.environ.get("STOCK_MASKBANK_INVERT", "0") == "1":
        keep_NLK = 1.0 - keep_NLK
        print(f"[maskbank] inverted shared keepmask: {path}")

    print(f"[maskbank] loaded {path} shape={keep_NLK.shape} mean_keep={keep_NLK.mean():.4f}")
    return keep_NLK.astype(np.float32)


def align_keepmask_count(keep_NLK, n_data, split, r_masked):
    """
    Prefer exact alignment. If a maskbank has too many masks, truncate.
    If it has too few masks, tile to avoid crashing, but warn loudly.
    Exact regenerated maskbanks are still preferred for final fair reporting.
    """
    n_mask = int(keep_NLK.shape[0])

    if n_mask == n_data:
        return keep_NLK

    if n_mask > n_data:
        print(
            f"[warning] shared maskbank has more masks than data for split={split}, r={r_masked:.2f}: "
            f"maskbank={n_mask}, data={n_data}. Truncating to data length."
        )
        return keep_NLK[:n_data]

    # n_mask < n_data
    reps = int(np.ceil(n_data / max(n_mask, 1)))
    print(
        f"[warning] shared maskbank has fewer masks than data for split={split}, r={r_masked:.2f}: "
        f"maskbank={n_mask}, data={n_data}. Repeating masks to avoid crash. "
        f"Regenerate maskbank for final fair results."
    )
    return np.tile(keep_NLK, (reps, 1, 1))[:n_data]


# ------------------------------------------------
# Dataset
# ------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.x = windows.astype(np.float32)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


# ------------------------------------------------
# Load stock CSV
# ------------------------------------------------

def load_stock_csv(csv):
    df = pd.read_csv(csv)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.select_dtypes(include=[np.number])

    X = df.values.astype(np.float32)
    X[~np.isfinite(X)] = np.nan

    return X


# ------------------------------------------------
# Row split -> window
# ------------------------------------------------

def split_rows_then_window(X, seq_len):
    T, K = X.shape

    train_end = int(0.70 * T)
    val_end = int(0.85 * T)

    Xtr = X[:train_end]
    Xva = X[train_end:val_end]
    Xte = X[val_end:]

    def window(x):
        n = len(x) // seq_len
        if n <= 0:
            raise ValueError(
                f"No windows formed for split with {len(x)} rows and seq_len={seq_len}."
            )
        return x[: n * seq_len].reshape(n, seq_len, K)

    return window(Xtr), window(Xva), window(Xte)


# ------------------------------------------------
# Standardise by train
# ------------------------------------------------

def standardize(train, val, test):
    train_flat = train.reshape(-1, train.shape[-1])
    mu = np.nanmean(train_flat, axis=0)
    sd = np.nanstd(train_flat, axis=0)

    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    def norm(x):
        x = x.copy().astype(np.float32)
        nan = ~np.isfinite(x)
        if nan.any():
            x[nan] = np.take(mu, np.where(nan)[2])
        x = (x - mu[None, None, :]) / sd[None, None, :]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values after standardization.")
        return x.astype(np.float32)

    return norm(train), norm(val), norm(test)


# ------------------------------------------------
# iTransformer wrapper
# ------------------------------------------------

class ITransformerWrapper(nn.Module):
    def __init__(self, K, L):
        super().__init__()

        configs = SimpleNamespace(
            task_name="imputation",
            seq_len=L,
            label_len=0,
            pred_len=0,
            enc_in=K,
            dec_in=K,
            c_out=K,
            d_model=64,
            d_ff=128,
            e_layers=2,
            d_layers=1,
            n_heads=4,
            dropout=0.1,
            factor=1,
            output_attention=False,
            embed="timeF",
            freq="h",
            activation="gelu",
        )

        self.model = iTransformer(configs)

    def forward(self, x):
        B, L, K = x.shape
        x_mark = torch.zeros((B, L, 4), device=x.device)

        try:
            out = self.model(x, x_mark, None, None)
        except TypeError:
            out = self.model(x)

        if isinstance(out, (tuple, list)):
            out = out[0]

        # Some iTransformer variants return (B,K,L); normalize to (B,L,K).
        if out.ndim == 3 and out.shape[1] == K and out.shape[2] == L:
            out = out.permute(0, 2, 1).contiguous()

        if out.shape != x.shape:
            raise RuntimeError(f"Model output shape {tuple(out.shape)} does not match input shape {tuple(x.shape)}")

        return out


# ------------------------------------------------
# Training
# ------------------------------------------------

def train_epoch(model, loader, opt, device, K, L, r_masked, lm):
    model.train()
    losses = []

    for x in loader:
        x = x.to(device).float()

        keep = markov_keep_mask(len(x), K, L, r_masked, lm, device).permute(0, 2, 1).contiguous()
        x_mask = x * keep

        pred = model(x_mask)
        evalmask = 1.0 - keep
        denom = evalmask.sum().clamp_min(1.0)
        loss = (torch.abs(pred - x) * evalmask).sum() / denom

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

@torch.no_grad()
def eval_split(
    model,
    loader,
    split,
    ratios,
    device,
    K,
    L,
    lm,
    use_shared_evalmask=False,
    shared_evalmask_dir=None,
    seed=1,
    invert_shared_keepmask=False,
):
    model.eval()
    rows = []
    n_data = len(loader.dataset)

    for r in ratios:
        total_abs = 0.0
        total_sq = 0.0
        total_n = 0.0
        total_keep = 0.0
        total_keep_count = 0.0

        keep_NLK = None
        offset = 0

        if use_shared_evalmask:
            keep_NLK = load_shared_keepmask(
                shared_evalmask_dir,
                split=split,
                r_masked=r,
                seq_len=L,
                seed=seed,
                invert=invert_shared_keepmask,
            )

            if keep_NLK.shape[1] != L:
                raise ValueError(f"Shared keepmask L mismatch: mask L={keep_NLK.shape[1]}, data L={L}")
            if keep_NLK.shape[2] != K:
                raise ValueError(f"Shared keepmask K mismatch: mask K={keep_NLK.shape[2]}, data K={K}")

            keep_NLK = align_keepmask_count(keep_NLK, n_data=n_data, split=split, r_masked=r)

        for x in loader:
            x = x.to(device).float()
            B = x.shape[0]

            if keep_NLK is not None:
                keep_slice = keep_NLK[offset:offset + B]
                if keep_slice.shape[0] != B:
                    raise RuntimeError(
                        f"Internal mask slicing error for split={split}, r={r:.2f}: "
                        f"requested B={B}, got {keep_slice.shape[0]}, offset={offset}, "
                        f"maskbank size={keep_NLK.shape[0]}"
                    )
                keep = torch.from_numpy(keep_slice).to(device=device).float()
                offset += B
            else:
                keep = markov_keep_mask(B, K, L, r, lm, device).permute(0, 2, 1).contiguous()

            pred = model(x * keep)
            evalmask = 1.0 - keep
            diff = (pred - x) * evalmask

            total_abs += float(diff.abs().sum().item())
            total_sq += float((diff ** 2).sum().item())
            total_n += float(evalmask.sum().item())

            total_keep += float(keep.sum().item())
            total_keep_count += float(keep.numel())

        mae = total_abs / max(total_n, 1.0)
        mse = total_sq / max(total_n, 1.0)
        rmse = float(np.sqrt(mse))
        r_observed = total_keep / max(total_keep_count, 1.0)

        rows.append((split, r, r_observed, mae, mse, rmse))

    return rows


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--seq_len", type=int, default=48)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--eval_masked_ratios", default="0.10,0.30,0.50,0.70")

    # Shared eval masks expected by your 10-seed runner.
    ap.add_argument("--use_shared_evalmask", action="store_true")
    ap.add_argument("--shared_evalmask_dir", type=str, default=None)
    ap.add_argument("--invert_shared_keepmask", action="store_true")

    ap.add_argument("--out_txt", default="itransformer_stock_metrics.txt")

    args = ap.parse_args()

    if args.use_shared_evalmask and not args.shared_evalmask_dir:
        raise ValueError("--use_shared_evalmask requires --shared_evalmask_dir")

    set_seed(args.seed)
    device = torch.device(args.device)
    ratios = parse_ratios(args.eval_masked_ratios)

    X = load_stock_csv(args.csv)
    train, val, test = split_rows_then_window(X, args.seq_len)
    train, val, test = standardize(train, val, test)

    K = int(train.shape[-1])
    L = int(args.seq_len)

    print(f"[data] train/val/test windows = {len(train)}/{len(val)}/{len(test)}  L={L} K={K}")

    train_loader = DataLoader(
        WindowDataset(train),
        batch_size=safe_batch_size(args.batch, len(train)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindowDataset(val),
        batch_size=safe_batch_size(args.batch, len(val)),
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        WindowDataset(test),
        batch_size=safe_batch_size(args.batch, len(test)),
        shuffle=False,
        drop_last=False,
    )

    model = ITransformerWrapper(K, L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, opt, device, K, L, args.r_train_masked, args.lm)
        if ep == 1 or ep % 10 == 0:
            print(f"[train] epoch={ep:03d} loss={loss:.6f} time={time.time() - t0:.1f}s")

    val_rows = eval_split(
        model,
        val_loader,
        "val",
        ratios,
        device,
        K,
        L,
        args.lm,
        use_shared_evalmask=args.use_shared_evalmask,
        shared_evalmask_dir=args.shared_evalmask_dir,
        seed=args.seed,
        invert_shared_keepmask=args.invert_shared_keepmask,
    )

    test_rows = eval_split(
        model,
        test_loader,
        "test",
        ratios,
        device,
        K,
        L,
        args.lm,
        use_shared_evalmask=args.use_shared_evalmask,
        shared_evalmask_dir=args.shared_evalmask_dir,
        seed=args.seed,
        invert_shared_keepmask=args.invert_shared_keepmask,
    )

    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    new_file = (not os.path.exists(args.out_txt)) or (os.path.getsize(args.out_txt) == 0)

    with open(args.out_txt, "a") as f:
        if new_file:
            f.write("split\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for rows in [val_rows, test_rows]:
            for sp, r_masked, r_observed, mae, mse, rmse in rows:
                f.write(
                    f"{sp}\t{r_masked:.2f}\t{r_observed:.2f}\t"
                    f"{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n"
                )
                print(
                    f"[{sp}] r_masked={r_masked:.2f} r_obs={r_observed:.2f}  "
                    f"MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}"
                )


if __name__ == "__main__":
    main()

