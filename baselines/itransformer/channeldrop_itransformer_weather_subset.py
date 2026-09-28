#!/usr/bin/env python3
import os, sys, json, random, argparse
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
PREFIX = "weather"
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH = 64


# -----------------------------
# Weather feature subset
# -----------------------------
WEATHER_FEATURE_ORDER = [
    "p", "T", "Tpot", "Tdew", "rh", "VPmax", "VPact", "VPdef", "sh", "H2OC",
    "rho", "wv", "max. wv", "wd", "rain", "raining", "SWDR", "PAR", "max. PAR", "Tlog", "OT"
]
WEATHER_SUBSET_FEATURES = ["p", "T", "H2OC", "wv", "max. wv", "wd", "rain", "raining", "PAR", "OT"]
WEATHER_SUBSET_INDICES = [WEATHER_FEATURE_ORDER.index(f) for f in WEATHER_SUBSET_FEATURES]


def subset_weather_windows(windows_NLK: np.ndarray) -> np.ndarray:
    """Keep only the selected low-correlation weather features."""
    windows_NLK = windows_NLK.astype(np.float32)
    max_idx = max(WEATHER_SUBSET_INDICES)
    if windows_NLK.ndim != 3:
        raise ValueError(f"Expected windows with shape (N,L,K), got {windows_NLK.shape}")
    if windows_NLK.shape[-1] <= max_idx:
        raise ValueError(
            f"Weather windows have K={windows_NLK.shape[-1]}, but selected indices require index {max_idx}. "
            "Check WEATHER_FEATURE_ORDER against the preprocessing order."
        )
    return windows_NLK[:, :, WEATHER_SUBSET_INDICES].astype(np.float32)


def _read_maskbank_metadata(path: str):
    obj = np.load(path, allow_pickle=True)
    if "eval_mask" not in obj:
        raise KeyError(f"{path} missing eval_mask")
    full_eval_mask = obj["eval_mask"].astype(np.float32)
    if full_eval_mask.ndim != 3:
        raise ValueError(f"Expected eval_mask with shape (N,L,K), got {full_eval_mask.shape}")
    metadata = {}
    if "metadata_json" in obj:
        try:
            metadata = json.loads(str(obj["metadata_json"]))
        except Exception:
            metadata = {}
    return full_eval_mask, metadata


def _parse_seed_and_drop(path: str, metadata: dict):
    base = os.path.basename(path)
    seed_match = re.search(r"seed(\d+)", base)
    drop_match = re.search(r"drop(\d+)", base)
    seed = int(seed_match.group(1)) if seed_match else int(metadata.get("seed", 1))
    n_drop = int(drop_match.group(1)) if drop_match else int(str(metadata.get("protocol", "drop1")).replace("drop", ""))
    return seed, n_drop


def make_subset_weather_channeldrop_mask(path: str):
    """
    Generate a channel-drop mask over the selected feature subset only.

    The original maskbank was produced for all weather features. Since this experiment uses only
    WEATHER_SUBSET_FEATURES, we regenerate the drop1/drop2 mask over K=len(WEATHER_SUBSET_FEATURES)
    using the seed and protocol encoded in the maskbank filename. This keeps masks identical across
    all models while ensuring drop1/drop2 applies only to the selected features.
    """
    full_eval_mask, metadata = _read_maskbank_metadata(path)
    N, L, _ = full_eval_mask.shape
    K = len(WEATHER_SUBSET_FEATURES)
    seed, n_drop = _parse_seed_and_drop(path, metadata)
    if n_drop > K:
        raise ValueError(f"Cannot drop {n_drop} channels when selected subset has only K={K} features")
    rng = np.random.default_rng(seed)
    eval_mask = np.zeros((N, L, K), dtype=np.float32)
    dropped_channels = np.full((N, n_drop), -1, dtype=np.int64)
    for i in range(N):
        chosen = rng.choice(K, size=n_drop, replace=False)
        dropped_channels[i] = chosen
        eval_mask[i, :, chosen] = 1.0
    metadata = dict(metadata)
    metadata["feature_subset"] = list(WEATHER_SUBSET_FEATURES)
    metadata["feature_subset_indices_from_full_weather"] = [int(i) for i in WEATHER_SUBSET_INDICES]
    metadata["subset_mask_generated_in_script"] = True
    return eval_mask.astype(np.float32), dropped_channels.astype(np.int64), metadata

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_batch_size(requested: int, n_items: int) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(int(requested), int(n_items)))

def load_window_assets(asset_dir: str, seq_len: int):
    train_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_train_windows.npy")).astype(np.float32)
    val_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_val_windows.npy")).astype(np.float32)
    test_w = np.load(os.path.join(asset_dir, f"weather_seq{seq_len}_test_windows.npy")).astype(np.float32)
    train_w = subset_weather_windows(train_w)
    val_w = subset_weather_windows(val_w)
    test_w = subset_weather_windows(test_w)
    print(f"[features] using {WEATHER_SUBSET_FEATURES} from indices {WEATHER_SUBSET_INDICES}")
    return train_w, val_w, test_w
def load_maskbank_npz(path: str):
    return make_subset_weather_channeldrop_mask(path)
def markov_keep_mask(B: int, K: int, L: int, r_masked: float, lm: float, device):
    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    out = np.ones((B, K, L), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            state = int(np.random.rand() < r_keep)
            for t in range(L):
                out[b, k, t] = state
                if np.random.rand() < (p_m if state == 0 else p_u):
                    state = 1 - state
    return torch.from_numpy(out).to(device)

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.x = windows.astype(np.float32)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), idx

def import_itransformer(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from models.iTransformer import Model as iTransformerModel
        return iTransformerModel
    except Exception:
        from model.iTransformer import Model as iTransformerModel
        return iTransformerModel

def build_configs(args, K):
    return SimpleNamespace(
        task_name="imputation",
        seq_len=int(args.seq_len),
        pred_len=int(args.seq_len),
        enc_in=int(K),
        c_out=int(K),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        e_layers=int(args.e_layers),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        factor=int(args.factor),
        activation=str(args.activation),
        embed=str(args.embed),
        freq=str(args.freq),
        output_attention=bool(args.output_attention),
        num_class=int(args.num_class),
        label_len=0,
        dec_in=int(K),
        d_layers=1,
    )

def model_forward_impute(model: nn.Module, x_enc: torch.Tensor) -> torch.Tensor:
    try:
        out = model(x_enc, None, None, None, None)
    except Exception:
        B, L, K = x_enc.shape
        x_mark = torch.zeros((B, L, 4), device=x_enc.device)
        try:
            out = model(x_enc, x_mark, None, None)
        except Exception:
            out = model(x_enc)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.shape[1] == x_enc.shape[2] and out.shape[2] == x_enc.shape[1]:
        out = out.permute(0, 2, 1)
    return out

def train_epoch(model, loader, opt, device, K, L, r_masked, lm, clip_grad):
    model.train()
    losses = []
    for xb, _ in loader:
        xb = xb.to(device).float()
        keep = markov_keep_mask(xb.shape[0], K, L, r_masked, lm, device).permute(0, 2, 1).contiguous()
        x_obs = xb * keep
        pred = model_forward_impute(model, x_obs)
        evalmask = 1.0 - keep
        denom = evalmask.sum().clamp_min(1.0)
        loss = ((pred - xb).abs() * evalmask).sum() / denom
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")

@torch.no_grad()
def evaluate_split(model, loader, evalmask_nlk, device):
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    denom = 0.0
    total_keep = 0.0
    total_n = 0.0
    evalmask_all = torch.from_numpy(evalmask_nlk).to(device)
    for xb, idx in loader:
        xb = xb.to(device).float()
        idx = idx.to(device)
        evalmask = evalmask_all.index_select(0, idx)
        keep = 1.0 - evalmask
        pred = model_forward_impute(model, xb * keep)
        diff = (pred - xb) * evalmask
        sum_abs += diff.abs().sum().item()
        sum_sq += (diff ** 2).sum().item()
        denom += evalmask.sum().item()
        total_keep += keep.sum().item()
        total_n += keep.numel()
    denom = max(denom, 1.0)
    mae = sum_abs / denom
    mse = sum_sq / denom
    rmse = float(np.sqrt(mse))
    r_obs = total_keep / max(total_n, 1.0)
    return mae, mse, rmse, r_obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itransformer_root", type=str)
    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_grad", type=float, default=5.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--embed", type=str, default="fixed")
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument("--output_attention", action="store_true")
    ap.add_argument("--num_class", type=int, default=2)
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--out_txt", type=str, default="itransformer_metrics.txt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train, val, test = load_window_assets(args.asset_dir, args.seq_len)
    val_evalmask, _, _ = load_maskbank_npz(args.val_maskbank)
    test_evalmask, _, _ = load_maskbank_npz(args.test_maskbank)

    if val_evalmask.shape != val.shape:
        raise ValueError(f"val mask mismatch: {val_evalmask.shape} vs {val.shape}")
    if test_evalmask.shape != test.shape:
        raise ValueError(f"test mask mismatch: {test_evalmask.shape} vs {test.shape}")

    K = int(train.shape[-1]); L = int(args.seq_len)
    print(f"[data] train/val/test={len(train)}/{len(val)}/{len(test)}  K={K} L={L}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    train_loader = DataLoader(WindowDataset(train), batch_size=safe_batch_size(args.batch, len(train)), shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowDataset(val), batch_size=safe_batch_size(args.batch, len(val)), shuffle=False, drop_last=False)
    test_loader = DataLoader(WindowDataset(test), batch_size=safe_batch_size(args.batch, len(test)), shuffle=False, drop_last=False)

    iTransformerModel = import_itransformer(args.itransformer_root)
    model = iTransformerModel(build_configs(args, K)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, device, K, L, args.r_train_masked, args.lm, args.clip_grad)
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[epoch {ep:03d}] loss={loss:.6f}")

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")
        for split_name, loader, evalmask_nlk in [("val", val_loader, val_evalmask), ("test", test_loader, test_evalmask)]:
            mae, mse, rmse, r_obs = evaluate_split(model, loader, evalmask_nlk, device)
            f.write(f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
            print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}")

    print(f"Done. Results appended to: {args.out_txt}")

if __name__ == "__main__":
    main()
