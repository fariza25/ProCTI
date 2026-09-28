import os, sys, json, random, argparse
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PREFIX = "physionet"
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH = 64


TARGET_FEATURES = ["SBP", "O2Sat", "MAP", "Resp", "HR"]
CANONICAL_PHYSIONET_INDEX = {
    "HR": 0,
    "O2Sat": 1,
    "SBP": 3,
    "MAP": 4,
    "Resp": 6,
}

def get_target_feature_indices(num_features: int):
    if num_features == len(TARGET_FEATURES):
        return list(range(len(TARGET_FEATURES)))
    needed = max(CANONICAL_PHYSIONET_INDEX[name] for name in TARGET_FEATURES)
    if num_features <= needed:
        raise ValueError(
            f"Cannot select {TARGET_FEATURES} from tensor with K={num_features}. "
            "Expected either the original PhysioNet channel order or an already-filtered 5-channel tensor."
        )
    return [CANONICAL_PHYSIONET_INDEX[name] for name in TARGET_FEATURES]

def restrict_physionet_windows(windows_NLK: np.ndarray):
    idx = get_target_feature_indices(int(windows_NLK.shape[-1]))
    return windows_NLK[:, :, idx].astype(np.float32), idx

def restrict_physionet_evalmask(evalmask_NLK: np.ndarray, idx):
    return evalmask_NLK[:, :, idx].astype(np.float32)

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
    train_path = os.path.join(asset_dir, f"{PREFIX}_seq{seq_len}_train_windows.npy")
    val_path   = os.path.join(asset_dir, f"{PREFIX}_seq{seq_len}_val_windows.npy")
    test_path  = os.path.join(asset_dir, f"{PREFIX}_seq{seq_len}_test_windows.npy")
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing window asset: {p}")
    return (np.load(train_path).astype(np.float32),
            np.load(val_path).astype(np.float32),
            np.load(test_path).astype(np.float32))

def load_maskbank_npz(path: str):
    obj = np.load(path, allow_pickle=True)
    if "eval_mask" not in obj:
        raise KeyError(f"{path} missing eval_mask")
    eval_mask = obj["eval_mask"].astype(np.float32)
    dropped_channels = obj["dropped_channels"].astype(np.int64) if "dropped_channels" in obj else None
    meta = {}
    if "metadata_json" in obj:
        try:
            meta = json.loads(str(obj["metadata_json"]))
        except Exception:
            meta = {}
    return eval_mask, dropped_channels, meta

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

    train, feat_idx = restrict_physionet_windows(train)
    val, _ = restrict_physionet_windows(val)
    test, _ = restrict_physionet_windows(test)
    val_evalmask = restrict_physionet_evalmask(val_evalmask, feat_idx)
    test_evalmask = restrict_physionet_evalmask(test_evalmask, feat_idx)

    if val_evalmask.shape != val.shape:
        raise ValueError(f"val mask mismatch: {val_evalmask.shape} vs {val.shape}")
    if test_evalmask.shape != test.shape:
        raise ValueError(f"test mask mismatch: {test_evalmask.shape} vs {test.shape}")

    K = int(train.shape[-1]); L = int(args.seq_len)
    print(f"[data] train/val/test={len(train)}/{len(val)}/{len(test)}  K={K} L={L}")
    print(f"[features] using {TARGET_FEATURES} from indices {feat_idx}")
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

