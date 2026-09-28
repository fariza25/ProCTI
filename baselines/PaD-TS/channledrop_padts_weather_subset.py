#!/usr/bin/env python3
import os, sys, json, random, argparse
import numpy as np
import torch
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
class WindowDataset(Dataset):
    def __init__(self, windows_nlk: np.ndarray):
        self.x = windows_nlk.astype(np.float32)
    def __len__(self):
        return int(self.x.shape[0])
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])

def import_padts(root: str):
    if root not in sys.path:
        sys.path.insert(0, root)
    from Model import PaD_TS
    from diffmodel_init import create_gaussian_diffusion
    from training import Trainer
    from resample import UniformSampler, Batch_Same_Sampler
    return PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler

def diffusion_impute_project(model, diffusion, x0_blk: torch.Tensor, keep_blk: torch.Tensor):
    device = x0_blk.device
    B, L, K = x0_blk.shape
    x_t = torch.randn_like(x0_blk)
    eps_obs = torch.randn_like(x0_blk)
    T = diffusion.num_timesteps
    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        out = diffusion.p_sample(model, x_t, t_tensor, clip_denoised=True)
        x_prev = out["sample"]
        if t > 0:
            t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
            x_obs_prev = diffusion.q_sample(x0_blk, t_prev, noise=eps_obs)
        else:
            x_obs_prev = x0_blk
        x_t = keep_blk * x_obs_prev + (1.0 - keep_blk) * x_prev
    return x_t

@torch.no_grad()
def evaluate_split(model, diffusion, windows_nlk: np.ndarray, evalmask_nlk: np.ndarray, device, batch_size: int):
    ds = WindowDataset(windows_nlk)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    sum_abs = 0.0
    sum_sq = 0.0
    sum_den = 0.0
    global_idx = 0
    for x in loader:
        x = x.to(device)
        B = x.shape[0]
        this_evalmask = torch.from_numpy(evalmask_nlk[global_idx:global_idx + B]).to(device)
        global_idx += B
        keep = 1.0 - this_evalmask
        xhat = diffusion_impute_project(model, diffusion, x, keep)
        diff = (xhat - x) * this_evalmask
        sum_abs += diff.abs().sum().item()
        sum_sq += (diff ** 2).sum().item()
        sum_den += this_evalmask.sum().item()
    denom = max(sum_den, 1.0)
    mae = sum_abs / denom
    mse = sum_sq / denom
    rmse = float(np.sqrt(mse))
    r_obs = float((1.0 - evalmask_nlk).mean())
    return mae, mse, rmse, r_obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--padts_root", type=str, required=True)
    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--train_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--n_encoder", type=int, default=1)
    ap.add_argument("--n_decoder", type=int, default=3)
    ap.add_argument("--feature_last", action="store_true", default=True)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--diffusion_steps", type=int, default=250)
    ap.add_argument("--noise_schedule", type=str, default="cosine")
    ap.add_argument("--loss", type=str, default="MSE_MMD")
    ap.add_argument("--predict_xstart", action="store_true", default=True)
    ap.add_argument("--rescale_timesteps", action="store_true", default=False)
    ap.add_argument("--schedule_sampler", type=str, default="batch", choices=["batch", "uniform"])
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--mmd_alpha", type=float, default=0.0005)
    ap.add_argument("--save_dir", type=str, default="OUTPUT/padts_channeldrop/")
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--out_txt", type=str, default="padts_metrics.txt")
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

    K = train.shape[2]
    L = train.shape[1]
    print(f"[data] train/val/test={len(train)}/{len(val)}/{len(test)}  K={K} L={L}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    PaD_TS, create_gaussian_diffusion, Trainer, UniformSampler, Batch_Same_Sampler = import_padts(args.padts_root)

    model = PaD_TS(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        feature_last=args.feature_last,
        mlp_ratio=args.mlp_ratio,
        input_shape=(args.seq_len, K),
    )

    diffusion = create_gaussian_diffusion(
        predict_xstart=args.predict_xstart,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        loss=args.loss,
        rescale_timesteps=args.rescale_timesteps,
    )

    train_bs = safe_batch_size(args.batch, len(train))
    train_loader = DataLoader(
        WindowDataset(train),
        batch_size=train_bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if args.schedule_sampler == "batch":
        schedule_sampler = Batch_Same_Sampler(diffusion)
    else:
        schedule_sampler = UniformSampler(diffusion)

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=train_bs,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.train_steps,
        save_dir=args.save_dir if args.save_dir.endswith("/") else args.save_dir + "/",
        mmd_alpha=args.mmd_alpha,
    )

    print("====== Training ======")
    trainer.train()
    print("====== Training done ======")

    model.eval()
    model.to(device)

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")
        for split_name, windows_nlk, evalmask_nlk in [
            ("val", val, val_evalmask),
            ("test", test, test_evalmask),
        ]:
            mae, mse, rmse, r_obs = evaluate_split(
                model=model,
                diffusion=diffusion,
                windows_nlk=windows_nlk,
                evalmask_nlk=evalmask_nlk,
                device=device,
                batch_size=train_bs,
            )
            f.write(f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
            print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}")

    print(f"Done. Results appended to: {args.out_txt}")

if __name__ == "__main__":
    main()

