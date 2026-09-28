#!/usr/bin/env python3
import os, sys, time, random, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
import re
DATASET = "weather"
PREFIX = "weather"
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH = 32


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

def import_tider_class(tider_py: str):
    old_argv = sys.argv[:]
    try:
        sys.argv = [tider_py]
        spec = importlib.util.spec_from_file_location("tider_mod", tider_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    if not hasattr(mod, "TIDER"):
        raise AttributeError(f"{tider_py} loaded, but no TIDER class found.")
    return mod.TIDER

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
    eval_mask, _, _ = make_subset_weather_channeldrop_mask(path)
    return eval_mask
def windows_to_TK(w_nlk: np.ndarray) -> np.ndarray:
    return w_nlk.reshape(-1, w_nlk.shape[-1]).astype(np.float32)

def markov_keep_mask_KT(K: int, T: int, r_masked: float, lm: float, rng: np.random.Generator) -> np.ndarray:
    r_keep = 1.0 - r_masked
    p_m = 1.0 / lm
    p_u = p_m * (1.0 - r_keep) / max(r_keep, 1e-12)
    p = [p_m, p_u]
    out = np.ones((K, T), dtype=np.float32)
    for k in range(K):
        state = int(rng.random() < r_keep)
        for t in range(T):
            out[k, t] = state
            if rng.random() < p[state]:
                state = 1 - state
    return out

def obs_mse_loss(Xhat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(X)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=X.device)
    return F.mse_loss(Xhat[mask], X[mask])

def evaluate_maskbank(model, X_full_KT, start_t: int, end_t: int, keep_seg_KT: np.ndarray):
    model.eval()
    with torch.no_grad():
        Xhat = model(torch.arange(X_full_KT.shape[0], device=X_full_KT.device, dtype=torch.long))
        gt_seg = X_full_KT[:, start_t:end_t]
        pred_seg = Xhat[:, start_t:end_t]
        keep_t = torch.from_numpy(keep_seg_KT).to(X_full_KT.device)
        obs_seg = torch.isfinite(gt_seg).float()
        evalmask = (1.0 - keep_t) * obs_seg
        denom = torch.clamp(evalmask.sum(), min=1.0)
        diff = pred_seg - gt_seg
        mae = (torch.abs(diff) * evalmask).sum() / denom
        mse = ((diff ** 2) * evalmask).sum() / denom
        rmse = torch.sqrt(mse + 1e-12)
        r_obs = float((keep_t * obs_seg).sum().item() / max(obs_seg.sum().item(), 1.0))
    return float(mae.item()), float(mse.item()), float(rmse.item()), r_obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tider_py", type=str)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--lambda_ar", type=float, default=0.2)
    ap.add_argument("--lambda_trend", type=float, default=0.1)
    ap.add_argument("--dim_size", type=int, default=50)
    ap.add_argument("--bias_dimension", type=int, default=5)
    ap.add_argument("--lag_list", type=str, default="list(range(5))")
    ap.add_argument("--season_num", type=int, default=30)
    ap.add_argument("--seasonality", type=float, default=168.0)
    ap.add_argument("--r_train_masked", type=float, default=0.15)
    ap.add_argument("--lm", type=float, default=6.0)
    ap.add_argument("--val_maskbank", type=str, required=True)
    ap.add_argument("--test_maskbank", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="drop1")
    ap.add_argument("--out_txt", type=str, default="tider_metrics.txt")
    ap.add_argument("--save_path", type=str, default="TIDER_channeldrop.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    TIDER = import_tider_class(args.tider_py)
    lag_list = eval(args.lag_list)

    train_w, val_w, test_w = load_window_assets(args.asset_dir, args.seq_len)
    val_evalmask = load_maskbank_npz(args.val_maskbank)
    test_evalmask = load_maskbank_npz(args.test_maskbank)
    if val_evalmask.shape != val_w.shape:
        raise ValueError(f"val mask shape mismatch: {val_evalmask.shape} vs {val_w.shape}")
    if test_evalmask.shape != test_w.shape:
        raise ValueError(f"test mask shape mismatch: {test_evalmask.shape} vs {test_w.shape}")

    Ntr, Nva, Nte = len(train_w), len(val_w), len(test_w)
    K = int(train_w.shape[-1]); L = int(train_w.shape[1])
    Ttr, Tva, Tte = Ntr * L, Nva * L, Nte * L
    full_TK = np.concatenate([windows_to_TK(train_w), windows_to_TK(val_w), windows_to_TK(test_w)], axis=0)
    T_full = int(full_TK.shape[0])
    X_full_KT = torch.from_numpy(full_TK.T).to(device)
    tr0, tr1 = 0, Ttr
    va0, va1 = Ttr, Ttr + Tva
    te0, te1 = Ttr + Tva, T_full

    print(f"[data] train/val/test={Ntr}/{Nva}/{Nte}  K={K} L={L}")
    print(f"[maskbank] val={args.val_maskbank}")
    print(f"[maskbank] test={args.test_maskbank}")
    print(f"[protocol] {args.protocol}")

    X_train_KT = torch.full((K, T_full), float("nan"), device=device)
    X_train_KT[:, tr0:tr1] = X_full_KT[:, tr0:tr1]
    rng_train = np.random.default_rng(args.seed + 12345)
    keep_train = markov_keep_mask_KT(K, Ttr, args.r_train_masked, args.lm, rng_train)
    keep_train_t = torch.from_numpy(keep_train).to(device)
    X_train_KT[:, tr0:tr1] = torch.where(keep_train_t > 0, X_train_KT[:, tr0:tr1], torch.tensor(float("nan"), device=device))

    X_val_KT = torch.full((K, T_full), float("nan"), device=device)
    X_val_KT[:, va0:va1] = X_full_KT[:, va0:va1]

    model = TIDER(
        n=K,
        t=T_full,
        hid_size=args.dim_size,
        bias_lag_list=lag_list,
        bias_dim_=args.bias_dimension,
        season_num_=args.season_num,
        seasonality_=args.seasonality,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def l2loss():
        if args.eta <= 0: return torch.tensor(0.0, device=device)
        roads_all = torch.arange(K, device=device)
        times_all = torch.arange(T_full, device=device)
        return args.eta * (torch.linalg.norm(model.getu(roads_all)) + torch.linalg.norm(model.getv(times_all)))

    def arloss_bias():
        if args.lambda_ar <= 0: return torch.tensor(0.0, device=device)
        times_all = torch.arange(T_full, device=device)
        return args.lambda_ar * torch.linalg.norm(model.bias_loss(times_all))

    def trend_loss():
        if args.lambda_trend <= 0: return torch.tensor(0.0, device=device)
        times_all = torch.arange(T_full, device=device)
        trend = model.t_embeddings_trend(times_all)
        return args.lambda_trend * torch.linalg.norm(trend[:, 1:] - trend[:, :-1])

    best_val = float("inf")
    step_bs = max(1, min(args.batch_size, K))
    idx = np.arange(K)

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)
        losses = []
        t0 = time.time()
        for st in range(0, K, step_bs):
            roads = torch.from_numpy(idx[st:st + step_bs]).to(device=device, dtype=torch.long)
            Xhat = model(roads)
            Xobs = X_train_KT[roads]
            loss = obs_mse_loss(Xhat, Xobs) + l2loss() + arloss_bias() + trend_loss()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        with torch.no_grad():
            model.eval()
            Xhat_full = model(torch.arange(K, device=device, dtype=torch.long))
            val_loss = float(obs_mse_loss(Xhat_full[:, va0:va1], X_val_KT[:, va0:va1]).item())

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "best_val": best_val, "args": vars(args)}, args.save_path)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(f"[epoch {ep:03d}] train_loss={mean_loss:.6f} val_obs_mse={val_loss:.6f} time={time.time()-t0:.1f}s")

    if os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)

    val_keep = 1.0 - val_evalmask.reshape(-1, K).T
    test_keep = 1.0 - test_evalmask.reshape(-1, K).T

    write_header = not os.path.exists(args.out_txt)
    with open(args.out_txt, "a") as f:
        if write_header:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")
        for split_name, s0, s1, keep in [("val", va0, va1, val_keep), ("test", te0, te1, test_keep)]:
            mae, mse, rmse, r_obs = evaluate_maskbank(model, X_full_KT, s0, s1, keep)
            f.write(f"{split_name}\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
            print(f"[{split_name}] protocol={args.protocol} MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f} r_obs={r_obs:.2f}")

    print(f"Done. Results appended to: {args.out_txt}")

if __name__ == "__main__":
    main()

