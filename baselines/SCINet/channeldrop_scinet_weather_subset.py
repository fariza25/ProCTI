#!/usr/bin/env python3
import os, sys, random, argparse
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
def markov_keep_mask(B, K, L, r_masked=0.15, lm=6.0, device="cpu"):
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

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_bs(b, n): return max(1, min(int(b), int(n)))

def load_assets(d, pfx, L):
    def f(x): return os.path.join(d, f"{pfx}_seq{L}_{x}.npy")
    train_w = np.load(f("train_windows")).astype("float32")
    val_w = np.load(f("val_windows")).astype("float32")
    test_w = np.load(f("test_windows")).astype("float32")
    train_w = subset_weather_windows(train_w)
    val_w = subset_weather_windows(val_w)
    test_w = subset_weather_windows(test_w)
    print(f"[features] using {WEATHER_SUBSET_FEATURES} from indices {WEATHER_SUBSET_INDICES}")
    return train_w, val_w, test_w
def load_mask(p):
    eval_mask, _, _ = make_subset_weather_channeldrop_mask(p)
    return eval_mask
class DS(Dataset):
    def __init__(self, x): self.x=x
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return torch.from_numpy(self.x[i]), i

def import_scinet(root):
    if root not in sys.path: sys.path.insert(0, root)
    from models.SCINet import SCINet
    return SCINet

class Wrap(nn.Module):
    def __init__(self, SCINet, K, L):
        super().__init__()
        self.m = SCINet(output_len=L,input_len=L,input_dim=K,hid_size=1,num_stacks=1,num_levels=3)
    def forward(self,x): return self.m(x)
def markov_keep_mask(B, K, L, r_masked=0.15, lm=6.0, device="cpu"):
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
def train_ep(m, loader, opt, dev):
    m.train(); losses=[]
    for x,_ in loader:
        x=x.to(dev)
        y=m(x)
        loss=((y-x)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def eval_run(m, loader, eval_mask, dev, save_arrays=False, out_prefix=None, split_name="test"):
    m.eval()
    eval_mask = torch.from_numpy(eval_mask).to(dev)   # 1 = dropped target

    mae = mse = den = 0.0

    gt_list = []
    imputed_list = []
    evalmask_list = []
    keepmask_list = []

    for x, idx in loader:
        x = x.to(dev)
        em = eval_mask[idx]         # (B,L,K), 1 where target is hidden
        keep = 1.0 - em             # observed part only

        x_in = x * keep
        y = m(x_in)

        d = (y - x) * em
        mae += d.abs().sum().item()
        mse += (d ** 2).sum().item()
        den += em.sum().item()

        if save_arrays:
            gt_list.append(x.detach().cpu().numpy())
            imputed_list.append(y.detach().cpu().numpy())
            evalmask_list.append(em.detach().cpu().numpy())
            keepmask_list.append(keep.detach().cpu().numpy())

    den = max(den, 1.0)
    mae /= den
    mse /= den
    rmse = float(np.sqrt(mse))

    if save_arrays:
        if out_prefix is None:
            out_prefix = f"scinet_{PREFIX}"

        gt = np.concatenate(gt_list, axis=0)
        imputed = np.concatenate(imputed_list, axis=0)
        evalmask_np = np.concatenate(evalmask_list, axis=0)
        keepmask_np = np.concatenate(keepmask_list, axis=0)

        np.save(f"{out_prefix}_{split_name}_gt.npy", gt)
        np.save(f"{out_prefix}_{split_name}_imputed.npy", imputed)
        np.save(f"{out_prefix}_{split_name}_evalmask.npy", evalmask_np)
        np.save(f"{out_prefix}_{split_name}_keepmask.npy", keepmask_np)

    return mae, mse, rmse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scinet_root',required=True)
    ap.add_argument('--asset_dir',required=True)
    ap.add_argument('--seq_len',type=int,default=DEFAULT_SEQ_LEN)
    ap.add_argument('--batch',type=int,default=DEFAULT_BATCH)
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--val_maskbank',required=True)
    ap.add_argument('--test_maskbank',required=True)
    ap.add_argument('--protocol',default='drop1')
    ap.add_argument('--out_txt',required=True)
    ap.add_argument('--save_test_arrays', action='store_true')
    ap.add_argument('--array_out_prefix', type=str, default='scinet_arrays')
    ap.add_argument('--r_train_masked', type=float, default=0.15)
    ap.add_argument('--lm', type=float, default=6.0)
    args=ap.parse_args()

    set_seed(args.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr,va,te=load_assets(args.asset_dir,PREFIX,args.seq_len)
    vm=load_mask(args.val_maskbank)
    tm=load_mask(args.test_maskbank)

    trL=DataLoader(DS(tr),batch_size=safe_bs(args.batch,len(tr)),shuffle=True)
    vaL=DataLoader(DS(va),batch_size=safe_bs(args.batch,len(va)))
    teL=DataLoader(DS(te),batch_size=safe_bs(args.batch,len(te)))

    SCINet=import_scinet(args.scinet_root)
    m=Wrap(SCINet,tr.shape[-1],args.seq_len).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)

    for ep in range(1,args.epochs+1):
        l = train_ep(m, trL, opt, dev)
        if ep==1 or ep%10==0:
            print(f"[epoch {ep}] loss={l:.4f}")

    head = not os.path.exists(args.out_txt)
    with open(args.out_txt, 'a') as f:
        if head:
            f.write("split\tprotocol\tMAE\tMSE\tRMSE\n")

        mae, mse, rmse = eval_run(
            m, vaL, vm, dev,
            save_arrays=False,
            out_prefix=args.array_out_prefix,
            split_name="val"
        )
        f.write(f"val\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print("val", mae, rmse)

        mae, mse, rmse = eval_run(
            m, teL, tm, dev,
            save_arrays=args.save_test_arrays,
            out_prefix=args.array_out_prefix,
            split_name="test"
        )
        f.write(f"test\t{args.protocol}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n")
        print("test", mae, rmse)

if __name__=='__main__':
    main()

