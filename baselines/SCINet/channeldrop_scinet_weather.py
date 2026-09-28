#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PREFIX = "weather"
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH = 64
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
    return (np.load(f("train_windows")).astype('float32'),
            np.load(f("val_windows")).astype('float32'),
            np.load(f("test_windows")).astype('float32'))

def load_mask(p):
    return np.load(p, allow_pickle=True)["eval_mask"].astype('float32')

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
