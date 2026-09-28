import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.SCINet import SCINet


# -------------------------------------------------
# Repro
# -------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_batch_size(requested, n):

    if n <= 0:
        return 1

    return max(1, min(requested, n))


# -------------------------------------------------
# Markov masking
# -------------------------------------------------

def markov_keep_mask(B,K,L,r_masked,lm,device):

    r_keep = 1 - r_masked

    p_m = 1/lm
    p_u = p_m*(1-r_keep)/r_keep

    p=[p_m,p_u]

    out=np.ones((B,K,L),dtype=np.float32)

    for b in range(B):
        for k in range(K):

            state=int(np.random.rand()<r_keep)

            for t in range(L):

                out[b,k,t]=state

                if np.random.rand()<p[state]:
                    state=1-state

    return torch.from_numpy(out).to(device)


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class WindowDataset(Dataset):

    def __init__(self,x):

        self.x=x.astype(np.float32)

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self,i):

        return torch.from_numpy(self.x[i])


# -------------------------------------------------
# Data loading
# -------------------------------------------------

def load_weather_windows(csv,seq_len):

    df=pd.read_csv(csv)

    if "date" in df.columns:
        df=df.sort_values("date")

    num_cols=df.select_dtypes(include=[np.number]).columns

    X=df[num_cols].values.astype(np.float32)

    T,K=X.shape

    N=T//seq_len

    X=X[:N*seq_len].reshape(N,seq_len,K)

    return X


def chrono_split(X):

    N=len(X)

    n_train=int(N*0.7)
    n_val=int(N*0.15)

    train=X[:n_train]
    val=X[n_train:n_train+n_val]
    test=X[n_train+n_val:]

    return train,val,test


def standardize(train,val,test):

    mu=train.reshape(-1,train.shape[-1]).mean(0)
    sd=train.reshape(-1,train.shape[-1]).std(0)

    sd=np.maximum(sd,1e-6)

    def z(x):

        return ((x-mu)/sd).astype(np.float32)

    return z(train),z(val),z(test)


# -------------------------------------------------
# SCINet wrapper
# -------------------------------------------------

class SCINetWrapper(nn.Module):

    def __init__(self,K,seq_len):

        super().__init__()

        self.model=SCINet(
            output_len=seq_len,
            input_len=seq_len,
            input_dim=K,
            hid_size=1,
            num_stacks=1,
            num_levels=3
        )

    def forward(self,x):

        return self.model(x)


# -------------------------------------------------
# Train
# -------------------------------------------------

def train_epoch(model,loader,opt,device,lm,r_train):

    model.train()

    total_loss=0

    for x in loader:

        x=x.to(device)

        B,L,K=x.shape

        keep=markov_keep_mask(B,K,L,r_train,lm,device).permute(0,2,1)

        x_masked=x*keep

        pred=model(x_masked)

        mask=1-keep

        loss=((pred-x)**2*mask).sum()/mask.sum()

        opt.zero_grad()

        loss.backward()

        opt.step()

        total_loss+=loss.item()

    return total_loss/len(loader)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

@torch.no_grad()
def eval_model(model,loader,device,lm,r):

    model.eval()

    sum_abs=0
    sum_sq=0
    denom=0

    for x in loader:

        x=x.to(device)

        B,L,K=x.shape

        keep=markov_keep_mask(B,K,L,r,lm,device).permute(0,2,1)

        x_masked=x*keep

        pred=model(x_masked)

        mask=1-keep

        diff=(pred-x)*mask

        sum_abs+=diff.abs().sum().item()

        sum_sq+=(diff**2).sum().item()

        denom+=mask.sum().item()

    mae=sum_abs/denom
    mse=sum_sq/denom
    rmse=np.sqrt(mse)

    return mae,mse,rmse


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--csv",required=True)

    ap.add_argument("--seq_len",type=int,default=96)

    ap.add_argument("--epochs",type=int,default=50)

    ap.add_argument("--batch",type=int,default=64)

    ap.add_argument("--lr",type=float,default=1e-3)

    ap.add_argument("--seed",type=int,default=1)

    ap.add_argument("--lm",type=float,default=6.0)

    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--out_txt",default="scinet_weather_5runs_metrics.txt")

    args=ap.parse_args()

    set_seed(args.seed)

    device=torch.device(args.device)

    X=load_weather_windows(args.csv,args.seq_len)

    train,val,test=chrono_split(X)

    train,val,test=standardize(train,val,test)

    print("Total windows:",len(X))

    train_loader=DataLoader(
        WindowDataset(train),
        batch_size=safe_batch_size(args.batch,len(train)),
        shuffle=True
    )

    val_loader=DataLoader(
        WindowDataset(val),
        batch_size=safe_batch_size(args.batch,len(val))
    )

    test_loader=DataLoader(
        WindowDataset(test),
        batch_size=safe_batch_size(args.batch,len(test))
    )

    K=train.shape[-1]

    model=SCINetWrapper(K,args.seq_len).to(device)

    opt=torch.optim.Adam(model.parameters(),lr=args.lr)

    for epoch in range(args.epochs):

        loss=train_epoch(model,train_loader,opt,device,args.lm,0.15)

        print(f"epoch {epoch} loss {loss:.6f}")

    ratios=[0.1,0.3,0.5,0.7]

    file_exists=os.path.exists(args.out_txt)

    with open(args.out_txt,"a") as f:

        if not file_exists:

            f.write("seed\tsplit\tr_masked\tr_observed\tMAE\tMSE\tRMSE\n")

        for split,loader in [("val",val_loader),("test",test_loader)]:

            for r in ratios:

                mae,mse,rmse=eval_model(model,loader,device,args.lm,r)

                f.write(
                    f"{args.seed}\t{split}\t{r:.2f}\t{1-r:.2f}\t{mae:.6f}\t{mse:.6f}\t{rmse:.6f}\n"
                )

                print(split,r,mae,mse,rmse)


if __name__=="__main__":
    main()
