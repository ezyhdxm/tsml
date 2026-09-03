#!/usr/bin/env python3
"""Reproducible structural replication of volatility-surface VAEs.

Generates 6x7 discretely arbitrage-free SSVI surfaces, trains masked MLP and
2-D convolutional VAEs, and evaluates missing-surface completion and prior
sampling. This is a structural replication, not a reproduction on proprietary
FX/SPX or Binance source data.
"""
from __future__ import annotations

import argparse, json, os, platform, random, time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TENOR_DAYS = np.array([14, 30, 60, 90, 120, 180], dtype=float)
TENORS = TENOR_DAYS / 365.0
KLOG = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], dtype=float)
NT, NK, P = len(TENORS), len(KLOG), len(TENORS) * len(KLOG)


@dataclass(frozen=True)
class Config:
    seed: int = 20260903
    n: int = 1800
    train_frac: float = 0.70
    val_frac: float = 0.15
    z_dim: int = 16
    mlp_hidden: int = 64
    conv_hidden: int = 16
    beta: float = 1e-3
    obs_weight: float = 0.1
    batch: int = 128
    lr: float = 1e-3
    epochs: int = 45
    patience: int = 12
    mask_min: float = 0.10
    mask_max: float = 0.50
    arb_weight: float = 25.0
    pca_dim: int = 8
    n_gen: int = 1800
    bootstrap: int = 500


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def ssvi(sig_s: np.ndarray, sig_l: np.ndarray, rho: np.ndarray,
         eta: np.ndarray, kappa: float = 3.0, gamma: float = 0.5) -> np.ndarray:
    """SSVI surface with a positive, increasing ATM total-variance curve."""
    t = TENORS[None, :]
    vs, vl = sig_s[:, None] ** 2, sig_l[:, None] ** 2
    theta = vl * t + (vs - vl) * (1.0 - np.exp(-kappa * t)) / kappa
    theta = np.maximum(theta, 1e-12)
    th = theta[:, :, None]
    k = KLOG[None, None, :]
    r = rho[:, None, None]
    phi = eta[:, None, None] / th ** gamma
    pk = phi * k
    w = 0.5 * th * (1.0 + r * pk + np.sqrt((pk + r) ** 2 + 1.0 - r ** 2))
    return np.sqrt(np.maximum(w / t[:, :, None], 1e-12))


def calls_np(iv: np.ndarray) -> np.ndarray:
    t, k = TENORS[None, :, None], KLOG[None, None, :]
    w = np.maximum(iv ** 2 * t, 1e-12); root = np.sqrt(w)
    d1 = -k / root + 0.5 * root; d2 = d1 - root
    return ndtr(d1) - np.exp(k) * ndtr(d2)


def arb_flags(iv: np.ndarray, tol: float = 2e-7) -> dict[str, np.ndarray]:
    """Discrete calendar, vertical-spread, and butterfly tests at F=1."""
    c = calls_np(iv)
    calendar = np.any(c[:, :-1] - c[:, 1:] > tol, axis=(1, 2))
    strike = np.exp(KLOG)
    slopes = np.diff(c, axis=2) / np.diff(strike)[None, None, :]
    butterfly = (np.any(slopes > tol, axis=(1, 2)) |
                 np.any(slopes < -1.0 - tol, axis=(1, 2)) |
                 np.any(np.diff(slopes, axis=2) < -tol, axis=(1, 2)))
    positive = np.any(~np.isfinite(iv) | (iv <= 0), axis=(1, 2))
    return {"calendar": calendar, "butterfly": butterfly,
            "positive": positive, "all": calendar | butterfly | positive}


def make_data(cfg: Config) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    surfaces, parameters = [], []
    got = 0
    while got < cfg.n:
        m = max(cfg.n, 4096)
        sig_s = rng.uniform(0.16, 0.62, m)
        term = rng.uniform(-0.55, 0.35, m)
        sig_l = sig_s * np.exp(term)
        rho = rng.uniform(-0.90, -0.05, m)
        cap = np.minimum(np.sqrt(3.6 / (1 + np.abs(rho))), 1.25)
        eta = rng.uniform(0.20, 0.95, m) * cap
        iv = ssvi(sig_s, sig_l, rho, eta)
        keep = ~arb_flags(iv)["all"]
        surfaces.append(iv[keep])
        parameters.append(np.c_[sig_s[keep], term[keep], rho[keep], eta[keep]])
        got += int(keep.sum())
    iv = np.concatenate(surfaces)[:cfg.n]
    pars = np.concatenate(parameters)[:cfg.n]
    return iv, pd.DataFrame(pars, columns=["sigma_short", "term_log_ratio", "rho", "eta"])


def eval_mask(n: int, scheme: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.ones((n, NT, NK), dtype=np.float32)
    if scheme.startswith("random_"):
        q = float(scheme.rsplit("_", 1)[1]) / 100
        m = (rng.random(m.shape) >= q).astype(np.float32)
        flat = m.reshape(n, P)
        for i in range(n):
            if flat[i].sum() == 0: flat[i, rng.integers(P)] = 1
            if flat[i].sum() == P: flat[i, rng.integers(P)] = 0
        return flat
    if scheme == "row": m[np.arange(n), rng.integers(NT, size=n), :] = 0
    elif scheme == "column": m[np.arange(n), :, rng.integers(NK, size=n)] = 0
    elif scheme == "wings": m[:, :, [0, -1]] = 0
    elif scheme == "long_tenor": m[:, -1, :] = 0
    else: raise ValueError(scheme)
    return m.reshape(n, P)


class MLPVAE(nn.Module):
    def __init__(self, z: int, h: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2*P, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.mu, self.logvar = nn.Linear(h, z), nn.Linear(h, z)
        self.decoder = nn.Sequential(nn.Linear(z, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, P))
    def encode(self, x, mask):
        h = self.encoder(torch.cat([x*mask, mask], 1)); return self.mu(h), self.logvar(h)
    def decode(self, z): return self.decoder(z)
    def forward(self, x, mask):
        mu, lv = self.encode(x, mask)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv) if self.training else mu
        return self.decode(z), mu, lv


class ConvVAE(nn.Module):
    def __init__(self, z: int, h: int):
        super().__init__(); self.h = h
        self.enc = nn.Sequential(nn.Conv2d(2,h,3,padding=1),nn.GELU(),
            nn.Conv2d(h,h,3,padding=1),nn.GELU(),nn.Conv2d(h,h,3,padding=1),nn.GELU())
        f = h*P; self.mu, self.logvar = nn.Linear(f,z), nn.Linear(f,z)
        self.dec0 = nn.Linear(z,f)
        self.dec = nn.Sequential(nn.Conv2d(h,h,3,padding=1),nn.GELU(),
            nn.Conv2d(h,h,3,padding=1),nn.GELU(),nn.Conv2d(h,h,3,padding=1),nn.GELU(),nn.Conv2d(h,1,1))
    def encode(self, x, mask):
        a = torch.cat([(x*mask).view(-1,1,NT,NK), mask.view(-1,1,NT,NK)], 1)
        h = self.enc(a).flatten(1); return self.mu(h), self.logvar(h)
    def decode(self, z): return self.dec(self.dec0(z).view(-1,self.h,NT,NK)).flatten(1)
    def forward(self, x, mask):
        mu, lv = self.encode(x, mask)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv) if self.training else mu
        return self.decode(z), mu, lv

def train_mask(n: int, g: torch.Generator, cfg: Config) -> torch.Tensor:
    q = torch.empty((n,1)).uniform_(cfg.mask_min, cfg.mask_max, generator=g)
    return (torch.rand((n,P), generator=g) >= q).float()


def iv_from_x_torch(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.exp((mean + std*x).clamp(-5,1.5)).view(-1,NT,NK)


def calls_torch(iv: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(TENORS, dtype=iv.dtype).view(1,NT,1)
    k = torch.as_tensor(KLOG, dtype=iv.dtype).view(1,1,NK)
    w = (iv.square()*t).clamp_min(1e-10); root = torch.sqrt(w)
    d1 = -k/root + .5*root; d2 = d1-root
    return torch.special.ndtr(d1) - torch.exp(k)*torch.special.ndtr(d2)


def arb_penalty(iv: torch.Tensor) -> tuple[torch.Tensor,float,float]:
    c = calls_torch(iv)
    cal = torch.relu(c[:,:-1]-c[:,1:]).square().mean()
    strike = torch.exp(torch.as_tensor(KLOG, dtype=iv.dtype))
    s = torch.diff(c,dim=2)/torch.diff(strike).view(1,1,-1)
    bf = (torch.relu(s).square().mean() + torch.relu(-1-s).square().mean() +
          torch.relu(-torch.diff(s,dim=2)).square().mean())
    return cal+bf, float(cal.detach()), float(bf.detach())


def batch_loss(model, x, mask, cfg, mean, std, use_arb):
    recon, mu, lv = model(x,mask)
    sq = (recon-x).square(); hidden=1-mask
    lh=(sq*hidden).sum(1)/hidden.sum(1).clamp_min(1)
    lo=(sq*mask).sum(1)/mask.sum(1).clamp_min(1)
    kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1)
    ap=torch.tensor(0.); cal=bf=0.0
    if use_arb: ap,cal,bf=arb_penalty(iv_from_x_torch(recon,mean,std))
    loss=(lh+cfg.obs_weight*lo+cfg.beta*kl).mean()+cfg.arb_weight*ap
    return loss,{"loss":float(loss.detach()),"hidden":float(lh.mean().detach()),"observed":float(lo.mean().detach()),
                 "kl":float(kl.mean().detach()),"arb":float(ap.detach()),"arb_calendar":cal,"arb_butterfly":bf}


def weighted(rows):
    n=sum(n for _,n in rows); keys=rows[0][0]
    return {k:sum(r[k]*m for r,m in rows)/n for k in keys}


def fit_model(name,model,train_x,val_x,mean,std,cfg,use_arb,out):
    mean_t=torch.tensor(mean,dtype=torch.float32); std_t=torch.tensor(std,dtype=torch.float32)
    loader=DataLoader(TensorDataset(torch.tensor(train_x)),batch_size=cfg.batch,shuffle=True,
                      generator=torch.Generator().manual_seed(cfg.seed+301))
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    mg=torch.Generator().manual_seed(cfg.seed+1000+sum(map(ord,name)))
    vm=torch.tensor(eval_mask(len(val_x),"random_30",cfg.seed+2000))
    vx=torch.tensor(val_x); best=float("inf"); state=None; left=cfg.patience; hist=[]; t0=time.time()
    for ep in range(1,cfg.epochs+1):
        model.train(); rr=[]
        for (xb,) in loader:
            mb=train_mask(len(xb),mg,cfg); opt.zero_grad(set_to_none=True)
            loss,parts=batch_loss(model,xb,mb,cfg,mean_t,std_t,use_arb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10); opt.step(); rr.append((parts,len(xb)))
        tr=weighted(rr); model.eval(); rr=[]
        with torch.no_grad():
            for i in range(0,len(vx),cfg.batch):
                _,parts=batch_loss(model,vx[i:i+cfg.batch],vm[i:i+cfg.batch],cfg,mean_t,std_t,use_arb)
                rr.append((parts,len(vx[i:i+cfg.batch])))
        va=weighted(rr); row={"model":name,"epoch":ep,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}; hist.append(row)
        if va["loss"]<best-1e-6:
            best=va["loss"]; state={k:v.detach().clone() for k,v in model.state_dict().items()}; left=cfg.patience
        else: left-=1
        if ep==1 or ep%25==0:
            print(f"[{name:10s}] ep={ep:3d} train_h={tr['hidden']:.4f} val_h={va['hidden']:.4f} val={va['loss']:.4f} patience={left}",flush=True)
        if left<=0:
            print(f"[{name}] early stop at {ep}; {time.time()-t0:.1f}s",flush=True); break
    if state is None: raise RuntimeError("no checkpoint")
    model.load_state_dict(state); torch.save({"state_dict":state,"config":asdict(cfg)},out/f"{name}.pt")
    return model,hist


def predict(model,x,mask,batch=512):
    model.eval(); ans=[]
    with torch.no_grad():
        for i in range(0,len(x),batch):
            y,_,_=model(torch.tensor(x[i:i+batch]),torch.tensor(mask[i:i+batch])); ans.append(y.numpy())
    return np.concatenate(ans)


def pca_predict(pca,x,mask,ridge=1e-4):
    comp,mu=pca.components_,pca.mean_; eye=np.eye(comp.shape[0]); out=np.empty_like(x)
    for i in range(len(x)):
        obs=mask[i].astype(bool); a=comp[:,obs].T; b=x[i,obs]-mu[obs]
        z=np.linalg.solve(a.T@a+ridge*eye,a.T@b); out[i]=mu+z@comp
    return out


def to_iv(x,mean,std): return np.exp(np.clip(mean[None]+std[None]*x,-5,1.5)).reshape(-1,NT,NK)


def hidden_rmse(pred,true,mask):
    e=(pred.reshape(len(pred),P)-true.reshape(len(true),P))**2; h=1-mask
    return np.sqrt((e*h).sum(1)/h.sum(1).clip(min=1))*100


def boot_ci(x,reps,seed):
    rng=np.random.default_rng(seed); n=len(x)
    vals=np.array([x[rng.integers(0,n,n)].mean() for _ in range(reps)])
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def gen_stats(name,model,mean,std,cfg,scale):
    g=torch.Generator().manual_seed(cfg.seed+int(scale*1000)+sum(map(ord,name)))
    z=torch.randn((cfg.n_gen,cfg.z_dim),generator=g)*scale; model.eval()
    with torch.no_grad(): iv=to_iv(model.decode(z).numpy(),mean,std)
    f=arb_flags(iv)
    return {"model":name,"scale":scale,"valid_pct":float(100*(~f['all']).mean()),
            "calendar_valid_pct":float(100*(~f['calendar']).mean()),
            "butterfly_valid_pct":float(100*(~f['butterfly']).mean()),
            "iv_p01":float(np.quantile(iv,.01)),"iv_p50":float(np.quantile(iv,.5)),"iv_p99":float(np.quantile(iv,.99))}


def latent_means(model,x):
    out=[]; mask=np.ones_like(x,dtype=np.float32); model.eval()
    with torch.no_grad():
        for i in range(0,len(x),512): out.append(model.encode(torch.tensor(x[i:i+512]),torch.tensor(mask[i:i+512]))[0].numpy())
    return np.concatenate(out)

def save_plots(out,iv,hist,completion,generation,corr,sample):
    # Surface and smile slices.
    kk,tt=np.meshgrid(KLOG,TENOR_DAYS)
    fig=plt.figure(figsize=(7.2,5.2)); ax=fig.add_subplot(111,projection="3d")
    ax.plot_surface(kk,tt,100*iv,cmap="viridis",edgecolor="none"); ax.set(xlabel="log-forward-moneyness k",ylabel="maturity (days)",zlabel="implied vol (%)",title="Synthetic arbitrage-free SSVI surface")
    fig.tight_layout(); fig.savefig(out/"sample_surface_3d.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.5))
    for j in [0,2,5]: ax.plot(KLOG,100*iv[j],marker="o",label=f"{int(TENOR_DAYS[j])}d")
    ax.set(xlabel="log-forward-moneyness k",ylabel="implied vol (%)",title="One surface, three smiles"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"sample_smiles.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    # Curves.
    fig,ax=plt.subplots(figsize=(8,4.8))
    for name,g in hist.groupby("model",sort=False): ax.plot(g.epoch,g.val_hidden,label=name)
    ax.set(xlabel="epoch",ylabel="validation hidden-cell MSE",title="Training curves"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"training_curves.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    # Completion grouped bars.
    d=pd.DataFrame(completion); schemes=list(dict.fromkeys(d.scheme)); models=list(dict.fromkeys(d.model)); x=np.arange(len(schemes)); w=.8/len(models)
    fig,ax=plt.subplots(figsize=(10.5,5.2))
    for j,m in enumerate(models):
        q=d[d.model.eq(m)].set_index("scheme").loc[schemes]; ax.bar(x+(j-(len(models)-1)/2)*w,q.rmse_vol_points,w,label=m)
    ax.set_xticks(x,schemes,rotation=25,ha="right"); ax.set(ylabel="hidden-cell RMSE (vol points)",title="Structural replication: completion"); ax.legend(ncol=2); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(out/"completion_results.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    # Generation validity.
    d=pd.DataFrame(generation); p=d.pivot(index="model",columns="scale",values="valid_pct"); x=np.arange(len(p)); w=.35
    fig,ax=plt.subplots(figsize=(8,4.8))
    for j,s in enumerate(p.columns): ax.bar(x+(j-(len(p.columns)-1)/2)*w,p[s],w,label=f"z scale={s:g}")
    ax.set_xticks(x,p.index); ax.set_ylim(0,105); ax.set(ylabel="discretely arbitrage-free (%)",title="Prior sampling validity"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(out/"generation_validity.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    # Latent correlations.
    fig,ax=plt.subplots(figsize=(10,4.2)); im=ax.imshow(corr,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_yticks(range(4),["short level","term ratio","rho (skew)","eta (smile)"]); ax.set_xticks(range(corr.shape[1]),[f"z{i+1}" for i in range(corr.shape[1])],rotation=45); ax.set_title("True factors versus encoded latent means")
    fig.colorbar(im,ax=ax,label="Pearson correlation"); fig.tight_layout(); fig.savefig(out/"latent_factor_correlations.png",dpi=170,bbox_inches="tight"); plt.close(fig)
    # One row-hole example: observed, truth, and four reconstructions.
    items=[("Observed",sample["observed"]),("Truth",sample["truth"]),("PCA-8",sample["PCA-8"]),("MLP-VAE",sample["MLP-VAE"]),("ConvVAE",sample["ConvVAE"]),("ConvVAE+NA",sample["ConvVAE+NA"])]
    vals=[x for _,x in items]; lo=min(np.nanmin(x) for x in vals)*100; hi=max(np.nanmax(x) for x in vals)*100
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,(title,a) in zip(axs.flat,items):
        im=ax.imshow(100*a,origin="lower",aspect="auto",vmin=lo,vmax=hi); ax.set_title(title); ax.set_xticks(range(NK),[f"{v:.1f}" for v in KLOG],rotation=45); ax.set_yticks(range(NT),[f"{int(v)}d" for v in TENOR_DAYS]); ax.set_xlabel("k"); ax.set_ylabel("tenor")
    fig.colorbar(im,ax=axs,shrink=.78,label="implied vol (%)"); fig.savefig(out/"completion_example.png",dpi=170,bbox_inches="tight"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",type=Path,default=Path("reproduction_output")); ap.add_argument("--quick",action="store_true"); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=Config()
    if args.quick: cfg=replace(cfg,n=1200,epochs=12,patience=5,n_gen=300,bootstrap=50,conv_hidden=24,mlp_hidden=64)
    seed_all(cfg.seed); print("Generating SSVI surfaces...",flush=True)
    iv,factors=make_data(cfg); assert not arb_flags(iv)["all"].any()
    ntr=int(cfg.n*cfg.train_frac); nv=int(cfg.n*cfg.val_frac); nt=cfg.n-ntr-nv
    tr,va,te=iv[:ntr],iv[ntr:ntr+nv],iv[ntr+nv:]; tf=factors.iloc[ntr+nv:].reset_index(drop=True)
    log=np.log(tr.reshape(ntr,P)); mean=log.mean(0); std=log.std(0).clip(min=1e-6)
    transform=lambda a:((np.log(a.reshape(len(a),P))-mean)/std).astype(np.float32)
    xtr,xv,xt=transform(tr),transform(va),transform(te)
    np.savez_compressed(args.output_dir/"synthetic_ssvi_data.npz",iv=iv.astype(np.float32),factors=factors.to_numpy(np.float32),tenors_days=TENOR_DAYS,log_moneyness=KLOG,train_size=ntr,val_size=nv,test_size=nt,log_mean=mean,log_std=std)
    factors.to_csv(args.output_dir/"synthetic_ssvi_factors.csv",index=False)
    pca=PCA(n_components=cfg.pca_dim,random_state=cfg.seed).fit(xtr)
    print("Training models...",flush=True)
    specs=[("MLP-VAE",MLPVAE(cfg.z_dim,cfg.mlp_hidden),False),("ConvVAE",ConvVAE(cfg.z_dim,cfg.conv_hidden),False),("ConvVAE+NA",ConvVAE(cfg.z_dim,cfg.conv_hidden),True)]
    models={}; hh=[]
    for name,model,arb in specs:
        models[name],h=fit_model(name,model,xtr,xv,mean,std,cfg,arb,args.output_dir); hh+=h
    hist=pd.DataFrame(hh); hist.to_csv(args.output_dir/"training_history.csv",index=False)
    schemes=["random_10","random_30","random_50","row","column","wings","long_tenor"]
    completion=[]; cache={}; masks={}
    print("Evaluating completion...",flush=True)
    for si,scheme in enumerate(schemes):
        mask=eval_mask(nt,scheme,cfg.seed+5000+si); masks[scheme]=mask
        methods={"Mean":np.zeros_like(xt),f"PCA-{cfg.pca_dim}":pca_predict(pca,xt,mask)}
        methods.update({name:predict(model,xt,mask) for name,model in models.items()})
        for mi,(name,predx) in enumerate(methods.items()):
            piv=to_iv(predx,mean,std); e=hidden_rmse(piv,te,mask); ci=boot_ci(e,cfg.bootstrap,cfg.seed+7000+31*si+mi)
            completion.append({"scheme":scheme,"model":name,"rmse_vol_points":float(e.mean()),"median_vol_points":float(np.median(e)),"ci95_low":ci[0],"ci95_high":ci[1],"n_test":nt,"hidden_cells_mean":float((1-mask).sum(1).mean())}); cache[(scheme,name)]=piv
    print("Evaluating prior generation...",flush=True)
    generation=[gen_stats(name,model,mean,std,cfg,s) for name,model in models.items() for s in (1.0,1.5)]
    full=np.ones_like(xt); reconstruction=[]
    for name,model in models.items():
        piv=to_iv(predict(model,xt,full),mean,std); e=np.sqrt(np.mean((piv-te)**2,axis=(1,2)))*100; flags=arb_flags(piv)
        reconstruction.append({"model":name,"rmse_vol_points":float(e.mean()),"valid_pct":float(100*(~flags['all']).mean())})
    z=latent_means(models["ConvVAE"],xt); corr=np.corrcoef(tf.to_numpy().T,z.T)[:4,4:]
    idx=min(17,nt-1); mask=masks["row"][idx].reshape(NT,NK); sample={"truth":te[idx],"observed":np.where(mask>0,te[idx],np.nan)}
    for name in [f"PCA-{cfg.pca_dim}","MLP-VAE","ConvVAE","ConvVAE+NA"]: sample[name]=cache[("row",name)][idx]
    rows=[]
    for i,t in enumerate(TENOR_DAYS):
        for j,k in enumerate(KLOG):
            r={"tenor_days":int(t),"log_moneyness":float(k),"true_iv":float(te[idx,i,j]),"observed":int(mask[i,j])}; r.update({name:float(sample[name][i,j]) for name in [f"PCA-{cfg.pca_dim}","MLP-VAE","ConvVAE","ConvVAE+NA"]}); rows.append(r)
    pd.DataFrame(rows).to_csv(args.output_dir/"sample_completion.csv",index=False)
    save_plots(args.output_dir,te[idx],hist,completion,generation,corr,sample)
    results={
      "run_timestamp_utc":pd.Timestamp.now(tz="UTC").isoformat(),"experiment_type":"structural replication on synthetic arbitrage-free SSVI surfaces",
      "not_exact_replication_reason":"Original FX/SPX sources are proprietary; the public Binance binary archive could not be transferred into this execution sandbox.",
      "config":asdict(cfg),"environment":{"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"device":"cpu"},
      "grid":{"tenors_days":TENOR_DAYS.tolist(),"log_moneyness":KLOG.tolist()},"data":{"n_total":cfg.n,"n_train":ntr,"n_val":nv,"n_test":nt,"source_valid_pct":100.0,"factor_ranges_realized":{c:[float(factors[c].min()),float(factors[c].max())] for c in factors}},
      "models":{name:{"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)} for name,model in models.items()},
      "pca":{"components":cfg.pca_dim,"explained_variance_ratio_sum":float(pca.explained_variance_ratio_.sum()),"explained_variance_ratio":pca.explained_variance_ratio_.tolist()},
      "completion":completion,"generation":generation,"full_reconstruction":reconstruction,
      "latent_factor_audit":{"factor_names":tf.columns.tolist(),"correlation_matrix":corr.tolist(),"max_abs_correlation_by_factor":dict(zip(tf.columns,np.max(np.abs(corr),axis=1).tolist())),"warning":"VAE coordinates are not identifiable; correlations are descriptive."}}
    (args.output_dir/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({"random_50":[r for r in completion if r['scheme']=='random_50'],"row":[r for r in completion if r['scheme']=='row'],"generation":generation},indent=2),flush=True)

if __name__=="__main__": main()
