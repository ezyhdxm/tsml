from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent / 'reproduction'
OUT.mkdir(exist_ok=True)

# Black call price and vega as functions of implied volatility.
F,K,T=1.0,1.0,0.5
sig=np.linspace(0.02,1.0,300)
root=sig*np.sqrt(T)
d1=(np.log(F/K)+0.5*sig**2*T)/root
d2=d1-root
price=F*ndtr(d1)-K*ndtr(d2)
vega=F*np.exp(-0.5*d1**2)/np.sqrt(2*np.pi)*np.sqrt(T)
fig,ax=plt.subplots(figsize=(7.4,4.6))
ax.plot(sig,price,label='normalized call price')
ax.plot(sig,vega,label='vega = dC/dsigma')
ax.set_xlabel('volatility sigma')
ax.set_ylabel('value')
ax.set_title('Black price is strictly increasing in volatility')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUT/'black_price_and_vega.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Decreasing and convex call price across strike.
S=1.0; T=0.5; sigma=.28
Kgrid=np.linspace(.55,1.55,300)
r=sigma*np.sqrt(T)
d1=(np.log(S/Kgrid)+.5*sigma**2*T)/r; d2=d1-r
C=S*ndtr(d1)-Kgrid*ndtr(d2)
fig,ax=plt.subplots(figsize=(7.4,4.6))
ax.plot(Kgrid,C)
pts=np.array([.82,1.0,1.22])
d1p=(np.log(S/pts)+.5*sigma**2*T)/r; d2p=d1p-r
Cp=S*ndtr(d1p)-pts*ndtr(d2p)
ax.scatter(pts,Cp,zorder=4)
ax.plot(pts,Cp,linestyle='--',label='three strikes form a butterfly test')
ax.set_xlabel('strike K'); ax.set_ylabel('call price C(K)')
ax.set_title('No static arbitrage: call price decreases and is convex in strike')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUT/'call_convexity.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Total variance versus raw volatility term structure.
t=np.linspace(.03,2.0,250)
sigterm=.22+.13*np.exp(-2.2*t)  # decreasing IV but increasing total variance
w=sigterm**2*t
fig,ax=plt.subplots(figsize=(7.4,4.6))
ax.plot(t,sigterm,label='implied volatility sigma(T)')
ax.plot(t,w,label='total variance w(T)=sigma(T)^2 T')
ax.set_xlabel('maturity T (years)'); ax.set_ylabel('value')
ax.set_title('Calendar no-arbitrage does not require raw IV to increase')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(OUT/'iv_vs_total_variance.png',dpi=180,bbox_inches='tight'); plt.close(fig)
