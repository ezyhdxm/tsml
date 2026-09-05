"""Public synthetic demonstrations; no real or connected financial data.
Run: python experiments.py. Requires only NumPy. Seeds and DGP are fixed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPEATS = 200

def ols_predict(x, y, train, test):
    xtr = np.column_stack([np.ones(len(train)), x[train]])
    beta = np.linalg.lstsq(xtr, y[train], rcond=None)[0]
    return np.column_stack([np.ones(len(test)), x[test]]) @ beta

def mae(x, y, train, test):
    return float(np.mean(np.abs(y[test] - ols_predict(x, y, train, test))))

def random_cv(x, y, n, rng):
    chunks = np.array_split(rng.permutation(n), 5)
    all_rows = np.arange(n)
    return np.mean([mae(x, y, np.setdiff1d(all_rows, te), te) for te in chunks])

def forward_cv(x, y, origins, width, window=None):
    return np.mean([mae(x, y, np.arange(0 if window is None else max(0, c-window), c),
                          np.arange(c, c+width)) for c in origins])

def mean_se(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "mcse": float(a.std(ddof=1) / np.sqrt(len(a)))}

def hac_mean_se(z, bandwidth):
    u = z - z.mean()
    n = len(u)
    lrv = np.dot(u, u) / n
    for k in range(1, bandwidth + 1):
        lrv += 2 * (1-k/(bandwidth+1)) * np.dot(u[k:], u[:-k]) / n
    return float(np.sqrt(max(lrv, 0) / n))

def run():
    stationary = {k: [] for k in ['random_cv', 'expanding_cv', 'future']}
    drift = {k: [] for k in ['random_cv', 'recent_expanding_cv', 'recent_rolling_cv',
                             'future_expanding', 'future_rolling']}
    overlap = {k: [] for k in ['mae', 'iid_se', 'hac24', 'hac48', 'hac96']}
    for seed in range(REPEATS):
        rng = np.random.default_rng(10000 + seed)
        innovations = rng.normal(size=1901)
        series = np.empty(1901)
        series[0] = rng.normal(scale=1 / np.sqrt(1-0.8**2))
        for t in range(1, 1901):
            series[t] = 0.8 * series[t-1] + innovations[t]
        series = series[300:]
        x, y = series[:-1], series[1:]
        stationary['random_cv'].append(random_cv(x, y, 1200, rng))
        stationary['expanding_cv'].append(forward_cv(x, y, [400,600,800,1000], 200))
        stationary['future'].append(mae(x, y, np.arange(1200), np.arange(1200,1600)))
        rng = np.random.default_rng(20000 + seed)
        x = rng.normal(size=1600)
        slope = np.where(np.arange(1600) < 800, 1.0, -1.0)
        y = slope * x + rng.normal(scale=0.3, size=1600)
        drift['random_cv'].append(random_cv(x, y, 1200, rng))
        drift['recent_expanding_cv'].append(forward_cv(x, y, [900,1000,1100],100))
        drift['recent_rolling_cv'].append(forward_cv(x, y, [900,1000,1100],100,200))
        drift['future_expanding'].append(mae(x,y,np.arange(1200),np.arange(1200,1600)))
        drift['future_rolling'].append(mae(x,y,np.arange(1000,1200),np.arange(1200,1600)))
        rng = np.random.default_rng(30000 + seed)
        eps = rng.normal(size=2023)
        target = np.convolve(eps, np.ones(24), mode='valid')
        loss = np.abs(target)
        assert len(loss) == 2000
        overlap['mae'].append(float(loss.mean()))
        overlap['iid_se'].append(float(loss.std(ddof=1)/np.sqrt(2000)))
        for lag in (24,48,96):
            overlap[f'hac{lag}'].append(hac_mean_se(loss,lag))
    result = {
        'repeats': REPEATS, 'numpy_version': np.__version__,
        'stationary': {k:mean_se(v) for k,v in stationary.items()},
        'drift': {k:mean_se(v) for k,v in drift.items()},
        'overlap': {k:mean_se(v) for k,v in overlap.items()},
        'overlap_empirical_sd': float(np.std(overlap['mae'],ddof=1)),
        'oracle_stationary_mae': float(np.sqrt(2/np.pi)),
        'oracle_overlap_mae': float(np.sqrt(24)*np.sqrt(2/np.pi)),
    }
    Path(__file__).with_name('results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    run()
