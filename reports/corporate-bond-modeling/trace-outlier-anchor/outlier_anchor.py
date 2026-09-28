"""Causal outlier flags and a robust mid-spread anchor for TRACE prints.

Input: one or more CSV/Parquet files with the columns listed in REQUIRED. Spreads
are in percentage points and are converted to basis points. Effective timestamps
are expected in Eastern time if naive; aware timestamps are converted to Eastern.

The anchor for the current print is ANCHOR_FOR_THIS_PRINT_BP (prior state).
ROBUST_LEVEL_BP is the state after incorporating this print. This is a research
prototype: validate side conventions, timestamps, and spread units on your data.

Example:
    python outlier_anchor.py 'data/*.parquet' --out flags.parquet --evaluate
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd


PARAMS = dict(
    tau_bp=10.0,
    dur_ref=5.0,
    k_sigma=4.0,
    c_sigma=2.0,
    ewma_halflife=20,
    sigma_floor_bp=1.0,
    sigma_floor_price=0.05,
    winsor_sigmas=3.0,
    cpp_fresh_min=60.0,
    burst_cusips=20,
    pyc_resid_bp=0.3,
    prior_sigma_by_tg={1: 5.0, 2: 1.8, 3: 1.3, 5: 1.2, 7: 1.1, 10: 1.0, 20: 1.0, 30: 1.0},
)
REQUIRED = ["CUSIP", "EFFECTIVE_DATETIME_TS", "BM_SPREAD", "TRADE_TYPE", "QUANTITY", "YRS_TO_MATURITY"]


def modified_duration(coupon_pct, yield_pct, years):
    """Modified duration for a semiannual bullet; fallback to 0.9 * years."""
    c = np.asarray(coupon_pct, float) / 200.0
    y = np.asarray(yield_pct, float)
    i = y / 200.0
    yrs = np.asarray(years, float)
    n = np.maximum(np.round(2 * yrs), 1)
    with np.errstate(all="ignore"):
        g = (1 + i) ** n
        mac_periods = (1 + i) / i - ((1 + i) + n * (c - i)) / (c * (g - 1) + i)
        mod = mac_periods / 2.0 / (1 + i)
    bad = ~np.isfinite(mod) | (mod <= 0) | (mod > 40) | (y <= 0.05)
    return np.where(bad, 0.9 * yrs, mod)


def load(files):
    frames = [pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, low_memory=False) for f in files]
    if not frames:
        sys.exit("no matching input files")
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"missing required columns: {missing}")
    return df


def prepare(df, p=PARAMS):
    d = df.copy()
    t = pd.to_datetime(d["EFFECTIVE_DATETIME_TS"], errors="coerce")
    if t.dt.tz is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["t"] = t

    if "REPORT_DATETIME" in d.columns:
        rep = pd.to_datetime(d["REPORT_DATETIME"], errors="coerce")
        if rep.dt.tz is not None:
            rep = rep.dt.tz_convert("America/New_York").dt.tz_localize(None)
        else:
            # Historical input convention: naive report time may be UTC.
            gap_h = (rep - d["t"]).dt.total_seconds().median() / 3600.0
            if 3.5 < gap_h < 5.5:
                rep = rep - pd.Timedelta(hours=round(gap_h))
        d["t_report"] = rep.fillna(d["t"])
    else:
        d["t_report"] = d["t"]
    d = d.sort_values(["t", "t_report"], kind="stable").reset_index(drop=True)

    d["S_BP"] = pd.to_numeric(d["BM_SPREAD"], errors="coerce") * 100.0
    for c in ["MID_SPREAD_CPP_EXEC_TIME", "BID_SPREAD_CPP_EXEC_TIME", "ASK_SPREAD_CPP_EXEC_TIME"]:
        d[c + "_BP"] = pd.to_numeric(d[c], errors="coerce") * 100.0 if c in d else np.nan

    if "LASTUPDATETIME_CPP" in d:
        update = pd.to_datetime(d["LASTUPDATETIME_CPP"], errors="coerce")
        if update.dt.tz is not None:
            update = update.dt.tz_convert("America/New_York").dt.tz_localize(None)
        d["CPP_AGE_MIN"] = (d["t"] - update).dt.total_seconds() / 60.0
    else:
        d["CPP_AGE_MIN"] = np.nan

    # +1 dealer buy, -1 dealer sell, 0 interdealer.
    side = np.where(d["TRADE_TYPE"].eq("B"), 1, np.where(d["TRADE_TYPE"].eq("S"), -1, 0))
    if {"CONTRA_PARTY_TYPE", "SIDE"}.issubset(d.columns):
        dd = d["TRADE_TYPE"].eq("D") & d["CONTRA_PARTY_TYPE"].eq("C")
        side = np.where(dd & d["SIDE"].eq("B"), 1, np.where(dd & d["SIDE"].eq("S"), -1, side))
    d["SGN"] = side
    d["CPP_SIDE_BP"] = np.where(side > 0, d["BID_SPREAD_CPP_EXEC_TIME_BP"],
                                np.where(side < 0, d["ASK_SPREAD_CPP_EXEC_TIME_BP"], d["MID_SPREAD_CPP_EXEC_TIME_BP"]))
    d["DEV_SIDE_BP"] = d["S_BP"] - d["CPP_SIDE_BP"]
    d["DEV_MID_BP"] = d["S_BP"] - d["MID_SPREAD_CPP_EXEC_TIME_BP"]
    d["S_MID_BP"] = d["S_BP"] - (d["CPP_SIDE_BP"] - d["MID_SPREAD_CPP_EXEC_TIME_BP"]).fillna(0.0)

    coupon = d["COUPON"] if "COUPON" in d else pd.Series(5.0, index=d.index)
    yld = d["YIELD"] if "YIELD" in d else pd.Series(5.0, index=d.index)
    d["MOD_DUR"] = modified_duration(coupon, yld, d["YRS_TO_MATURITY"])
    d["IS_WALL"] = (d["t"].dt.hour == 16) & (d["t"].dt.minute <= 1)
    d["IS_BURST"] = d.groupby(d["t"].dt.floor("s"))["CUSIP"].transform("nunique") >= p["burst_cusips"]
    d["IS_SHORT"] = d["MOD_DUR"] < 1.5
    est = pd.to_numeric(d["ESTIMATED_QUANTITY"], errors="coerce").fillna(0) if "ESTIMATED_QUANTITY" in d else 0
    d["IS_CAPPED"] = (pd.to_numeric(d["QUANTITY"], errors="coerce") >= 5_000_000) | (np.asarray(est) > 0)
    d["PYC_RESID_FLAG"] = (pd.to_numeric(d["diff_bbg_pce_yield"], errors="coerce").abs() * 100 > p["pyc_resid_bp"]
                           if "diff_bbg_pce_yield" in d else False)
    d["TENOR_GROUP_INT"] = (pd.to_numeric(d["TENOR_GROUP"], errors="coerce").fillna(10).astype(int)
                            if "TENOR_GROUP" in d else 10)
    return d


def cancel_times(d):
    """Earliest reported cancellation per (CUSIP, SEQUENCE_NUMBER)."""
    if not {"CANCELLED", "SEQUENCE_NUMBER"}.issubset(d.columns):
        return {}
    canc = d[d["CANCELLED"].astype(str).eq("Y")]
    return canc.groupby(["CUSIP", "SEQUENCE_NUMBER"])["t_report"].min().to_dict()


def run_filter(d, p=PARAMS):
    d = d.copy()
    n = len(d)
    out = {k: np.full(n, np.nan) for k in [
        "SIGMA_BP", "THRESH_BP", "Z", "JUMP_BP", "ROBUST_LEVEL_BP",
        "ANCHOR_FOR_THIS_PRINT_BP", "CPP_MOVE_SINCE_PREV_BP", "ABSORBED_FRAC",
    ]}
    is_out = np.zeros(n, bool)
    reason = np.full(n, "", dtype=object)
    prev_out = np.zeros(n, bool)
    n_since = np.full(n, -1)
    prev_canc = np.zeros(n, bool)
    canc = cancel_times(d)
    is_cancel_row = d["CANCELLED"].astype(str).eq("Y").values if "CANCELLED" in d else np.zeros(n, bool)

    alpha = 1 - 0.5 ** (1.0 / p["ewma_halflife"])
    smid = d["S_MID_BP"].to_numpy()
    devs = d["DEV_SIDE_BP"].to_numpy()
    cpp_mid = d["MID_SPREAD_CPP_EXEC_TIME_BP"].to_numpy()
    age = d["CPP_AGE_MIN"].to_numpy()
    dur = d["MOD_DUR"].to_numpy()
    tg = d["TENOR_GROUP_INT"].to_numpy()
    times = d["t"].to_numpy()
    seq = d["SEQUENCE_NUMBER"].to_numpy() if "SEQUENCE_NUMBER" in d else np.full(n, None)

    for cusip, idx in d.groupby("CUSIP", sort=False).indices.items():
        level = np.nan
        level_before = np.nan
        cpp_at_level = np.nan
        cpp_before = np.nan
        sigma = p["prior_sigma_by_tg"].get(int(tg[idx[0]]), 1.2)
        last_dev = np.nan
        last_cpp = np.nan
        last_flag = False
        last_seq = None
        since = -1

        for i in idx:
            if is_cancel_row[i]:
                continue
            cpp = cpp_mid[i]
            fresh = np.isfinite(cpp) and (not np.isfinite(age[i]) or 0 <= age[i] <= p["cpp_fresh_min"])

            # Apply a previous cancellation only once it was reported.
            if last_seq is not None and (cusip, last_seq) in canc and canc[cusip, last_seq] <= times[i]:
                level = level_before
                cpp_at_level = cpp_before
                prev_canc[i] = True
                last_flag = False
                last_seq = None

            level_pred = (level + cpp - cpp_at_level
                          if np.isfinite(level) and np.isfinite(cpp) and np.isfinite(cpp_at_level) else level)
            out["ANCHOR_FOR_THIS_PRINT_BP"][i] = level_pred
            prev_out[i] = last_flag
            n_since[i] = since
            if np.isfinite(cpp) and np.isfinite(last_cpp):
                move = cpp - last_cpp
                out["CPP_MOVE_SINCE_PREV_BP"][i] = move
                if last_flag and np.isfinite(last_dev) and last_dev != 0:
                    out["ABSORBED_FRAC"][i] = move / last_dev

            duration = dur[i] if np.isfinite(dur[i]) and dur[i] > 0.1 else 5.0
            floor = max(p["sigma_floor_bp"], p["sigma_floor_price"] * 100.0 / duration)
            sig = max(sigma, floor)
            tau = p["tau_bp"] * max(1.0, p["dur_ref"] / duration)
            threshold = max(tau, p["k_sigma"] * sig)
            dev = devs[i]
            jump = smid[i] - level_pred if np.isfinite(level_pred) else np.nan
            flag = False
            why = ""
            if fresh and np.isfinite(dev):
                if abs(dev) > threshold:
                    flag, why = True, "far_from_composite"
            elif np.isfinite(jump) and abs(jump) > threshold:
                flag, why = True, "jump_vs_level_no_fresh_composite"
            if (not flag and np.isfinite(jump) and np.isfinite(dev)
                    and abs(jump) > threshold and abs(dev) > threshold / 2):
                flag, why = True, "jump_and_half_off_composite"

            out["SIGMA_BP"][i] = sig
            out["THRESH_BP"][i] = threshold
            out["Z"][i] = dev / sig if np.isfinite(dev) else np.nan
            out["JUMP_BP"][i] = jump
            is_out[i] = flag
            reason[i] = why

            level_before = level
            cpp_before = cpp_at_level
            if not np.isfinite(level_pred):
                level = smid[i]
            elif flag:
                level = level_pred + float(np.clip(jump, -p["c_sigma"] * sig, p["c_sigma"] * sig))
            else:
                level = smid[i]
            cpp_at_level = cpp if np.isfinite(cpp) else cpp_at_level
            out["ROBUST_LEVEL_BP"][i] = level
            if np.isfinite(dev):
                sigma = (1 - alpha) * sigma + alpha * min(abs(dev), p["winsor_sigmas"] * sig)
            last_dev = dev
            last_cpp = cpp if np.isfinite(cpp) else last_cpp
            last_flag = flag
            last_seq = seq[i]
            since = 0 if flag else (since + 1 if since >= 0 else -1)

    for k, v in out.items():
        d[k] = v
    d["IS_OUTLIER"] = is_out
    d["OUTLIER_REASON"] = reason
    d["PREV_IS_OUTLIER"] = prev_out
    d["N_SINCE_FLAG"] = n_since
    d["PREV_CANCELLED_KNOWN"] = prev_canc
    return d


def evaluate(d):
    """Compare each causal anchor against the next print of the same CUSIP."""
    if "CANCELLED" in d:
        d = d[~d["CANCELLED"].astype(str).eq("Y")].copy()
    g = d.groupby("CUSIP", sort=False)
    nxt = g["S_BP"].shift(-1)
    nxt_sgn = g["SGN"].shift(-1)
    half = ((d["ASK_SPREAD_CPP_EXEC_TIME_BP"] - d["BID_SPREAD_CPP_EXEC_TIME_BP"]) / 2).abs().fillna(0.7)
    side_adj = np.where(nxt_sgn > 0, half, np.where(nxt_sgn < 0, -half, 0.0))
    preds = {
        "previous print, raw": d["S_BP"],
        "robust level after this print, + side": d["ROBUST_LEVEL_BP"] + side_adj,
        "composite quote at this print (stale by gap)": d["CPP_SIDE_BP"].where(
            nxt_sgn == d["SGN"], d["MID_SPREAD_CPP_EXEC_TIME_BP"] + side_adj),
    }
    ok = nxt.notna()
    print(f"\n=== next-print prediction error (bp), {ok.sum()} pairs ===")
    for name, pr in preds.items():
        e = (nxt - pr)[ok].abs()
        print(f"  {name:52s} MAE {e.mean():6.3f}  median {e.median():5.3f}  p95 {e.quantile(.95):6.2f}")
    fl = ok & d["IS_OUTLIER"]
    print(f"\n=== after a flagged print ({fl.sum()} pairs) ===")
    for name, pr in preds.items():
        e = (nxt - pr)[fl].abs()
        print(f"  {name:52s} MAE {e.mean():6.3f}  median {e.median():5.3f}")
    next_move = nxt - d["S_BP"]
    rev = ((np.sign(next_move) == -np.sign(d["DEV_SIDE_BP"]))
           & (next_move.abs() > 0.5 * d["DEV_SIDE_BP"].abs()))
    print("  next move reverses >half the flagged deviation: %.0f%%" % (100 * rev[fl].mean()))
    print("\n=== flags by reason ===\n" + d.loc[d["IS_OUTLIER"], "OUTLIER_REASON"].value_counts().to_string())
    print("\n=== flag rate by class ===")
    for c in ["IS_SHORT", "IS_WALL", "IS_BURST", "IS_CAPPED", "PYC_RESID_FLAG"]:
        m = d[c].astype(bool)
        print(f"  {c:16s} share {m.mean():6.2%}  flag rate inside {d.loc[m, 'IS_OUTLIER'].mean():6.2%}"
              f"  outside {d.loc[~m, 'IS_OUTLIER'].mean():6.2%}")
    ab = d.loc[d["PREV_IS_OUTLIER"], "ABSORBED_FRAC"].dropna()
    if len(ab):
        print(f"\n=== composite absorption after flagged print: median {ab.median():.2f}, n={len(ab)} ===")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="Parquet or CSV print files; shell globs accepted")
    ap.add_argument("--out", help="write the flagged frame to this Parquet path")
    ap.add_argument("--evaluate", action="store_true", help="print next-print diagnostics")
    ap.add_argument("--min-quantity", type=float, default=100_000)
    a = ap.parse_args()
    files = [f for pat in a.files for f in sorted(glob.glob(pat))]
    df = load(files)
    df = df[pd.to_numeric(df["QUANTITY"], errors="coerce") >= a.min_quantity]
    d = run_filter(prepare(df))
    print(f"prints {len(d):,}  flagged {d['IS_OUTLIER'].sum():,} ({100 * d['IS_OUTLIER'].mean():.2f}%)")
    if a.evaluate:
        evaluate(d)
    if a.out:
        keep = list(df.columns) + [
            "DEV_SIDE_BP", "DEV_MID_BP", "SIGMA_BP", "THRESH_BP", "Z", "JUMP_BP",
            "IS_OUTLIER", "OUTLIER_REASON", "IS_WALL", "IS_BURST", "IS_SHORT", "IS_CAPPED",
            "PYC_RESID_FLAG", "MOD_DUR", "ROBUST_LEVEL_BP", "ANCHOR_FOR_THIS_PRINT_BP",
            "PREV_IS_OUTLIER", "CPP_MOVE_SINCE_PREV_BP", "ABSORBED_FRAC", "N_SINCE_FLAG",
            "PREV_CANCELLED_KNOWN",
        ]
        d[list(dict.fromkeys(c for c in keep if c in d))].to_parquet(a.out, index=False)
        print("written", a.out)


if __name__ == "__main__":
    main()
