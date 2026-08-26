from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd

from irregular_acf import (
    build_within_series_pairs,
    event_time_acf,
    kernel_acf_from_pairs,
    ou_half_life_from_acf,
)

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def simulate_irregular_ou(seed: int = 20260825, n: int = 5000):
    rng = np.random.default_rng(seed)

    # A bursty observation clock: mostly a few minutes apart, occasionally
    # much longer, plus explicit session-like gaps.
    mixture = rng.uniform(size=n - 1)
    dt = np.where(
        mixture < 0.78,
        rng.exponential(scale=2.5, size=n - 1),
        np.where(
            mixture < 0.96,
            rng.exponential(scale=15.0, size=n - 1),
            rng.exponential(scale=75.0, size=n - 1),
        ),
    )
    dt = np.maximum(dt, 0.08)
    dt[np.arange(119, n - 1, 120)] += 180.0

    t = np.concatenate([[0.0], np.cumsum(dt)])
    half_life = 30.0
    lam = math.log(2.0) / half_life

    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        a = math.exp(-lam * dt[i - 1])
        x[i] = a * x[i - 1] + math.sqrt(max(1.0 - a * a, 0.0)) * rng.normal()

    z = (x - x.mean()) / x.std(ddof=0)
    frame = pd.DataFrame(
        {
            "series": "synthetic_ou",
            "timestamp": pd.Timestamp("2026-01-01", tz="UTC")
            + pd.to_timedelta(t, unit="m"),
            "z": z,
            "x": x,
        }
    )
    return frame, dt, half_life, lam


def regular_ffill_acf(frame: pd.DataFrame, lags: np.ndarray) -> np.ndarray:
    s = frame.set_index("timestamp")["x"].sort_index()
    grid = s.resample("1min").last().ffill()
    z = (grid.to_numpy() - grid.mean()) / grid.std(ddof=0)
    out = []
    for lag in lags.astype(int):
        if lag == 0:
            out.append(1.0)
        elif lag >= len(z):
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(z[:-lag], z[lag:])[0, 1]))
    return np.asarray(out)


def make_estimand_figure(frame: pd.DataFrame):
    sample = frame.iloc[:12].copy()
    t0 = sample["timestamp"].iloc[0]
    minutes = (sample["timestamp"] - t0).dt.total_seconds() / 60.0

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(minutes, sample["z"], marker="o", linewidth=1.2)
    for i, (minute, value) in enumerate(zip(minutes, sample["z"])):
        ax.annotate(f"i={i}", (minute, value), xytext=(4, 7), textcoords="offset points", fontsize=8)

    # Event-time lag 1: first observation to the next observation.
    ax.annotate(
        "event lag k=1",
        xy=(minutes.iloc[1], sample["z"].iloc[1]),
        xytext=(minutes.iloc[0], sample["z"].max() + 0.65),
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.15"},
        fontsize=10,
    )

    # Clock-time target lag: from an anchor to the observation closest to +30m.
    anchor = 3
    target = minutes.iloc[anchor] + 30.0
    j = int(np.argmin(np.abs(minutes.to_numpy() - target)))
    ax.axvline(target, linestyle="--", linewidth=1.0)
    ax.annotate(
        "clock lag τ=30 min\n(actual pair lag differs slightly)",
        xy=(minutes.iloc[j], sample["z"].iloc[j]),
        xytext=(target + 8, sample["z"].min() - 0.7),
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.12"},
        fontsize=10,
    )

    ax.set_xlabel("clock time since first observation (minutes)")
    ax.set_ylabel("standardized value")
    ax.set_title("Event-time lag and clock-time lag are different estimands")
    ax.margins(x=0.04, y=0.22)
    fig.tight_layout()
    fig.savefig(FIG / "estimands.svg")
    plt.close(fig)


def make_acf_figures(frame: pd.DataFrame, dt: np.ndarray, half_life: float, lam: float):
    max_lag = 200.0
    pairs, summary = build_within_series_pairs(
        frame,
        time_col="timestamp",
        value_col="z",
        series_cols=["series"],
        max_lag=max_lag + 20.0,
        time_unit="minutes",
        max_pairs=10_000_000,
    )

    lags = np.arange(0.0, 181.0, 5.0)
    gaussian = kernel_acf_from_pairs(
        pairs,
        lags=lags,
        bandwidth=5.0,
        kernel="gaussian",
        truncate=4.0,
    )
    slot = kernel_acf_from_pairs(
        pairs,
        lags=lags,
        bandwidth=5.0,
        kernel="rectangular",
    )
    ffill = regular_ffill_acf(frame, lags)
    truth = np.exp(-lam * lags)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(lags, truth, linewidth=2.0, label="true OU clock-time ACF")
    ax.plot(gaussian["lag"], gaussian["acf"], marker="o", markersize=3.2, label="Gaussian-kernel estimate")
    ax.plot(slot["lag"], slot["acf"], marker="s", markersize=3.0, label="rectangular slotting estimate")
    ax.plot(lags, ffill, linestyle="--", label="1-minute last-value carry-forward + ordinary ACF")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("clock-time lag (minutes)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Irregular OU simulation: direct pair estimators vs forward-fill resampling")
    ax.legend(frameon=False)
    ax.set_ylim(-0.22, 1.04)
    fig.tight_layout()
    fig.savefig(FIG / "ou_simulation_acf.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    mask = gaussian["lag"] > 0
    ax.plot(gaussian.loc[mask, "lag"], gaussian.loc[mask, "pair_count"], marker="o", label="pairs with nonzero kernel weight")
    ax.plot(gaussian.loc[mask, "lag"], gaussian.loc[mask, "weight_kish"], marker="s", label="Kish weight support (diagnostic only)")
    ax.set_xlabel("clock-time lag (minutes)")
    ax.set_ylabel("support")
    ax.set_title("Every ACF point has different empirical pair support")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pair_support.svg")
    plt.close(fig)

    event = event_time_acf(
        frame.sort_values("timestamp"),
        value_col="z",
        series_cols=["series"],
        max_event_lag=10,
    )
    event_theory_lag1 = float(np.mean(np.exp(-lam * dt)))
    mean_gap_plugin = float(np.exp(-lam * np.mean(dt)))
    ou_fit = ou_half_life_from_acf(gaussian.loc[gaussian["lag"].between(5, 120)])

    metrics = {
        "n_observations": summary.n_observations,
        "n_pairs": summary.n_pairs,
        "median_gap_minutes": float(np.median(dt)),
        "mean_gap_minutes": float(np.mean(dt)),
        "gap_p95_minutes": float(np.quantile(dt, 0.95)),
        "true_half_life_minutes": half_life,
        "kernel_half_life_minutes": ou_fit["half_life"],
        "empirical_event_lag1": float(event.loc[event["event_lag"].eq(1), "acf"].iloc[0]),
        "theoretical_event_lag1": event_theory_lag1,
        "mean_gap_plugin_lag1": mean_gap_plugin,
        "ffill_acf_5m": float(ffill[np.where(lags == 5)[0][0]]),
        "kernel_acf_5m": float(gaussian.loc[gaussian["lag"].eq(5), "acf"].iloc[0]),
        "true_acf_5m": float(np.exp(-lam * 5)),
    }
    pd.Series(metrics, name="value").to_csv(OUT / "simulation_metrics.csv")
    gaussian.to_csv(OUT / "simulation_gaussian_acf.csv", index=False)
    slot.to_csv(OUT / "simulation_slot_acf.csv", index=False)
    event.to_csv(OUT / "simulation_event_acf.csv", index=False)
    return metrics


def main():
    frame, dt, half_life, lam = simulate_irregular_ou()
    make_estimand_figure(frame)
    metrics = make_acf_figures(frame, dt, half_life, lam)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
