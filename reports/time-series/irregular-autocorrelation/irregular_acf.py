"""Autocorrelation diagnostics for irregularly sampled panel time series.

The implementation is intentionally explicit: it constructs only pairs whose
clock-time separation is below ``max_lag + truncate * bandwidth``.  Therefore
its work is O(n + P), where P is the number of eligible within-series pairs,
not O(n^2) in the total number of observations.

The primary estimator is a pair-weighted kernel estimate of
    E[z_i z_j | t_j - t_i approximately tau],
where z is standardized within each series (or within a user-selected
standardization group).

This is research/diagnostic code, not a drop-in high-frequency production
engine.  For very dense data, use Numba/C++ or aggregate pair products into
fine lag bins before smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_TIME_SCALE_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}


@dataclass(frozen=True)
class PairTableSummary:
    """Metadata describing a constructed pair table."""

    n_observations: int
    n_series: int
    n_pairs: int
    max_lag: float
    time_unit: str


def _as_list(cols: str | Sequence[str] | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _validate_time_unit(time_unit: str) -> float:
    try:
        return _TIME_SCALE_SECONDS[time_unit]
    except KeyError as exc:
        allowed = ", ".join(_TIME_SCALE_SECONDS)
        raise ValueError(f"time_unit must be one of: {allowed}") from exc


def standardize_within(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_cols: str | Sequence[str],
    output_col: str = "z",
    min_std: float = 1e-12,
) -> pd.DataFrame:
    """Demean and scale a value within user-selected groups.

    Population standard deviation (ddof=0) is used because the standardized
    values are inputs to a covariance diagnostic rather than a stand-alone
    unbiased variance estimate.  Groups with essentially zero variance receive
    NaN and are dropped later by pair construction.
    """

    groups = _as_list(group_cols)
    if not groups:
        raise ValueError("group_cols must contain at least one column")

    out = frame.copy()
    grouped = out.groupby(groups, observed=True, sort=False)[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    std = std.where(std > min_std)
    out[output_col] = (out[value_col] - mean) / std
    return out


def build_within_series_pairs(
    frame: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    series_cols: str | Sequence[str],
    max_lag: float,
    time_unit: str = "minutes",
    cluster_cols: str | Sequence[str] | None = None,
    max_pairs: int | None = 20_000_000,
) -> tuple[pd.DataFrame, PairTableSummary]:
    """Construct eligible ordered pairs within each series.

    Parameters
    ----------
    frame:
        One row per observation. ``value_col`` should normally already be a
        centered/standardized residual.
    time_col:
        Datetime-like timestamp.
    value_col:
        Numeric series value, commonly a standardized residual ``z``.
    series_cols:
        Pairs are never formed across distinct series (for example, CUSIPs).
    max_lag:
        Largest positive clock-time separation retained, in ``time_unit``.
    cluster_cols:
        Optional bootstrap cluster identifiers copied from the left/anchor
        observation.  If pairs must not cross sessions, include the session in
        ``series_cols`` as well; merely placing it in ``cluster_cols`` does not
        prevent cross-session pairs.
    max_pairs:
        Safety guard against accidental pair explosion. Set to None to disable.

    Returns
    -------
    pairs:
        Columns ``dt``, ``product``, ``left_sq``, ``right_sq`` plus series and
        cluster identifiers.
    summary:
        Observation and pair counts.
    """

    series = _as_list(series_cols)
    clusters = _as_list(cluster_cols)
    if not series:
        raise ValueError("series_cols must contain at least one column")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")

    seconds_per_unit = _validate_time_unit(time_unit)
    required = list(dict.fromkeys(series + clusters + [time_col, value_col]))
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = frame.loc[:, required].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=list(dict.fromkeys(series + clusters + [time_col, value_col])))
    work = work.sort_values(series + [time_col], kind="mergesort")

    chunks: list[pd.DataFrame] = []
    pair_total = 0

    group_key: str | list[str] = series[0] if len(series) == 1 else series
    for key, group in work.groupby(group_key, observed=True, sort=False):
        n = len(group)
        if n < 2:
            continue

        timestamps = group[time_col].astype("int64").to_numpy() / 1e9
        values = group[value_col].to_numpy(dtype=float)

        # Convert key to a tuple so metadata assignment is uniform.
        key_tuple = key if isinstance(key, tuple) else (key,)
        metadata = {col: val for col, val in zip(series, key_tuple)}

        cluster_arrays = {
            col: group[col].to_numpy(copy=False) for col in clusters
        }

        for i in range(n - 1):
            cutoff = timestamps[i] + max_lag * seconds_per_unit
            stop = int(np.searchsorted(timestamps, cutoff, side="right"))
            if stop <= i + 1:
                continue

            right = np.arange(i + 1, stop)
            dt = (timestamps[right] - timestamps[i]) / seconds_per_unit
            left_value = values[i]
            right_values = values[right]
            m = len(right)

            data: dict[str, object] = {
                "dt": dt,
                "product": left_value * right_values,
                "left_sq": np.full(m, left_value * left_value),
                "right_sq": right_values * right_values,
            }
            data.update({col: np.full(m, val) for col, val in metadata.items()})
            for col, arr in cluster_arrays.items():
                data[col] = np.full(m, arr[i])

            chunks.append(pd.DataFrame(data))
            pair_total += m
            if max_pairs is not None and pair_total > max_pairs:
                raise MemoryError(
                    f"Eligible pair count exceeded max_pairs={max_pairs:,}. "
                    "Reduce max_lag, split by session, aggregate to fine lag "
                    "bins, or use a compiled implementation."
                )

    if chunks:
        pairs = pd.concat(chunks, ignore_index=True)
    else:
        pairs = pd.DataFrame(
            columns=["dt", "product", "left_sq", "right_sq", *series, *clusters]
        )

    summary = PairTableSummary(
        n_observations=len(work),
        n_series=work.groupby(group_key, observed=True).ngroups,
        n_pairs=len(pairs),
        max_lag=float(max_lag),
        time_unit=time_unit,
    )
    return pairs, summary


def _kernel_weights(
    distance: np.ndarray,
    *,
    bandwidth: float,
    kernel: str,
    truncate: float,
) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    u = distance / bandwidth
    if kernel == "gaussian":
        weights = np.exp(-0.5 * u * u)
        weights[np.abs(u) > truncate] = 0.0
        return weights
    if kernel in {"rectangular", "slot"}:
        return (np.abs(u) <= 1.0).astype(float)
    if kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
    raise ValueError("kernel must be 'gaussian', 'rectangular', or 'epanechnikov'")


def kernel_acf_from_pairs(
    pairs: pd.DataFrame,
    *,
    lags: Iterable[float],
    bandwidth: float,
    kernel: str = "gaussian",
    truncate: float = 4.0,
) -> pd.DataFrame:
    """Estimate a clock-time ACF from a within-series pair table.

    The estimator uses a weighted Pearson-style normalization at each lag:

        sum(w z_i z_j) / sqrt(sum(w z_i^2) sum(w z_j^2)).

    This keeps each reported value in [-1, 1].  When the input values have been
    standardized within series and stationarity is plausible, it is close to
    the simpler weighted mean of products.  The result also reports that raw
    product mean so the normalization choice remains visible.
    """

    required = {"dt", "product", "left_sq", "right_sq"}
    missing = required.difference(pairs.columns)
    if missing:
        raise KeyError(f"Pair table is missing columns: {sorted(missing)}")

    dt = pairs["dt"].to_numpy(dtype=float)
    product = pairs["product"].to_numpy(dtype=float)
    left_sq = pairs["left_sq"].to_numpy(dtype=float)
    right_sq = pairs["right_sq"].to_numpy(dtype=float)

    rows: list[dict[str, float | int]] = []
    for lag in np.asarray(list(lags), dtype=float):
        if lag < 0:
            raise ValueError("lags must be nonnegative")
        if lag == 0:
            rows.append(
                {
                    "lag": 0.0,
                    "acf": 1.0,
                    "product_mean": 1.0,
                    "weight_sum": np.nan,
                    "weight_kish": np.nan,
                    "pair_count": 0,
                }
            )
            continue

        w = _kernel_weights(
            dt - lag, bandwidth=bandwidth, kernel=kernel, truncate=truncate
        )
        positive = w > 0
        if not np.any(positive):
            rows.append(
                {
                    "lag": float(lag),
                    "acf": np.nan,
                    "product_mean": np.nan,
                    "weight_sum": 0.0,
                    "weight_kish": 0.0,
                    "pair_count": 0,
                }
            )
            continue

        w = w[positive]
        p = product[positive]
        ls = left_sq[positive]
        rs = right_sq[positive]
        numerator = float(np.dot(w, p))
        denom = float(np.sqrt(np.dot(w, ls) * np.dot(w, rs)))
        weight_sum = float(w.sum())
        weight_sq_sum = float(np.dot(w, w))

        rows.append(
            {
                "lag": float(lag),
                "acf": numerator / denom if denom > 0 else np.nan,
                "product_mean": numerator / weight_sum,
                "weight_sum": weight_sum,
                # Weight concentration only; not an inferential sample size.
                "weight_kish": weight_sum * weight_sum / weight_sq_sum,
                "pair_count": int(positive.sum()),
            }
        )

    return pd.DataFrame(rows)


def event_time_acf(
    frame: pd.DataFrame,
    *,
    value_col: str,
    series_cols: str | Sequence[str],
    max_event_lag: int,
) -> pd.DataFrame:
    """Compute a pair-weighted event-index ACF within each series."""

    series = _as_list(series_cols)
    if max_event_lag < 0:
        raise ValueError("max_event_lag must be nonnegative")

    work = frame.dropna(subset=[value_col]).copy()
    group_key: str | list[str] = series[0] if len(series) == 1 else series
    groups = list(work.groupby(group_key, observed=True, sort=False))

    rows = [{"event_lag": 0, "acf": 1.0, "pair_count": len(work)}]
    for k in range(1, max_event_lag + 1):
        numerator = 0.0
        left_ss = 0.0
        right_ss = 0.0
        count = 0
        for _, group in groups:
            x = group[value_col].to_numpy(dtype=float)
            if len(x) <= k:
                continue
            left = x[:-k]
            right = x[k:]
            numerator += float(np.dot(left, right))
            left_ss += float(np.dot(left, left))
            right_ss += float(np.dot(right, right))
            count += len(left)

        denom = np.sqrt(left_ss * right_ss)
        rows.append(
            {
                "event_lag": k,
                "acf": numerator / denom if denom > 0 else np.nan,
                "pair_count": count,
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_acf_from_pairs(
    pairs: pd.DataFrame,
    *,
    cluster_col: str,
    lags: Iterable[float],
    bandwidth: float,
    kernel: str = "gaussian",
    truncate: float = 4.0,
    n_boot: int = 500,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Cluster bootstrap bands using precomputed pair contributions.

    All pairs in a resampled cluster move together.  This is much safer than
    bootstrapping individual pairs, but it is valid only when the chosen
    clusters are plausibly independent.  For one long series, use a moving or
    stationary block bootstrap on the original observations instead.
    """

    if cluster_col not in pairs.columns:
        raise KeyError(f"{cluster_col!r} is not present in the pair table")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    lag_values = np.asarray(list(lags), dtype=float)
    nonzero_lags = lag_values[lag_values > 0]
    cluster_codes, cluster_labels = pd.factorize(pairs[cluster_col], sort=False)
    n_clusters = len(cluster_labels)
    if n_clusters < 2:
        raise ValueError("At least two bootstrap clusters are required")

    # For every cluster and lag, store numerator and the two normalization sums.
    numer = np.zeros((n_clusters, len(nonzero_lags)))
    left = np.zeros_like(numer)
    right = np.zeros_like(numer)

    dt = pairs["dt"].to_numpy(dtype=float)
    product = pairs["product"].to_numpy(dtype=float)
    left_sq = pairs["left_sq"].to_numpy(dtype=float)
    right_sq = pairs["right_sq"].to_numpy(dtype=float)

    for ell, lag in enumerate(nonzero_lags):
        w = _kernel_weights(
            dt - lag, bandwidth=bandwidth, kernel=kernel, truncate=truncate
        )
        numer[:, ell] = np.bincount(
            cluster_codes, weights=w * product, minlength=n_clusters
        )
        left[:, ell] = np.bincount(
            cluster_codes, weights=w * left_sq, minlength=n_clusters
        )
        right[:, ell] = np.bincount(
            cluster_codes, weights=w * right_sq, minlength=n_clusters
        )

    rng = np.random.default_rng(random_state)
    boot = np.full((n_boot, len(nonzero_lags)), np.nan)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        nsum = numer[draw].sum(axis=0)
        lsum = left[draw].sum(axis=0)
        rsum = right[draw].sum(axis=0)
        denom = np.sqrt(lsum * rsum)
        np.divide(nsum, denom, out=boot[b], where=denom > 0)

    alpha = 1.0 - confidence
    lower = np.nanquantile(boot, alpha / 2.0, axis=0)
    upper = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)

    point = kernel_acf_from_pairs(
        pairs,
        lags=lag_values,
        bandwidth=bandwidth,
        kernel=kernel,
        truncate=truncate,
    )
    point["lower"] = np.nan
    point["upper"] = np.nan
    mask = point["lag"] > 0
    point.loc[mask, "lower"] = lower
    point.loc[mask, "upper"] = upper
    point.loc[point["lag"] == 0, ["lower", "upper"]] = 1.0
    point["n_clusters"] = n_clusters
    return point


def ou_half_life_from_acf(
    acf: pd.DataFrame,
    *,
    lag_col: str = "lag",
    acf_col: str = "acf",
    min_acf: float = 0.05,
    max_acf: float = 0.98,
) -> dict[str, float]:
    """Fit log(rho(tau)) = -lambda * tau through the origin.

    This is a descriptive summary only.  Use it when the empirical ACF is
    positive and approximately monotone over the fitted lag range.
    """

    work = acf[[lag_col, acf_col]].dropna()
    work = work.loc[
        work[lag_col].gt(0)
        & work[acf_col].between(min_acf, max_acf, inclusive="both")
    ]
    if len(work) < 2:
        raise ValueError("Need at least two positive, nontrivial ACF points")

    tau = work[lag_col].to_numpy(dtype=float)
    y = np.log(work[acf_col].to_numpy(dtype=float))
    slope = float(np.dot(tau, y) / np.dot(tau, tau))
    decay_rate = -slope
    if decay_rate <= 0:
        raise ValueError("Estimated decay rate is nonpositive; OU fit is unsuitable")
    return {
        "decay_rate": decay_rate,
        "half_life": float(np.log(2.0) / decay_rate),
        "n_lags": float(len(work)),
    }
