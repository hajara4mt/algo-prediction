# modeling/outliers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL


@dataclass(frozen=True)
class AnomalyResult:
    outlier_mask: pd.Series
    cleaned: pd.Series
    debug: Dict[str, Any]


def _quantile_type7(x: np.ndarray, p: float) -> float:
    """
    Calcul du quantile type=7 (défaut R).
    Formule: Q(p) = (1-gamma)*x[j] + gamma*x[j+1]
    où j = floor((n-1)*p), gamma = (n-1)*p - floor((n-1)*p)
    """
    vals = x[~np.isnan(x)]
    if len(vals) == 0:
        return np.nan
    vals = np.sort(vals)
    n = len(vals)

    if n == 1:
        return float(vals[0])

    index = (n - 1) * p
    j = int(np.floor(index))
    gamma = index - j

    if j >= n - 1:
        return float(vals[-1])

    return float((1 - gamma) * vals[j] + gamma * vals[j + 1])


def _iqr_bounds_type7(resid: np.ndarray, thres: float) -> Optional[Tuple[float, float, float, float]]:
    """Return (low, high, q1, q3) using R's type=7 quantile method."""
    q1 = _quantile_type7(resid, 0.25)
    q3 = _quantile_type7(resid, 0.75)

    if not np.isfinite(q1) or not np.isfinite(q3):
        return None

    iqr = q3 - q1

    if iqr <= 0:
        return None

    low = q1 - thres * iqr
    high = q3 + thres * iqr

    if (high - low) <= 1e-14:
        return None

    return low, high, q1, q3


def _na_interp_ts_like(x: pd.Series, period: int = 12) -> pd.Series:
    """
    TS-like NA interpolation (approx R's na.interp):
    - If enough data: STL seasonal/trend decomposition, reconstruct missing
    - Else: linear + edge fill
    """
    s = pd.to_numeric(x, errors="coerce").astype(float)

    if s.notna().all():
        return s

    # base fallback: linear interpolation
    base = s.interpolate(method="linear", limit_direction="both").ffill().bfill()

    n = len(base)
    if period <= 1 or n <= 2 * period:
        return base

    try:
        stl = STL(base.to_numpy(), period=period, robust=True).fit()
        recon = stl.trend + stl.seasonal
        out = s.copy()
        out[s.isna()] = recon[s.isna().to_numpy()]
        out = out.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return out
    except Exception:
        return base


def _seasonal_strength_and_seasadj(xx: np.ndarray, period: int) -> Tuple[np.ndarray, float]:
    """
    Calculate seasonal strength and return seasonally adjusted series if strength >= 0.6.
    Matches R's: strength <- 1 - var(rem)/var(detrend)
    """
    if period <= 1:
        return xx, 0.0

    n = len(xx)
    if n <= 2 * period:
        return xx, 0.0

    try:
        stl = STL(xx, period=period, robust=True).fit()
        seasonal = stl.seasonal
        trend = stl.trend
        rem = stl.resid
    except Exception:
        return xx, 0.0

    detrend = xx - trend
    var_rem = np.nanvar(rem)
    var_detrend = np.nanvar(detrend)

    strength = 0.0
    if np.isfinite(var_rem) and np.isfinite(var_detrend) and var_detrend > 0:
        strength = 1.0 - (var_rem / var_detrend)

    if np.isfinite(strength) and strength >= 0.6:
        return (xx - seasonal), float(strength)

    return xx, float(strength)


def _supsmu_smooth(tt: np.ndarray, xx: np.ndarray, n: int, period: int) -> np.ndarray:
    """
    Smoothing function approximating R's supsmu.
    
    R's supsmu documentation recommends for n < 40:
    "Reasonable span values are 0.2 to 0.4"
    
    V24-FINAL: Use span=0.25 for n<40 (within R's recommended 0.2-0.4 range)
    This produces fewer false positives than larger spans.
    
    Key: lowess with it=0 (no robust iterations) is closer to supsmu behavior.
    """
    from scipy.stats import theilslopes

    # For very short series (n <= period), use robust linear trend
    if n <= period:
        try:
            slope, intercept, _, _ = theilslopes(xx, tt)
            return intercept + slope * tt
        except Exception:
            pass

    # V24-FINAL: Span values aligned with R's supsmu recommendations
    # R docs: "For n < 40, reasonable span values are 0.2 to 0.4"
    if n < 40:
        frac = 0.25  # Within R's recommended range
    elif n < 100:
        frac = 0.20
    else:
        frac = 0.15

    try:
        # it=0: No robust iterations (closer to supsmu which has no re-weighting)
        fitted = lowess(endog=xx, exog=tt, frac=frac, it=0, return_sorted=False)
        return np.asarray(fitted, dtype=float)
    except Exception:
        try:
            slope, intercept, _, _ = theilslopes(xx, tt)
            return intercept + slope * tt
        except Exception:
            return np.full_like(xx, np.nanmean(xx))


def _single_pass_outlier_detection(
    x: pd.Series,
    period: int,
    thres: float,
    original_missing: pd.Series,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Single pass of outlier detection (equivalent to one iteration of R's tsoutliers).
    """
    n = len(x)
    debug: Dict[str, Any] = {}

    # 1) NA interpolation
    xx_series = _na_interp_ts_like(x, period=period)
    xx = xx_series.to_numpy()

    # Check for constant series
    if np.nanstd(xx) == 0 or np.allclose(np.nanstd(xx), 0.0):
        out = pd.Series(False, index=x.index)
        debug["reason"] = "constant"
        return out, debug

    # 2) Seasonal strength and adjustment
    xx2, strength = _seasonal_strength_and_seasadj(xx, period=period)
    debug["strength"] = float(strength)

    # 3) Smooth
    tt = np.arange(1, n + 1, dtype=float)
    smooth = _supsmu_smooth(tt, xx2, n, period)
    debug["smooth"] = smooth.tolist()

    # 4) Residuals
    resid = (xx2 - smooth).astype(float)
    # Mark original missing values as NaN in residuals
    resid[original_missing.to_numpy()] = np.nan
    debug["resid"] = resid.tolist()

    # 5) IQR rule with R's type=7 quantiles
    bounds = _iqr_bounds_type7(resid, thres=thres)

    if bounds is None:
        out = pd.Series(False, index=x.index)
        debug["reason"] = "no_iqr_bounds"
        return out, debug

    low, high, q1, q3 = bounds
    debug["low"] = float(low)
    debug["high"] = float(high)
    debug["q1"] = float(q1)
    debug["q3"] = float(q3)
    debug["iqr"] = float(q3 - q1)

    # Detect outliers
    out_arr = (resid < low) | (resid > high)
    out_arr = np.where(np.isnan(out_arr), False, out_arr)
    out_mask = pd.Series(out_arr.astype(bool), index=x.index)

    # Never mark original missing as outliers
    out_mask = out_mask & (~original_missing)

    return out_mask, debug


def ts_anomaly_detection_like_r(
    values: pd.Series,
    period: int = 12,
    thres: float = 3.0,
    iterate: int = 2,
) -> AnomalyResult:
    """
    Python implementation of R's ts_anomaly_detection function.

    Pipeline matches R:
    1. na.interp: Interpolate missing values
    2. seasadj: Seasonal adjustment if strength >= 0.6
    3. supsmu: Smoothing (approximated with lowess, span=0.25 for n<40)
    4. IQR rule: Q1 - thres*IQR, Q3 + thres*IQR with R's type=7 quantiles
    5. iterate: Re-run algorithm on data with previous outliers set to NA
    """
    x = pd.to_numeric(values, errors="coerce").astype(float)
    original_missing = x.isna()

    # Track all outliers across iterations
    all_outliers = pd.Series(False, index=x.index)

    all_debug: Dict[str, Any] = {
        "thres": float(thres),
        "period": int(period),
        "iterate": int(iterate),
        "passes": [],
    }

    x_current = x.copy()

    for pass_num in range(1, iterate + 1):
        pass_outliers, pass_debug = _single_pass_outlier_detection(
            x_current, period, thres, original_missing
        )

        pass_info = {
            "pass": pass_num,
            "outliers_detected": int(pass_outliers.sum()),
            "outlier_positions": np.where(pass_outliers.to_numpy())[0].tolist(),
            **pass_debug,
        }
        all_debug["passes"].append(pass_info)

        new_outliers = pass_outliers & (~all_outliers)
        if new_outliers.sum() == 0:
            all_debug["stopped_at_pass"] = pass_num
            break

        all_outliers = all_outliers | pass_outliers

        if pass_num < iterate:
            x_current = x.copy()
            x_current.loc[all_outliers] = np.nan

    all_debug["total_outliers"] = int(all_outliers.sum())
    all_debug["outlier_positions"] = np.where(all_outliers.to_numpy())[0].tolist()

    if all_debug["passes"]:
        last_pass = all_debug["passes"][-1]
        for key in ["low", "high", "q1", "q3", "strength"]:
            if key in last_pass:
                all_debug[key] = last_pass[key]

    x_clean = x.copy()
    x_clean.loc[all_outliers] = np.nan

    return AnomalyResult(
        outlier_mask=all_outliers,
        cleaned=x_clean,
        debug=all_debug,
    )


def detect_outliers_iqr_on_residuals(values: pd.Series, thres: float = 3.0) -> pd.Series:
    """
    Convenience function for outlier detection with R-like defaults.
    """
    res = ts_anomaly_detection_like_r(values, period=12, thres=thres, iterate=2)
    return res.outlier_mask
