# modeling/outliers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class AnomalyResult:
    outlier_mask: pd.Series
    cleaned: pd.Series
    debug: Dict[str, Any]  # <- pour logguer comme R


def _nan_percentile(x: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(x, q))


def _iqr_bounds(resid: np.ndarray, thres: float) -> Optional[Tuple[float, float, float, float]]:
    """Return (low, high, q1, q3) using R-like IQR rule."""
    q1 = _nan_percentile(resid, 25)
    q3 = _nan_percentile(resid, 75)
    iqr = q3 - q1

    if (not np.isfinite(iqr)) or (iqr <= 0):
        return None

    low = q1 - thres * iqr
    high = q3 + thres * iqr

    if (high - low) <= 1e-14:
        return None

    return low, high, q1, q3


def _na_interp_ts_like(x: pd.Series, period: int = 12) -> pd.Series:
    """
    Better-than-linear TS-like NA fill (approx na.interp):
    - If enough data: STL seasonal/trend decomposition on filled series,
      then reconstruct missing using seasonal+trend, fallback to linear.
    - Else: linear + edge fill.
    """
    s = pd.to_numeric(x, errors="coerce").astype(float)

    if s.notna().all():
        return s

    # base fallback
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


def _supsmu_like_lowess(tt: np.ndarray, xx: np.ndarray, frac: float) -> np.ndarray:
    # it=0 => pas de robust iterations (plus proche supsmu “lisse”)
    fitted = lowess(endog=xx, exog=tt, frac=frac, it=0, return_sorted=False)
    return np.asarray(fitted, dtype=float)


def _pass2_tsoutliers_approx_conservative(
    x_clean: pd.Series,
    period: int = 12,
    z_thres: float = 4.5,      # <-- très conservateur
    max_p: int = 2,
    max_q: int = 2,
    allow_d: Tuple[int, ...] = (0, 1),
) -> pd.Series:
    """
    Pass2 optionnel : approximation TRÈS conservatrice pour éviter les faux positifs.
    (Ce n'est PAS tsoutliers exact, mais ça limite les cas Python=4 vs R=3.)
    """
    y = pd.to_numeric(x_clean, errors="coerce").astype(float)
    miss = y.isna()
    yy = _na_interp_ts_like(y, period=period)

    best_aic = np.inf
    best_resid = None

    for d in allow_d:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    mod = SARIMAX(
                        yy.to_numpy(),
                        order=(p, d, q),
                        seasonal_order=(0, 0, 0, 0),
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = mod.fit(disp=False)
                    aic = float(getattr(fit, "aic", np.inf))
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_resid = np.asarray(fit.resid, dtype=float)
                except Exception:
                    continue

    if best_resid is None:
        return pd.Series(False, index=y.index)

    resid = best_resid.copy()
    resid[miss.to_numpy()] = np.nan

    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(False, index=y.index)

    z = np.abs(resid - med) / (1.4826 * mad)
    out = z > z_thres
    out = np.where(np.isnan(out), False, out)
    return pd.Series(out.astype(bool), index=y.index)


def ts_anomaly_detection_like_r(
    values: pd.Series,
    period: int = 12,
    thres: float = 3.0,
    lowess_frac: float = 0.35,
    iterate: int = 1,                 # <-- IMPORTANT : default=1 (pass2 OFF)
    enable_pass2: bool = False,       # <-- OFF by default
) -> AnomalyResult:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    miss = x.isna()

    # 1) na.interp (TS-like)
    xx_series = _na_interp_ts_like(x, period=period)
    xx = xx_series.to_numpy()

    # constant -> no outliers
    if np.nanstd(xx) == 0 or np.allclose(np.nanstd(xx), 0.0):
        out = pd.Series(False, index=x.index)
        return AnomalyResult(outlier_mask=out, cleaned=x.copy(), debug={"reason": "constant"})

    # 2) seasonal strength / seasadj
    xx2, strength = _seasonal_strength_and_seasadj(xx, period=period)

    # 3) smoother (supsmu-like)
    tt = np.arange(1, len(xx2) + 1, dtype=float)
    smooth = _supsmu_like_lowess(tt, xx2, frac=lowess_frac)

    # 4) resid + IQR rule
    resid = (xx2 - smooth).astype(float)
    resid[miss.to_numpy()] = np.nan

    b = _iqr_bounds(resid, thres=thres)
    if b is None:
        out = pd.Series(False, index=x.index)
        return AnomalyResult(outlier_mask=out, cleaned=x.copy(), debug={"reason": "no_iqr_bounds", "strength": strength})

    low, high, q1, q3 = b
    out1 = (resid < low) | (resid > high)
    out1 = np.where(np.isnan(out1), False, out1)
    out_mask = pd.Series(out1.astype(bool), index=x.index)

    x_clean = x.copy()
    x_clean.loc[out_mask] = np.nan

    # 5) pass2 (optional, conservative)
    if enable_pass2 and iterate > 1:
        out2 = _pass2_tsoutliers_approx_conservative(x_clean, period=period, z_thres=4.5)
        out_mask = out_mask | out2
        x_clean = x.copy()
        x_clean.loc[out_mask] = np.nan

    # never mark original missing
    out_mask = out_mask & (~miss)

    debug = {
        "strength": float(strength),
        "low": float(low),
        "high": float(high),
        "q1": float(q1),
        "q3": float(q3),
        "lowess_frac": float(lowess_frac),
        "thres": float(thres),
        "enable_pass2": bool(enable_pass2),
        "n_outliers": int(out_mask.sum()),
        "outlier_positions": np.where(out_mask.to_numpy())[0].tolist(),
    }

    return AnomalyResult(outlier_mask=out_mask, cleaned=x_clean, debug=debug)


def detect_outliers_iqr_on_residuals(values: pd.Series, thres: float = 3.0) -> pd.Series:
    # R-like stable default: pass2 OFF
    res = ts_anomaly_detection_like_r(values, period=12, iterate=1, thres=thres, enable_pass2=False)
    return res.outlier_mask
