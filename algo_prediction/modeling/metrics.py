
# modeling/metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    """
    Metrics aligned with the common R convention used by forecast::accuracy:
      error = y_true - y_pred

    Important:
    - ME/RMSE/MAE/R2 are computed on all finite pairs (y, yhat), including zeros.
    - MPE/MAPE are computed only where y != 0 (to avoid division by zero).
    - Any NaN/inf pairs are excluded from all computations.
    """
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yhat = pd.to_numeric(pd.Series(y_pred), errors="coerce")

    # Base mask: keep only finite pairs
    mask_base = np.isfinite(y.to_numpy()) & np.isfinite(yhat.to_numpy())
    if mask_base.sum() == 0:
        return {
            "ME": np.nan, "RMSE": np.nan, "MAE": np.nan,
            "MPE": np.nan, "MAPE": np.nan, "R2": np.nan
        }

    y0 = y[mask_base].to_numpy(dtype=float)
    yhat0 = yhat[mask_base].to_numpy(dtype=float)

    err0 = y0 - yhat0

    me = float(np.mean(err0))
    rmse = float(np.sqrt(np.mean(err0 ** 2)))
    mae = float(np.mean(np.abs(err0)))

    # % metrics mask: also exclude y == 0 (division by zero)
    mask_pct = mask_base & (y.to_numpy() != 0.0)
    if mask_pct.sum() == 0:
        mpe = np.nan
        mape = np.nan
    else:
        yp = y[mask_pct].to_numpy(dtype=float)
        yhatp = yhat[mask_pct].to_numpy(dtype=float)
        errp = yp - yhatp

        # No extra epsilon here: match the "pure" definition
        mpe = float(np.mean((errp / yp) * 100.0))
        mape = float(np.mean((np.abs(errp) / np.abs(yp)) * 100.0))

    ss_res = float(np.sum((y0 - yhat0) ** 2))
    ss_tot = float(np.sum((y0 - np.mean(y0)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan

    return {"ME": me, "RMSE": rmse, "MAE": mae, "MPE": mpe, "MAPE": mape, "R2": r2}