# modeling/metrics.py - V2 (Match R's forecast::accuracy with swapped args)

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    """
    Metrics aligned with R's forecast::accuracy() when called as:
        accuracy(actual, fitted)  # arguments swapped!
    
    This means:
      error = fitted - actual = y_pred - y_true
      PE = 100 * error / fitted = 100 * (y_pred - y_true) / y_pred
    
    Division by y_pred avoids division-by-zero when actuals contain zeros.
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

    # R's effective error (due to swapped arguments): fitted - actual
    err0 = yhat0 - y0

    me = float(np.mean(err0))
    rmse = float(np.sqrt(np.mean(err0 ** 2)))
    mae = float(np.mean(np.abs(err0)))

    # PE/MAPE: divide by yhat (fitted), not y (actual)
    # This matches R and avoids division by zero when actuals are zero
    mask_pct = mask_base & (yhat.to_numpy() != 0.0)
    if mask_pct.sum() == 0:
        mpe = np.nan
        mape = np.nan
    else:
        yp = y[mask_pct].to_numpy(dtype=float)
        yhatp = yhat[mask_pct].to_numpy(dtype=float)
        
        pe = (yhatp - yp) / yhatp * 100.0
        mpe = float(np.mean(pe))
        mape = float(np.mean(np.abs(pe)))

    # R² unchanged
    ss_res = float(np.sum((y0 - yhat0) ** 2))
    ss_tot = float(np.sum((y0 - np.mean(y0)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan

    return {"ME": me, "RMSE": rmse, "MAE": mae, "MPE": mpe, "MAPE": mape, "R2": r2}
