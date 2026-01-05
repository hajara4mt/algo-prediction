# modeling/metrics.py

import numpy as np
import pandas as pd

def regression_metrics(y_true, y_pred) -> dict:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yhat = pd.to_numeric(pd.Series(y_pred), errors="coerce")

    mask = y.notna() & yhat.notna()
    if mask.sum() == 0:
        return {"ME": np.nan, "RMSE": np.nan, "MAE": np.nan, "MPE": np.nan, "MAPE": np.nan, "R2": np.nan}

    y = y[mask].to_numpy(float)
    yhat = yhat[mask].to_numpy(float)

    err = yhat - y
    me = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    denom = np.where(np.abs(y) < 1e-12, np.nan, y)
    mpe = float(np.nanmean((err / denom) * 100.0))
    mape = float(np.nanmean((np.abs(err) / np.abs(denom)) * 100.0))

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    return {"ME": me, "RMSE": rmse, "MAE": mae, "MPE": mpe, "MAPE": mape, "R2": r2}
