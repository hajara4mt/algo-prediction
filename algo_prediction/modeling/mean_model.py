# modeling/mean_model.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def run_mean_model_like_r(
    train: pd.DataFrame,
    test: pd.DataFrame,
    co2_coef: Optional[float] = None,   # <-- optionnel
    value_col: str = "value",
    month_col: str = "month_year",
) -> dict:
  

    # --- mean / sd sur train
    y = pd.to_numeric(train.get(value_col), errors="coerce")
    m = float(np.nanmean(y)) if y.notna().any() else float("nan")
    sd = float(np.nanstd(y, ddof=1)) if y.notna().sum() >= 2 else float("nan")

    annual_ref = 12.0 * m if np.isfinite(m) else float("nan")
   ## annual_co2 = (co2_coef * annual_ref) if (co2_coef is not None and np.isfinite(annual_ref)) else float("nan")

    # --- table de prédiction (sur les mois de test)
    if test is None or test.empty:
        out = pd.DataFrame(columns=["month", "real_consumption", "predictive_consumption",
                                    "confidence_lower95", "confidence_upper95",
                                    "real_ghg_emissions", "predictive_ghg_emissions"])
    else:
        if month_col not in test.columns:
            raise KeyError(f"test must contain column '{month_col}'")

        out = test[[month_col]].copy()
        out["real_consumption"] = pd.to_numeric(test.get(value_col), errors="coerce")

        out["predictive_consumption"] = m
        out["confidence_lower95"] = m - 1.96 * sd if np.isfinite(sd) else np.nan
        out["confidence_upper95"] = m + 1.96 * sd if np.isfinite(sd) else np.nan

        # CO2 (optionnel)
        #if co2_coef is None:
       #     out["real_ghg_emissions"] = np.nan
        #    out["predictive_ghg_emissions"] = np.nan
        #else:
          #  out["real_ghg_emissions"] = co2_coef * out["real_consumption"]
         #   out["predictive_ghg_emissions"] = co2_coef * out["predictive_consumption"]

        out = out.rename(columns={month_col: "month"})

    # --- accuracy table 
    accuracy = pd.DataFrame([{
        "annual_consumption_reference": annual_ref,
       # "annual_ghg_emissions_reference": annual_co2,
        "ME": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "MPE": np.nan,
        "MAPE": np.nan,
        "R2": np.nan,
    }])

    return {
        "model_coefficients": {"model": "mean", "monthly_mean": m, "monthly_sd": sd},
        "accuracy_reference_model": accuracy,
        "predictive_consumption": out,
    }
