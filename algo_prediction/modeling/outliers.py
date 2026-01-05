from __future__ import annotations

import numpy as np
import pandas as pd


def detect_outliers_iqr_on_residuals(
    values: pd.Series,
    thres: float = 3.0,
) -> pd.Series:
    """
    Détection d'anomalies type R (ts_anomaly_detection simplifié).

    Principe :
    - On calcule une tendance lissée (médiane glissante)
    - On calcule les résidus
    - On applique une règle IQR sur les résidus

    Paramètres
    ----------
    values : pd.Series
        Série de consommation (imputation ou correction)
    thres : float
        Multiplicateur IQR (R ~ 3)

    Retour
    ------
    pd.Series bool
        True = anomalie
    """

    x = pd.to_numeric(values, errors="coerce")

    # Pas assez de points exploitables → aucune anomalie
    if x.notna().sum() < 4:
        return pd.Series(False, index=values.index)

    # 1) Tendance robuste : médiane glissante
    trend = (
        x.rolling(window=5, center=True, min_periods=2)
        .median()
        .interpolate(limit_direction="both")
    )

    # 2) Résidus
    resid = x - trend

    # 3) IQR sur résidus
    q1 = np.nanpercentile(resid, 25)
    q3 = np.nanpercentile(resid, 75)
    iqr = q3 - q1

    # Série plate ou bruit quasi nul → pas d’anomalies
    if not np.isfinite(iqr) or iqr < 1e-12:
        return pd.Series(False, index=values.index)

    low = q1 - thres * iqr
    high = q3 + thres * iqr

    outliers = (resid < low) | (resid > high)
    outliers = outliers.fillna(False)

    return outliers
