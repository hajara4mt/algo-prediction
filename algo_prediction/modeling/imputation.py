# algo_prediction/preprocessing/imputation.py
from __future__ import annotations

import numpy as np
import pandas as pd


def interpolation_missing_linear(x: pd.Series) -> pd.Series:
    """
    Equivalent R: interpolation_missing(option="linear") -> approx(rule=2)

    - Si pas de NA -> retourne tel quel
    - Si < 3 points non-NA -> en R: stop. Ici: retourne inchangé (on laisse la couche au-dessus gérer).
    - Interpolation linéaire + extrapolation aux bords (limit_direction="both")
    """
    s = pd.to_numeric(x, errors="coerce")

    if not s.isna().any():
        return s

    if s.notna().sum() < 3:
        return s

    return s.interpolate(method="linear", limit_direction="both")


def kalman_smooth_structts_like(x: pd.Series) -> pd.Series:
    """
    Approximation Python du R: na_Kalman_Smooth(x, model="StructTS")

    Dans R: vrai modèle état-espace + KalmanSmooth.
    Ici: fallback robuste et simple:
    - on remplit d'abord avec linear pour éviter trous trop grands
    - puis on applique un lissage (EWMA) qui joue le rôle de 'smooth'
    - et on ne modifie que les positions initialement NA (comme l'imputation)
    """
    s = pd.to_numeric(x, errors="coerce")
    if not s.isna().any():
        return s
    if s.notna().sum() < 3:
        return s

    base = interpolation_missing_linear(s)
    smoothed = base.ewm(span=min(12, max(3, len(base) // 3)), adjust=False).mean()

    out = s.copy()
    out[s.isna()] = smoothed[s.isna()]
    return out


def seasonal_stl_loess_like(x: pd.Series, period: int = 12) -> pd.Series:
    """
    Approximation Python du R: forecast::na.interp(x) (STL/Loess sur série saisonnière)

    R ne l'active que si:
    - period > 1
    - et length(x) > 2 * period

    Ici:
    - on estime une saisonnalité moyenne par mois (position dans l'année)
    - on complète uniquement les NA
    """
    s = pd.to_numeric(x, errors="coerce")
    n = len(s)

    if not s.isna().any():
        return s
    if s.notna().sum() < 3:
        return s
    if period <= 1 or n <= 2 * period:
        # pas assez long pour saisonnalité fiable
        return interpolation_missing_linear(s)

    base = interpolation_missing_linear(s)

    # position dans l'année (0..period-1)
    idx = np.arange(n) % period
    seasonal = pd.Series(base.values).groupby(idx).transform("mean")
    seasonal = pd.Series(seasonal.values, index=base.index)

    out = s.copy()
    out[s.isna()] = seasonal[s.isna()]
    return out


def ranking_method_like_r(x: pd.Series, period: int = 12) -> pd.DataFrame:
    """
    Reproduction du R: ranking_method()

    Sort un DataFrame avec plusieurs imputations + weighted_combination (moyenne ligne à ligne).
    Dans R, ils prennent:
      - linear_interpolation
      - Kalman_StructTS
      - (optionnel) season_stl_loess si assez long
      - weighted_combination = rowMeans

    Ici on renvoie pareil.
    """
    s = pd.to_numeric(x, errors="coerce")

    linear = interpolation_missing_linear(s)
    kalman = kalman_smooth_structts_like(s)

    df = pd.DataFrame(
        {
            "linear_interpolation": linear,
            "kalman_structts": kalman,
        }
    )

    if period > 1 and len(s) > 2 * period:
        season = seasonal_stl_loess_like(s, period=period)
        df["season_stl_loess"] = season

    df["weighted_combination"] = df.mean(axis=1)
    return df
