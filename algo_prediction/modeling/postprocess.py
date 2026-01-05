# modeling/postprocess.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from algo_prediction.modeling.imputation import ranking_method_like_r
from algo_prediction.modeling.outliers import detect_outliers_iqr_on_residuals  # ✅ fallback existant
from algo_prediction.modeling.dju_model import r2_and_adj_r2


def build_y_like_r(
    train: pd.DataFrame,
    best_hdd: Optional[str],
    best_cdd: Optional[str],
    y_raw_col: str = "value",
    month_col: str = "month_year",
    messages: Optional[List[str]] = None,
) -> pd.DataFrame:
    if messages is None:
        messages = []

    df = train.copy()

    # -------------------------
    # Helpers
    # -------------------------
    def _factors(_df: pd.DataFrame) -> List[str]:
        return [c for c in [best_hdd, best_cdd] if c is not None and c in _df.columns]

    def _min_n_for_adj_r2(p_expl: int) -> int:
        # adjR² défini si n - p - 1 > 0  => n >= p+2
        return p_expl + 2

    def _score_adj_r2(_df: pd.DataFrame, ycol: str) -> float:
        factors = _factors(_df)
        p = len(factors)
        if p == 0:
            return float("-inf")

        yy = pd.to_numeric(_df[ycol], errors="coerce")
        X = _df[factors].apply(pd.to_numeric, errors="coerce")

        mask = yy.notna() & X.notna().all(axis=1)
        n = int(mask.sum())
        if n < _min_n_for_adj_r2(p):
            return float("-inf")

        X_np = X.loc[mask].to_numpy(float)
        y_np = yy.loc[mask].to_numpy(float)
        X_np = np.column_stack([np.ones((X_np.shape[0], 1)), X_np])

        beta, *_ = np.linalg.lstsq(X_np, y_np, rcond=None)
        yhat = X_np @ beta
        _, adj = r2_and_adj_r2(y_np, yhat, p_expl=p)
        return float(adj) if np.isfinite(adj) else float("-inf")

    def _predict_dju_fitted(_df: pd.DataFrame, ycol: str, fit_mask: pd.Series) -> pd.Series:
        """
        Fit OLS: ycol ~ DJU sur fit_mask.
        Retourne fitted partout où DJU dispo, NaN sinon.
        """
        factors = _factors(_df)
        p = len(factors)
        if p == 0:
            return pd.Series(np.nan, index=_df.index)

        y = pd.to_numeric(_df[ycol], errors="coerce")
        X = _df[factors].apply(pd.to_numeric, errors="coerce")

        m = fit_mask.copy()
        m &= y.notna()
        m &= X.notna().all(axis=1)

        n = int(m.sum())
        if n < _min_n_for_adj_r2(p):
            return pd.Series(np.nan, index=_df.index)

        X_fit = X.loc[m].to_numpy(float)
        y_fit = y.loc[m].to_numpy(float)
        X_fit = np.column_stack([np.ones((X_fit.shape[0], 1)), X_fit])

        beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)

        ok_pred = X.notna().all(axis=1)
        fitted = pd.Series(np.nan, index=_df.index)
        if bool(ok_pred.any()):
            X_all = X.loc[ok_pred].to_numpy(float)
            X_all = np.column_stack([np.ones((X_all.shape[0], 1)), X_all])
            fitted.loc[ok_pred] = X_all @ beta

        return fitted

    # -------------------------
    # 1) missing -> ranking imputation
    # -------------------------
    y_raw = pd.to_numeric(df[y_raw_col], errors="coerce")
    df["is_missing"] = y_raw.isna()

    gap_ratio = float(df["is_missing"].mean()) if len(df) else 0.0
    if gap_ratio >= 0.2:
        messages.append("note_003: number of MISSING data > 20%, the result is not guaranteed")

    # force float pour éviter FutureWarning dtype
    df["consumption_imputation"] = y_raw.astype(float)

    if int(df["is_missing"].sum()) > 0:
        messages.append(f"note_004: number of MISSING data occured in your data: {int(df['is_missing'].sum())}")
        rank_fill = ranking_method_like_r(y_raw, period=12)["weighted_combination"].to_numpy(dtype=float)
        miss_mask = df["is_missing"].to_numpy()
        df.loc[df["is_missing"], "consumption_imputation"] = rank_fill[miss_mask]

        # ✅ refit DJU sur imputation et overwrite UNIQUEMENT missing
        fitted_imp = _predict_dju_fitted(df, ycol="consumption_imputation", fit_mask=~df["is_missing"])
        can_replace = df["is_missing"] & fitted_imp.notna()
        df.loc[can_replace, "consumption_imputation"] = fitted_imp.loc[can_replace]

    # -------------------------
    # 2) compare adjR2(raw) vs adjR2(imputation) -> si raw > imp => drop NA raw
    # -------------------------
    s_raw = _score_adj_r2(df, y_raw_col)
    s_imp = _score_adj_r2(df, "consumption_imputation")
    if s_raw > s_imp:
        df = df[df[y_raw_col].notna()].copy()
        messages.append("note: raw DJU adjR2 > imputed DJU adjR2 -> drop raw NA rows")

    # -------------------------
    # 3) anomalies on imputation -> correction
    # -------------------------
    df = df.sort_values(month_col).copy()
    df["consumption_imputation"] = pd.to_numeric(df["consumption_imputation"], errors="coerce").astype(float)

    factors = _factors(df)

    # ✅ IMPORTANT: pas d'import en haut si la fonction n'existe pas
    out_mask = pd.Series(False, index=df.index)
    if len(factors) >= 1:
        try:
            # si tu l'as dans modeling/outliers.py, on l'utilise
            from modeling.outliers import detect_outliers_iqr_on_dju_residuals  # type: ignore

            out_mask = detect_outliers_iqr_on_dju_residuals(
                df=df,
                y_col="consumption_imputation",
                x_cols=factors,
                thres=3.0,
            )
        except Exception:
            # sinon fallback robuste (fonction existante)
            out_mask = detect_outliers_iqr_on_residuals(df["consumption_imputation"], thres=3.0)
    else:
        out_mask = detect_outliers_iqr_on_residuals(df["consumption_imputation"], thres=3.0)

    out_mask = pd.Series(out_mask, index=df.index).fillna(False).astype(bool)
    df["is_anomaly"] = out_mask

    # init float pour éviter warning dtype
    df["consumption_correction"] = df["consumption_imputation"].astype(float)

    if int(out_mask.sum()) > 0:
        messages.append(f"note_005: number of ANOMALIES data occured in your data: {int(out_mask.sum())}")

        base = df["consumption_imputation"].copy()
        base.loc[out_mask] = np.nan

        corr_rank = ranking_method_like_r(base, period=12)["weighted_combination"].to_numpy(dtype=float)
        out_np = out_mask.to_numpy()

        # remplace uniquement anomalies par ranking
        df.loc[out_mask, "consumption_correction"] = corr_rank[out_np]

        # ✅ refit DJU sur correction et overwrite UNIQUEMENT anomalies
        fitted_cor = _predict_dju_fitted(df, ycol="consumption_correction", fit_mask=~df["is_anomaly"])
        can_replace2 = df["is_anomaly"] & fitted_cor.notna()
        df.loc[can_replace2, "consumption_correction"] = fitted_cor.loc[can_replace2]

    else:
        # ✅ règle R : si pas d'anomalies -> correction = raw (pas imputation)
        df["consumption_correction"] = pd.to_numeric(df[y_raw_col], errors="coerce").astype(float)

    # -------------------------
    # 4) zero rule : comparer adjR2 imputation sans vs avec zéros
    # -------------------------
    s_with0 = _score_adj_r2(df, "consumption_imputation")

    df_wo0 = df[pd.to_numeric(df["consumption_imputation"], errors="coerce") != 0].copy()
    s_wo0 = _score_adj_r2(df_wo0, "consumption_imputation")

    if np.isfinite(s_wo0) and (s_wo0 >= s_with0):
        messages.append("note_006: reference data WITHOUT ZEROS is selected")
        df = df_wo0
    else:
        messages.append("note_007: reference data WITH CORRECTED ZEROS is selected")

    # -------------------------
    # 5) choix final Y
    # -------------------------
    s_imp2 = _score_adj_r2(df, "consumption_imputation")
    s_cor2 = _score_adj_r2(df, "consumption_correction")

    best_y = "consumption_imputation" if s_imp2 >= s_cor2 else "consumption_correction"
    messages.append(f"note_008: {best_y} was selected as the best outcome Y")

    df["y_final"] = pd.to_numeric(df[best_y], errors="coerce").astype(float)
    return df
