# modeling/postprocess.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from algo_prediction.modeling.imputation import ranking_method_like_r
from algo_prediction.modeling.dju_model import r2_and_adj_r2
from algo_prediction.modeling.outliers import ts_anomaly_detection_like_r


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
    # 1) missing -> ranking imputation (R lignes 1177-1183)
    # -------------------------
    y_raw = pd.to_numeric(df[y_raw_col], errors="coerce")
    df["is_missing"] = y_raw.isna()

    gap_ratio = float(df["is_missing"].mean()) if len(df) else 0.0
    if gap_ratio >= 0.2:
        messages.append("note_003: number of MISSING data > 20%, the result is not guaranteed")

    df["consumption_imputation"] = y_raw.astype(float)

    if int(df["is_missing"].sum()) > 0:
        messages.append(f"note_004: number of MISSING data occured in your data: {int(df['is_missing'].sum())}")
        rank_fill = ranking_method_like_r(y_raw, period=12)["weighted_combination"].to_numpy(dtype=float)
        miss_mask = df["is_missing"].to_numpy()
        df.loc[df["is_missing"], "consumption_imputation"] = rank_fill[miss_mask]

        fitted_imp = _predict_dju_fitted(df, ycol="consumption_imputation", fit_mask=~df["is_missing"])
        can_replace = df["is_missing"] & fitted_imp.notna()
        df.loc[can_replace, "consumption_imputation"] = fitted_imp.loc[can_replace]

    # -------------------------
    # 2) compare adjR2(raw) vs adjR2(imputation) (R lignes 1194-1200)
    # -------------------------
    s_raw = _score_adj_r2(df, y_raw_col)
    s_imp = _score_adj_r2(df, "consumption_imputation")
    if s_raw > s_imp:
        df = df[df[y_raw_col].notna()].copy()
        messages.append("note: raw DJU adjR2 > imputed DJU adjR2 -> drop raw NA rows")

    # -------------------------
    # 3) anomalies detection (R lignes 1205-1226)
    # -------------------------
    df = df.sort_values(month_col).copy()
    df["consumption_imputation"] = pd.to_numeric(df["consumption_imputation"], errors="coerce").astype(float)
    # ========== DEBUG START ==========
   # print("=== DEBUG POSTPROCESS AVANT OUTLIERS ===")
    #print(f"df shape: {df.shape}")
    #print(f"df index: {df.index.tolist()}")
    #print(f"consumption_imputation values: {df['consumption_imputation'].tolist()}")
    #print(f"consumption_imputation index: {df['consumption_imputation'].index.tolist()}")
    # ========== DEBUG END ==========


    # R ligne 1207: ts_anomaly_detection(ts_data, thres = 3, lambda = NULL)
    res_out = ts_anomaly_detection_like_r(
        df["consumption_imputation"],
        period=12,
        thres=3.0,
          iterate=2,  
          ##use_loo_cv=True,         # ← CORRIGÉ: R utilise thres=3 (ligne 1207)
        
    )
   # print("=== DEBUG POSTPROCESS APRÈS OUTLIERS ===")
  #  print(f"Version: {res_out.debug.get('version')}")
   # print(f"Pass1 positions: {res_out.debug.get('pass1_positions')}")
    #print(f"Pass2 positions: {res_out.debug.get('pass2_positions')}")
    #print(f"Total positions: {res_out.debug.get('outlier_positions')}")
    #print(f"outlier_mask index: {res_out.outlier_mask.index.tolist()}")
    #print(f"outlier_mask values: {res_out.outlier_mask.tolist()}")
    # ========== DEBUG END ==========


    out_mask = res_out.outlier_mask.fillna(False).astype(bool)
    df["is_anomaly"] = out_mask
    # ========== DEBUG START ==========
    print(f"out_mask sum: {out_mask.sum()}")
    print(f"df['is_anomaly'] sum: {df['is_anomaly'].sum()}")
    print(f"Positions is_anomaly=True: {df.index[df['is_anomaly']].tolist()}")
    # 

    # R lignes 1209-1226: correction des outliers
    if int(out_mask.sum()) > 0:
        # Log comme R
        months = df.loc[out_mask, month_col].astype(str).tolist()
        messages.append(f"note_005: number of ANOMALIES data occured in your data: {int(out_mask.sum())}")
        messages.append(f"debug_outliers_months: {months}")
        messages.append(f"debug_outliers_iqr: low={res_out.debug['low']:.2f}, high={res_out.debug['high']:.2f}, q1={res_out.debug['q1']:.2f}, q3={res_out.debug['q3']:.2f}, thres={res_out.debug['thres']}")

        # R lignes 1214-1217: ranking_method sur la série avec outliers=NA
        base = df["consumption_imputation"].copy()
        base.loc[out_mask] = np.nan

        corr_rank = ranking_method_like_r(base, period=12)["weighted_combination"].to_numpy(dtype=float)
        out_np = out_mask.to_numpy()

        df["consumption_correction"] = df["consumption_imputation"].astype(float)
        df.loc[out_mask, "consumption_correction"] = corr_rank[out_np]

        # R lignes 1219-1220: remplacer par fitted values du modèle DJU
       # fitted_cor = _predict_dju_fitted(df, ycol="consumption_correction", fit_mask=~df["is_anomaly"])
        fitted_cor = _predict_dju_fitted(df, ycol="consumption_correction", fit_mask=pd.Series(True, index=df.index))
        can_replace2 = df["is_anomaly"] & fitted_cor.notna()
        df.loc[can_replace2, "consumption_correction"] = fitted_cor.loc[can_replace2]
    else:
        # R ligne 1224: si pas d'outliers, consumption_correction = invoice.consumption (raw)
        df["consumption_correction"] = pd.to_numeric(df[y_raw_col], errors="coerce").astype(float)

    # -------------------------
    # 4) zero rule (R lignes 1235-1248)
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
    # 5) choix final Y (R lignes 1241-1244 + which.max)
    # -------------------------
    s_imp2 = _score_adj_r2(df, "consumption_imputation")
    s_cor2 = _score_adj_r2(df, "consumption_correction")

    best_y = "consumption_imputation" if s_imp2 >= s_cor2 else "consumption_correction"
    messages.append(f"note_008: {best_y} was selected as the best outcome Y")

    df["y_final"] = pd.to_numeric(df[best_y], errors="coerce").astype(float)
    return df
