# modeling/dju_model.py

from __future__ import annotations

from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from algo_prediction.modeling.metrics import regression_metrics

DJU_CANDIDATES = ["hdd10", "hdd15", "hdd18", "cdd21", "cdd24", "cdd26"]


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    OLS via least squares.
    Returns: beta, sigma2, dof, XtX_inv
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat

    n = int(X.shape[0])
    p = int(X.shape[1])
    dof = max(n - p, 1)
    sse = float(np.sum(resid ** 2))
    sigma2 = sse / dof

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    return beta, sigma2, dof, XtX_inv


def _t_crit_975(dof: int) -> float:
    """
    R utilise une loi t pour interval="confidence".
    Si scipy est dispo -> exact.
    Sinon fallback approx.
    """
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, df=dof))
    except Exception:
        if dof <= 1:
            return 12.706
        if dof == 2:
            return 4.303
        if dof == 3:
            return 3.182
        if dof == 4:
            return 2.776
        if dof == 5:
            return 2.571
        if dof == 6:
            return 2.447
        if dof == 7:
            return 2.365
        if dof == 8:
            return 2.306
        if dof == 9:
            return 2.262
        if dof == 10:
            return 2.228
        return 1.96


def _predict_confidence_interval(
    X_new: np.ndarray,
    beta: np.ndarray,
    sigma2: float,
    XtX_inv: np.ndarray,
    dof: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    interval="confidence" comme R:
      mean prediction +/- t * SE(mean)
      SE(mean) = sqrt( diag( X_new (XtX)^-1 X_new' ) * sigma2 )
    """
    yhat = X_new @ beta
    v = np.sum((X_new @ XtX_inv) * X_new, axis=1)
    se_mean = np.sqrt(np.maximum(v * sigma2, 0.0))

    tcrit = _t_crit_975(dof)
    lwr = yhat - tcrit * se_mean
    upr = yhat + tcrit * se_mean
    return yhat, lwr, upr


def r2_and_adj_r2(y: np.ndarray, yhat: np.ndarray, p_expl: int) -> Tuple[float, float]:
    """
    p_expl = nb variables explicatives (sans intercept), comme en R pour adj.r.squared.
    """
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    n = int(len(y))
    if n < 3:
        return float("nan"), float("nan")

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    denom = (n - p_expl - 1)
    if denom <= 0 or not np.isfinite(r2):
        adj = float("nan")
    else:
        adj = float(1.0 - (1.0 - r2) * (n - 1) / denom)

    return r2, adj


def _choose_best_single_dju_by_adj_r2(
    train: pd.DataFrame,
    value_col: str,
    candidates: List[str],
) -> Dict[str, float]:
    """
    Retourne dict: {candidate: adj_r2} pour celles qui sont testables.
    """
    y = _safe_numeric(train[value_col])
    scores: Dict[str, float] = {}

    for col in candidates:
        if col not in train.columns:
            continue

        x = _safe_numeric(train[col])
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 6:
            continue

        x_np = x[mask].to_numpy(float)
        y_np = y[mask].to_numpy(float)

        X = np.column_stack([np.ones_like(x_np), x_np])
        beta, *_ = np.linalg.lstsq(X, y_np, rcond=None)
        yhat = X @ beta

        _, adj = r2_and_adj_r2(y_np, yhat, p_expl=1)
        if np.isfinite(adj):
            scores[col] = float(adj)

    return scores


def choose_best_hdd_cdd_like_r(
    train: pd.DataFrame,
    value_col: str = "value",
    dju_candidates: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], Dict[str, float], Dict[str, float]]:
    """
    Renvoie:
      best_hdd, best_cdd, hdd_scores(adjR2), cdd_scores(adjR2)
    """
    if dju_candidates is None:
        dju_candidates = DJU_CANDIDATES

    hdd_candidates = [c for c in dju_candidates if c.startswith("hdd")]
    cdd_candidates = [c for c in dju_candidates if c.startswith("cdd")]

    hdd_scores = _choose_best_single_dju_by_adj_r2(train, value_col, hdd_candidates)
    cdd_scores = _choose_best_single_dju_by_adj_r2(train, value_col, cdd_candidates)

    best_hdd = max(hdd_scores, key=hdd_scores.get) if hdd_scores else None
    best_cdd = max(cdd_scores, key=cdd_scores.get) if cdd_scores else None

    return best_hdd, best_cdd, hdd_scores, cdd_scores


def _coef_map(beta: np.ndarray, x_cols: List[str]) -> Dict[str, float]:
    """
    Transforme beta en dict:
      - intercept -> b_coefficient
      - slopes -> a_coefficient.<col>
    """
    out: Dict[str, float] = {}
    if beta is None or len(beta) == 0:
        return out

    out["b_coefficient"] = float(beta[0])
    for i, col in enumerate(x_cols, start=1):
        out[f"a_coefficient.{col}"] = float(beta[i]) if i < len(beta) else float("nan")
    return out


def run_best_dju_model_like_r(
    train: pd.DataFrame,
    test: pd.DataFrame,
    value_col: str = "value",
    month_col: str = "month_year",
    dju_candidates: Optional[List[str]] = None,
    influencing_cols: Optional[List[str]] = None,
    usage_cols: Optional[List[str]] = None,
    messages: Optional[List[str]] = None,
    chosen_hdd: Optional[str] = None,
    chosen_cdd: Optional[str] = None,
) -> Optional[dict]:
    """
    Fidèle au R pour la partie DJU :
    - Choisir best HDD et best CDD via adj.r.squared en univarié sur train
    - Fit OLS final: y ~ best_hdd + best_cdd + influencing + usage
    - Predict sur test avec interval="confidence" => fit/lwr/upr
    - Si une feature nécessaire manque sur un mois => prediction NaN (pas d'arrêt)
    """
    if messages is None:
        messages = []
    if dju_candidates is None:
        dju_candidates = DJU_CANDIDATES

    influencing_cols = influencing_cols or []
    usage_cols = usage_cols or []

    
    # ------------------------------------------------------------------
    # 1) Choix DJU : soit "forced" (R-like via training), soit auto
    # ------------------------------------------------------------------
    if chosen_hdd is not None or chosen_cdd is not None:
        # R-like: on réutilise le choix fait une seule fois en amont
        best_hdd = chosen_hdd
        best_cdd = chosen_cdd

        # sécurité: si la colonne n'existe pas, on la drop
        if best_hdd is not None and (best_hdd not in train.columns or best_hdd not in test.columns):
            messages.append(f"note: chosen_hdd={best_hdd} not found in train/test -> ignored")
            best_hdd = None
        if best_cdd is not None and (best_cdd not in train.columns or best_cdd not in test.columns):
            messages.append(f"note: chosen_cdd={best_cdd} not found in train/test -> ignored")
            best_cdd = None

        if best_hdd is None and best_cdd is None:
            messages.append("note: forced HDD/CDD are not usable -> DJU model not usable")
            return None

        if best_hdd is not None:
            messages.append(f"debug_dju_choice: best_hdd={best_hdd} (forced_from_training)")
        if best_cdd is not None:
            messages.append(f"debug_dju_choice: best_cdd={best_cdd} (forced_from_training)")

        # pas de scores dispo dans ce mode
        hdd_scores: Dict[str, float] = {}
        cdd_scores: Dict[str, float] = {}

    else:
        best_hdd, best_cdd, hdd_scores, cdd_scores = choose_best_hdd_cdd_like_r(
            train=train,
            value_col=value_col,
            dju_candidates=dju_candidates,
        )

        if best_hdd is None and best_cdd is None:
            messages.append("note: no usable HDD or CDD in train -> DJU model not usable")
            return None

        if best_hdd is not None:
            messages.append(f"debug_dju_choice: best_hdd={best_hdd} (adjR2={hdd_scores.get(best_hdd):.3f})")
        if best_cdd is not None:
            messages.append(f"debug_dju_choice: best_cdd={best_cdd} (adjR2={cdd_scores.get(best_cdd):.3f})")

    # ------------------------------------------------------------------
    # 2) Construction des features X
    # ------------------------------------------------------------------
    x_cols = [c for c in [best_hdd, best_cdd] if c is not None]
    x_cols += [c for c in influencing_cols if c in train.columns]
    x_cols += [c for c in usage_cols if c in train.columns]

    # --- Fit OLS sur lignes complètes
    y_train = _safe_numeric(train[value_col])

    # si x_cols vide => pas de modèle
    if not x_cols:
        messages.append("note: no predictors available for DJU model -> DJU model not usable")
        return None

    X_train_df = train[x_cols].copy()
    for c in x_cols:
        X_train_df[c] = _safe_numeric(X_train_df[c])

    mask_train = y_train.notna()
    for c in x_cols:
        mask_train &= X_train_df[c].notna()

    if int(mask_train.sum()) < 6:
        messages.append("note: not enough complete rows for DJU+factors model -> DJU model not usable")
        return None

    y_np = y_train[mask_train].to_numpy(float)
    X_np = X_train_df.loc[mask_train, x_cols].to_numpy(float)
    X_np = np.column_stack([np.ones((X_np.shape[0], 1)), X_np])

    beta, sigma2, dof, XtX_inv = _ols_fit(X_np, y_np)
    yhat_train = X_np @ beta

    mets = regression_metrics(y_np, yhat_train)
    r2, adj_final = r2_and_adj_r2(y_np, yhat_train, p_expl=len(x_cols))
    mets["R2"] = r2
    mets["adjR2"] = adj_final

    # --- Prediction sur test
    out = test[[month_col]].copy()
    out["real_consumption"] = _safe_numeric(test.get("value"))

    # IMPORTANT: test peut ne pas avoir toutes les colonnes (usage/influencing)
    missing_required_cols = [c for c in x_cols if c not in test.columns]
    if missing_required_cols:
        messages.append(f"note: test missing required columns {missing_required_cols} -> DJU model not usable")
        return None

    X_test_df = test[x_cols].copy()
    for c in x_cols:
        X_test_df[c] = _safe_numeric(X_test_df[c])

    mask_test = pd.Series(True, index=test.index)
    for c in x_cols:
        mask_test &= X_test_df[c].notna()

    missing_months = out.loc[~mask_test, month_col].astype(str).tolist()
    if missing_months:
        messages.append(
            f"note: some months have missing predictors -> predictive_consumption=NA for months {missing_months}"
        )

    out["predictive_consumption"] = np.nan
    out["confidence_lower95"] = np.nan
    out["confidence_upper95"] = np.nan

    if bool(mask_test.any()):
        X_new = X_test_df.loc[mask_test, x_cols].to_numpy(float)
        X_new = np.column_stack([np.ones((X_new.shape[0], 1)), X_new])
        fit, lwr, upr = _predict_confidence_interval(X_new, beta, sigma2, XtX_inv, dof)

        out.loc[mask_test, "predictive_consumption"] = fit
        out.loc[mask_test, "confidence_lower95"] = lwr
        out.loc[mask_test, "confidence_upper95"] = upr

    out = out.rename(columns={month_col: "month"})

    coef_map = _coef_map(beta, x_cols)

    model_coefficients = {
        "model": "ols_dju_plus_factors",
        "chosen_hdd": best_hdd,
        "chosen_cdd": best_cdd,
        "x_cols": x_cols,
        "beta": beta.tolist(),
        "y_col": value_col,

        "b_coefficient": coef_map.get("b_coefficient"),
        "a_coefficient.hdd": coef_map.get(f"a_coefficient.{best_hdd}") if best_hdd else None,
        "a_coefficient.cdd": coef_map.get(f"a_coefficient.{best_cdd}") if best_cdd else None,

        "a_coefficients_by_feature": {
            k.replace("a_coefficient.", ""): v
            for k, v in coef_map.items()
            if k.startswith("a_coefficient.")
        },

        "adjR2_final_model": float(adj_final) if np.isfinite(adj_final) else None,
    }

    annual_ref = 12.0 * float(np.nanmean(yhat_train)) if len(yhat_train) else float("nan")
    accuracy = pd.DataFrame([{
        "annual_consumption_reference": annual_ref,
        **mets,
    }])

    out["hdd_used"] = best_hdd
    out["cdd_used"] = best_cdd


    # DEBUG: Afficher les valeurs utilisées pour MAPE
    print("=== DEBUG MAPE ===")
    print(f"y_np (consumption_correction): {y_np.tolist()}")
    print(f"yhat_train (fitted): {yhat_train.tolist()}")
    print(f"Nombre de points: {len(y_np)}")

# Calculer PE pour chaque point
    pe = (y_np - yhat_train) / y_np * 100
    print(f"PE par point: {pe.tolist()}")
    print(f"MAPE calculé: {np.mean(np.abs(pe)):.2f}")


    return {
        "model_coefficients": model_coefficients,
        "accuracy_reference_model": accuracy,
        "predictive_consumption": out,
    }
