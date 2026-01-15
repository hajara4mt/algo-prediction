# modeling/training.py

from __future__ import annotations

from typing import List, Optional, Dict, Any

import pandas as pd

from algo_prediction.modeling.status import TrainStatus
from algo_prediction.modeling.decision import decide_training_strategy_like_r
from algo_prediction.modeling.mean_model import run_mean_model_like_r
from algo_prediction.modeling.dju_model import run_best_dju_model_like_r, choose_best_hdd_cdd_like_r
from algo_prediction.modeling.postprocess import build_y_like_r


def train_like_r(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fluid: str,
    pdl_id: str,
    messages: Optional[List[str]] = None,
    influencing_cols: Optional[List[str]] = None,
    usage_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if messages is None:
        messages = []

    influencing_cols = influencing_cols or []
    usage_cols = usage_cols or []

    # ✅ IMPORTANT: toujours initialiser
    outliers_df = pd.DataFrame()

    decision = decide_training_strategy_like_r(
        train=train,
        test=test,
        fluid=fluid,
        pdl_id=pdl_id,
        messages=messages,
    )
    status = decision["status"]

    # note_000
    if status == TrainStatus.NO_REFERENCE_DATA:
        return {
            **decision,
            "outliers_reference": outliers_df,
            "model_coefficients": None,
            "accuracy_reference_model": None,
            "predictive_consumption": None,
        }

    # note_001
    if status == TrainStatus.TOO_FEW_OBSERVATIONS:
        out = run_mean_model_like_r(
            train=train,
            test=test,
            co2_coef=0.0,
            value_col="value",
            month_col="month_year",
        )
        return {**decision, **out, "outliers_reference": outliers_df}

    # note_annual_ref
    if status == TrainStatus.OK_ANNUAL_REFERENCE:
        # 1) Choisir best_hdd/best_cdd sur le Y brut (value) pour le postprocess
        best_hdd, best_cdd, hdd_scores, cdd_scores = choose_best_hdd_cdd_like_r(
            train=train,
            value_col="value",
        )

        # logs (optionnel mais utile)
        if best_hdd is None and best_cdd is None:
            messages.append("note: no usable HDD/CDD found for postprocess scoring (best_hdd=None, best_cdd=None)")
        else:
            if best_hdd is not None:
                msg = f"debug_postprocess_dju: best_hdd={best_hdd}"
                if best_hdd in hdd_scores:
                    msg += f" (adjR2={hdd_scores[best_hdd]:.3f})"
                messages.append(msg)
            if best_cdd is not None:
                msg = f"debug_postprocess_dju: best_cdd={best_cdd}"
                if best_cdd in cdd_scores:
                    msg += f" (adjR2={cdd_scores[best_cdd]:.3f})"
                messages.append(msg)

        # 2) Postprocess Y (missing + outliers + choix Y via adjR2 du modèle DJU simple)
        processed_train = build_y_like_r(
            train=train,
            best_hdd=best_hdd,
            best_cdd=best_cdd,
            y_raw_col="value",
            month_col="month_year",
            messages=messages,
        )

        # 2bis) Construire la table outliers (style R)
        if not processed_train.empty and "is_anomaly" in processed_train.columns:
            keep = [c for c in [
                "deliverypoint_id_primaire",
                "fluid",
                "start",
                "end",
                "value",
                "is_missing",
                "consumption_imputation",
                "is_anomaly",
                "consumption_correction",
            ] if c in processed_train.columns]

            outliers_df = processed_train.loc[
                processed_train["is_anomaly"].fillna(False).astype(bool),
                keep
            ].copy()

            outliers_df = outliers_df.rename(columns={
                "deliverypoint_id_primaire": "invoice.delivery_point",
                "fluid": "invoice.fluid",
                "start": "invoice_start_date",
                "end": "invoice_end_date",
                "value": "invoice.consumption",
                "consumption_imputation": "invoice.consumption_imputation",
                "consumption_correction": "invoice.consumption_correction",
            })
        else:
            # utile pour debug : on sait pourquoi c'est vide
            messages.append("debug_outliers: no is_anomaly column or processed_train empty -> outliers_reference empty")

        # 3) Modèle final DJU + facteurs sur y_final
        out = run_best_dju_model_like_r(
            train=processed_train,
            test=test,
            value_col="y_final",
            month_col="month_year",
            influencing_cols=influencing_cols,
            usage_cols=usage_cols,
            messages=messages,
            chosen_hdd=best_hdd,
            chosen_cdd=best_cdd,
        )

        # 4) fallback mean si DJU inutilisable
        if out is None:
            messages.append("note: DJU+factors model not usable -> fallback to mean model")
            out = run_mean_model_like_r(
                train=train,
                test=test,
                co2_coef=0.0,
                value_col="value",
                month_col="month_year",
            )

        # ✅ ICI: on renvoie outliers_reference
        return {**decision, **out, "outliers_reference": outliers_df}

    # fallback sécurité
    messages.append("note: unknown status -> no training performed")
    return {
        **decision,
        "outliers_reference": outliers_df,
        "model_coefficients": None,
        "accuracy_reference_model": None,
        "predictive_consumption": None,
    }
