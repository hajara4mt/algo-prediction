from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd

from algo_prediction.backend_gestion.adls_silver import AdlsSilverBackend
from algo_prediction.backend_gestion.silver_results_writer import (
    persist_models,
    persist_predictions_monthly,
)
from algo_prediction.domain import RequestParams

from algo_prediction.preprocessing.invoices import build_monthly_invoices
from algo_prediction.preprocessing.usage_data import build_monthly_usage_factors
from algo_prediction.preprocessing.dju import get_degreedays_mentuel
from algo_prediction.preprocessing.model_table import (
    build_model_table_for_pdl_fluid,
    split_train_test_like_r,
)

from algo_prediction.modeling.training import train_like_r


def _to_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")
    
import re

def _normalize_month_year_list(months: list[str]) -> list[str]:
    out = []
    for m in months:
        if m is None:
            continue
        s = str(m).strip()
        # YYYYMM -> YYYY-MM
        if re.fullmatch(r"\d{6}", s):
            s = s[:4] + "-" + s[4:6]
        # YYYY-MM déjà OK
        elif re.fullmatch(r"\d{4}-\d{2}", s):
            pass
        else:
            # dernier recours: essayer de parser puis reformater
            try:
                s = pd.to_datetime(s).strftime("%Y-%m")
            except Exception:
                continue
        out.append(s)
    return sorted(set(out))


# ... (imports identiques)

def run_building_and_persist(
    building_id: str,
    start_ref: date,
    end_ref: date,
    start_pred: date,
    end_pred: date,
) -> Dict[str, Any]:
    backend = AdlsSilverBackend()
    params = RequestParams(
        building_id=building_id,
        start_date_ref=start_ref,
        end_date_ref=end_ref,
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )

    site = backend.get_site_info(params)
    if not site.weather_station:
        raise ValueError("Pas de weather_station pour ce building -> impossible de récupérer les DJU.")

    inv_raw = backend.get_invoices(params)
    if inv_raw.empty:
        return {
            "id_building_primaire": building_id,
            "run_id": None,
            "created_at": None,
            "results": [],
            "models": [],
            "warning": "Aucune facture (invoices) trouvée.",
        }

    inv_monthly = build_monthly_invoices(inv_raw)

    inv_monthly["month_year_start_dbg"] = pd.to_datetime(inv_monthly["start"]).dt.strftime("%Y-%m")
    inv_monthly["month_year_end_dbg"] = pd.to_datetime(inv_monthly["end"]).dt.strftime("%Y-%m")

    usage_raw = backend.get_usage_data(
        params,
        start=min(start_ref, start_pred),
        end=max(end_ref, end_pred),
    )
    usage_messages: List[str] = []
    usage_pivot = build_monthly_usage_factors(
        df_usage=usage_raw,
        building_id=building_id,
        start=min(start_ref, start_pred),
        end=max(end_ref, end_pred),
        messages=usage_messages,
    )

    if (
        not usage_pivot.empty
        and usage_pivot["month_year"].astype(str).str.match(r"^\d{6}$").any()
    ):
        s = usage_pivot["month_year"].astype(str)
        usage_pivot["month_year"] = s.str.slice(0, 4) + "-" + s.str.slice(4, 6)

    pred_dfs: List[pd.DataFrame] = []
    model_rows: List[Dict[str, Any]] = []

    pairs = (
        inv_monthly[["deliverypoint_id_primaire", "fluid"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["deliverypoint_id_primaire", "fluid"])
    )

    for pdl_id, fluid in pairs.itertuples(index=False):
        inv_pf = inv_monthly[
            (inv_monthly["deliverypoint_id_primaire"] == pdl_id)
            & (inv_monthly["fluid"] == fluid)
        ].copy()
        if inv_pf.empty:
            continue

        messages: List[str] = []

        # Pred months
        pred_months = pd.date_range(
            pd.Timestamp(start_pred).replace(day=1),
            pd.Timestamp(end_pred).replace(day=1),
            freq="MS",
        ).strftime("%Y-%m").tolist()

        # ✅ Cohérent : si inv_pf a month_year, on l'utilise
        if "month_year" in inv_pf.columns:
            inv_months = inv_pf["month_year"].astype(str).dropna().unique().tolist()
        else:
            inv_months = pd.to_datetime(inv_pf["start"]).dt.strftime("%Y-%m").dropna().unique().tolist()

        month_year_invoice = _normalize_month_year_list(inv_months + pred_months)
        messages.append(
            f"debug_month_year_invoice: inv_months={len(inv_months)} pred_months={len(pred_months)} total={len(month_year_invoice)}"
        )

        dju_pivot = get_degreedays_mentuel(
            station_id=site.weather_station,
            month_year_invoice=month_year_invoice,
            messages=messages,
        )

        model_table = build_model_table_for_pdl_fluid(
            df_invoices_monthly=inv_monthly,
            df_dju_monthly=dju_pivot,
            df_usage_monthly=usage_pivot,
            pdl_id=pdl_id,
            fluid=fluid,
            month_year_invoice=month_year_invoice,
            messages=messages,
        )

        train, test = split_train_test_like_r(
            model_table=model_table,
            start_ref=start_ref,
            end_ref=end_ref,
            start_pred=start_pred,
            end_pred=end_pred,
            messages=messages,
        )

        standard_cols = {"month_year", "start", "end", "deliverypoint_id_primaire", "fluid", "value"}
        dju_cols = [c for c in model_table.columns if c.startswith("hdd") or c.startswith("cdd")]
        influencing_cols: List[str] = [c for c in model_table.columns if c.startswith("fi_")]

        usage_cols = [
            c for c in model_table.columns
            if (c not in standard_cols) and (c not in dju_cols) and (c not in influencing_cols)
        ]

        out = train_like_r(
            train=train,
            test=test,
            fluid=fluid,
            pdl_id=pdl_id,
            messages=messages,
            influencing_cols=influencing_cols,
            usage_cols=usage_cols,
        )

        status = out.get("status")
        model_coeffs = out.get("model_coefficients") or {}
        acc = out.get("accuracy_reference_model")
        pred = out.get("predictive_consumption")

        # ---------- Models row (create first) ----------
        row: Dict[str, Any] = {
            "deliverypoint_id_primaire": pdl_id,
            "fluid": fluid,
            "model_family": model_coeffs.get("model"),
            "chosen_hdd": model_coeffs.get("chosen_hdd"),
            "chosen_cdd": model_coeffs.get("chosen_cdd"),
            "status": str(status),
            "b_coefficient": model_coeffs.get("b_coefficient"),
            "a_hdd": model_coeffs.get("a_coefficient.hdd"),
            "a_cdd": model_coeffs.get("a_coefficient.cdd"),
            "annual_consumption_reference": None,
            "ME": None,
            "RMSE": None,
            "MAE": None,
            "MPE": None,
            "MAPE": None,
            "R2": None,
            "adjR2": model_coeffs.get("adjR2_final_model"),
            "debug_messages": "\n".join(messages),
        }

        # ✅ Outliers AFTER row exists
        outliers_df = out.get("outliers_reference")
        if outliers_df is not None and not outliers_df.empty:
            row["outliers_json"] = outliers_df.to_dict(orient="records")
        else:
            row["outliers_json"] = []

        # Metrics
        try:
            if acc is not None and not acc.empty:
                a0 = acc.iloc[0].to_dict()
                row["annual_consumption_reference"] = a0.get("annual_consumption_reference")
                row["ME"] = a0.get("ME")
                row["RMSE"] = a0.get("RMSE")
                row["MAE"] = a0.get("MAE")
                row["MPE"] = a0.get("MPE")
                row["MAPE"] = a0.get("MAPE")
                row["R2"] = a0.get("R2")
        except Exception:
            pass

        model_rows.append(row)

        # Predictions
        if pred is not None and not pred.empty:
            dfp = pred.copy()
            if "month_str" not in dfp.columns and "month" in dfp.columns:
                dfp = dfp.rename(columns={"month": "month_str"})

            dfp["deliverypoint_id_primaire"] = pdl_id
            dfp["fluid"] = fluid
            if "confidence_lower95" not in dfp.columns:
                dfp["confidence_lower95"] = pd.NA
            if "confidence_upper95" not in dfp.columns:
                dfp["confidence_upper95"] = pd.NA

            pred_dfs.append(dfp[[
                "deliverypoint_id_primaire",
                "fluid",
                "month_str",
                "real_consumption",
                "predictive_consumption",
                "confidence_lower95",
                "confidence_upper95",
            ]])

    # Assemble
    df_predictions_all = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else pd.DataFrame(
        columns=[
            "deliverypoint_id_primaire", "fluid", "month_str",
            "real_consumption", "predictive_consumption",
            "confidence_lower95", "confidence_upper95"
        ]
    )
    df_models_all = pd.DataFrame(model_rows)

    run_id, created_at = persist_predictions_monthly(
        df_predictions=df_predictions_all,
        id_building_primaire=building_id,
    )
    persist_models(
        df_models=df_models_all,
        id_building_primaire=building_id,
        run_id=run_id,
        created_at=created_at,
    )

    return {
        "id_building_primaire": building_id,
        "run_id": run_id,
        "created_at": str(created_at),
        "results": df_predictions_all.to_dict(orient="records"),
        "models": df_models_all.to_dict(orient="records"),
    }
