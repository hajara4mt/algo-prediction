# algo_prediction/tests/test_merge_split.py

from __future__ import annotations
import math

from datetime import date
from typing import Dict, List, Any

import pandas as pd

from backend_gestion.adls_silver import AdlsSilverBackend
from backend_gestion.silver_results_writer import persist_predictions_monthly, persist_models
from algo_prediction.domain import RequestParams

from preprocessing.invoices import build_monthly_invoices
from preprocessing.usage_data import build_monthly_usage_factors
from preprocessing.months import build_month_year_invoice
from preprocessing.dju import get_degreedays_mentuel
from preprocessing.model_table import build_model_table_for_pdl_fluid, split_train_test_like_r

from modeling.training import train_like_r
from modeling.status import TrainStatus


# -----------------------------
# Helpers printing
# -----------------------------
def _print_df_head(df: pd.DataFrame, name: str, n: int = 8) -> None:
    print(f"\n--- {name} (head {n}) ---")
    if df is None or df.empty:
        print("(empty)")
        return
    print(df.head(n))

def _print_outliers_if_note005(out: Dict[str, Any], messages: List[str], max_rows: int = 200) -> None:
    """
    N'ajoute rien aux sorties existantes sauf:
    - Si note_005 est présente dans messages => on affiche le détail des outliers.
    """
    has_note005 = any(m.startswith("note_005") for m in messages)
    if not has_note005:
        return

    outliers_df = out.get("outliers_reference")

    print("\n--- OUTLIERS DETAILS (triggered by note_005) ---")
    if outliers_df is None or outliers_df.empty:
        print("note_005 present but outliers_reference is empty/None -> check training.py return payload")
        return

    # colonnes attendues proches R (si dispo)
    cols = [
        "invoice.delivery_point",
        "invoice.fluid",
        "invoice_start_date",
        "invoice_end_date",
        "invoice.consumption",
        "is_missing",
        "invoice.consumption_imputation",
        "is_anomaly",
        "invoice.consumption_correction",
    ]
    cols = [c for c in cols if c in outliers_df.columns]

    # Affichage
    if cols:
        print(outliers_df[cols].head(max_rows).to_string(index=False))
    else:
        print(outliers_df.head(max_rows).to_string(index=False))

def _to_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def compute_building_metrics_from_pairs(
    annual_refs: List[float],
    surface: float,
    occupant: float,
) -> Dict[str, float]:
    """
    - annual_refs: liste des annual_consumption_reference (kWh/an) de chaque PDL×fluid modélisé
    - surface, occupant: récupérés depuis get_site_info

    Règle demandée:
      - si surface==0 ou occupant==0 => ratio = NaN
      - si annual_refs vide => total = NaN
    """
    if len(annual_refs) == 0:
        total_ref = float("nan")
    else:
        total_ref = float(sum(annual_refs))

    ratio_m2 = None
    ratio_occ = None

    if not math.isnan(total_ref):
        if surface != 0:
            ratio_m2 = total_ref / surface
        if occupant != 0:
            ratio_occ = total_ref / occupant

    return {
        "total_energy_annual_consumption_reference": total_ref,
        "ratio_kwh_m2": ratio_m2,
        "ratio_kwh_occupant": ratio_occ,
    }


def _print_metrics(acc: pd.DataFrame) -> None:
    print("\n--- accuracy_reference_model ---")
    if acc is None or acc.empty:
        print("None")
        return
    row = acc.iloc[0].to_dict()
    for k, v in row.items():
        print(f"{k}: {v}")


def _print_coeffs(model_coeffs: Dict) -> None:
    print("\n--- model_coefficients ---")
    if not model_coeffs:
        print("None")
        return
    keys_first = [
        "model", "chosen_hdd", "chosen_cdd", "adjR2_final_model",
        "b_coefficient", "a_coefficient.hdd", "a_coefficient.cdd",
        "x_cols", "y_col",
    ]
    for k in keys_first:
        if k in model_coeffs:
            print(f"{k}: {model_coeffs.get(k)}")

    if "a_coefficients_by_feature" in model_coeffs:
        print("a_coefficients_by_feature:")
        for kk, vv in model_coeffs["a_coefficients_by_feature"].items():
            print(f"  - {kk}: {vv}")


def _print_predictions(pred: pd.DataFrame, n: int = 8) -> None:
    print(f"\n--- predictive_consumption (head {n}) ---")
    if pred is None or pred.empty:
        print("None")
        return
    cols = [c for c in ["month", "real_consumption", "predictive_consumption", "confidence_lower95", "confidence_upper95"] if c in pred.columns]
    print(pred[cols].head(n))


def _extract_missing_months(pred: pd.DataFrame) -> List[str]:
    if pred is None or pred.empty:
        return []
    if "predictive_consumption" not in pred.columns or "month" not in pred.columns:
        return []
    miss = pred.loc[pred["predictive_consumption"].isna(), "month"].astype(str).tolist()
    return miss


# -----------------------------
# Main runner
# -----------------------------
def run_test_building(
    building_id: str,
    start_ref: date,
    end_ref: date,
    start_pred: date,
    end_pred: date,
    show_details: bool = True,
    head_n: int = 8,
) -> None:
    backend = AdlsSilverBackend()

    params = RequestParams(
        building_id=building_id,
        start_date_ref=start_ref,
        end_date_ref=end_ref,
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )

    annual_refs_all_pairs: List[float] = []
    pairs_used: List[str] = []
    messages_global: List[str] = []

    # >>> Accumulateurs pour écrire dans silver
    pred_dfs: List[pd.DataFrame] = []
    model_rows: List[Dict[str, Any]] = []

    print("\n===== 1) get_site_info =====")
    site = backend.get_site_info(params)
    print(f"Building : {site.id_building_primaire} | {site.name} | station={site.weather_station} | ")

    surface = _to_float_or_nan(getattr(site, "surface", None))
    occupant = _to_float_or_nan(getattr(site, "occupant", None))
    print(f"surface={surface} | occupant={occupant}")

    if not site.weather_station:
        print("⚠️  Pas de weather_station dans building -> impossible de récupérer DJU")
        return

    print("\n===== 2) get_invoices (raw) -> monthly invoices =====")
    inv_raw = backend.get_invoices(params)
    print(f"raw invoices: {len(inv_raw)} lignes")
    if inv_raw.empty:
        print("⚠️  Aucune facture -> stop test")
        return

    inv_monthly = build_monthly_invoices(inv_raw)
    inv_monthly["month_year"] = pd.to_datetime(inv_monthly["start"]).dt.strftime("%Y-%m")
    print(f"monthly invoices: {len(inv_monthly)} lignes")

    fluids = sorted(inv_monthly["fluid"].dropna().unique().tolist())
    pdls = sorted(inv_monthly["deliverypoint_id_primaire"].dropna().unique().tolist())
    print(f"fluids (building-level): {fluids}")
    print(f"pdls  : {pdls}")

    print("\n===== 3) get_usage_data (raw) -> monthly usage pivot =====")
    usage_raw = backend.get_usage_data(
        params,
        start=min(start_ref, start_pred),
        end=max(end_ref, end_pred),
    )
    print(f"usage raw: {len(usage_raw)} lignes")

    usage_pivot = build_monthly_usage_factors(
        df_usage=usage_raw,
        building_id=building_id,
        start=min(start_ref, start_pred),
        end=max(end_ref, end_pred),
        messages=messages_global,
    )
    if not usage_pivot.empty and usage_pivot["month_year"].astype(str).str.match(r"^\d{6}$").any():
        usage_pivot["month_year"] = (
            usage_pivot["month_year"].str.slice(0, 4)
            + "-"
            + usage_pivot["month_year"].str.slice(4, 6)
        )

    print(f"usage pivot: {usage_pivot.shape}")
    if usage_pivot.empty:
        print("usage pivot vide (OK)")
    else:
        print(usage_pivot.head(5))

    computed = 0
    built = 0
    months_missing_by_pair: Dict[str, List[str]] = {}

    print("\n" + "=" * 90)
    print("START LOOP PDL × FLUID (R-like)")
    print("=" * 90)

    for pdl_id in pdls:
        fluids_for_pdl = sorted(
            inv_monthly.loc[inv_monthly["deliverypoint_id_primaire"] == pdl_id, "fluid"]
            .dropna()
            .unique()
            .tolist()
        )

        for fluid in fluids:
            computed += 1

            print("\n" + "=" * 90)
            print(f"==> RUN PDL={pdl_id} | fluid={fluid}")
            print(f"PDL fluids present in invoices: {fluids_for_pdl}")
            print("=" * 90)

            messages: List[str] = []

            inv_pf = inv_monthly[
                (inv_monthly["deliverypoint_id_primaire"] == pdl_id)
                & (inv_monthly["fluid"] == fluid)
            ].copy()

            if inv_pf.empty:
                print(f"[PDL={pdl_id}][fluid={fluid}] note: no invoices for this PDL/fluid -> skip")
                continue

            month_year_invoice = build_month_year_invoice(
                df_monthly_invoices=inv_pf,
                start_date_pred=start_pred,
                end_date_pred=end_pred,
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

            if show_details:
                print(f"\nmodel_table: {model_table.shape}")
                _print_df_head(model_table, "MODEL_TABLE", n=head_n)
                print(f"\ntrain: {train.shape} | test: {test.shape}")
                _print_df_head(train, "TRAIN", n=head_n)
                _print_df_head(test, "TEST", n=head_n)

            standard_cols = {"month_year", "start", "end", "deliverypoint_id_primaire", "fluid", "value"}
            dju_cols = [c for c in model_table.columns if c.startswith("hdd") or c.startswith("cdd")]
            influencing_cols = [c for c in model_table.columns if c.startswith("fi_")]
            usage_cols = [
                c for c in model_table.columns
                if (c not in standard_cols) and (c not in dju_cols) and (c not in influencing_cols)
            ]

            if show_details:
                print("\nFEATURES DETECTED")
                print("dju_cols:", dju_cols)
                print("influencing_cols:", influencing_cols)
                print("usage_cols:", usage_cols)

            out = train_like_r(
                train=train,
                test=test,
                fluid=fluid,
                pdl_id=pdl_id,
                messages=messages,
                influencing_cols=influencing_cols,
                usage_cols=usage_cols,
            )
            _print_outliers_if_note005(out=out, messages=messages, max_rows=200)


            status = out.get("status")
            model_coeffs = out.get("model_coefficients") or {}
            acc = out.get("accuracy_reference_model")
            pred = out.get("predictive_consumption")

            if status == TrainStatus.OK_ANNUAL_REFERENCE and model_coeffs:
                built += 1

            chosen_hdd = model_coeffs.get("chosen_hdd")
            chosen_cdd = model_coeffs.get("chosen_cdd")
            adjr2 = model_coeffs.get("adjR2_final_model")

            annual_ref = float("nan")
            try:
                if acc is not None and len(acc) > 0:
                    annual_ref = _to_float_or_nan(acc.iloc[0].get("annual_consumption_reference"))
            except Exception:
                annual_ref = float("nan")

            if not math.isnan(annual_ref):
                annual_refs_all_pairs.append(float(annual_ref))
                pairs_used.append(f"{pdl_id}::{fluid}")

            print(f"\nSUMMARY: status={status} | chosen_hdd={chosen_hdd} | chosen_cdd={chosen_cdd} | adjR2={adjr2}")

            if show_details:
                _print_coeffs(model_coeffs)
                _print_metrics(acc)
                _print_predictions(pred, n=head_n)

            pair_key = f"{pdl_id}::{fluid}"
            months_missing_by_pair[pair_key] = _extract_missing_months(pred)

            if messages:
                print("\nMESSAGES (R-like):")
                for m in messages:
                    print("-", m)

            # ============================================================
            # >>> Collect pour ADLS SILVER (predictions_monthly + models)
            # ============================================================

            # A) predictions (month -> month_str obligatoire)
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

                dfp = dfp[[
                    "deliverypoint_id_primaire",
                    "fluid",
                    "month_str",
                    "real_consumption",
                    "predictive_consumption",
                    "confidence_lower95",
                    "confidence_upper95",
                ]]
                pred_dfs.append(dfp)

            # B) models
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
            }

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

    building_metrics = compute_building_metrics_from_pairs(
        annual_refs=annual_refs_all_pairs,
        surface=surface if not math.isnan(surface) else 0.0,
        occupant=occupant if not math.isnan(occupant) else 0.0,
    )

    print("\n===== 1bis) get_site_info (BUILDING METRICS from PDL models) =====")
    print(f"Pairs contributing: {len(pairs_used)} -> {pairs_used}")
    print("total_energy_annual_consumption_reference:", building_metrics["total_energy_annual_consumption_reference"])
    print("ratio_kwh_m2:", building_metrics["ratio_kwh_m2"])
    print("ratio_kwh_occupant:", building_metrics["ratio_kwh_occupant"])

    # ============================================================
    # >>> WRITE SILVER (LATEST) - en fin de run building
    # ============================================================

    # Assemble df predictions
    if pred_dfs:
        df_predictions_all = pd.concat(pred_dfs, ignore_index=True)
    else:
        df_predictions_all = pd.DataFrame(columns=[
            "deliverypoint_id_primaire", "fluid", "month_str",
            "real_consumption", "predictive_consumption",
            "confidence_lower95", "confidence_upper95"
        ])

    # >>> CONDITION CONSERVÉE SUR 2025-11
    # On ne garde que les mois <= "2025-11" (comparaison OK car format YYYY-MM)
    if "month_str" in df_predictions_all.columns:
        df_predictions_all = df_predictions_all[df_predictions_all["month_str"].astype(str) <= "2025-11"]

    df_models_all = pd.DataFrame(model_rows)

    # Écriture silver (mise à jour building via overwrite partition)
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

    print(f"\n✅ Written to silver for building={building_id} | run_id={run_id}")

    print("\n" + "=" * 90)
    print(f"DONE. Couples tested: {computed} | Models built: {built}")
    print("=" * 90)

    print("\nmonths_missing_by_deliverypoint (pair pdl::fluid):")
    for k, v in months_missing_by_pair.items():
        if v:
            print(f"- {k}: {v}")


if __name__ == "__main__":
    run_test_building(
       building_id="building_050",
        start_ref=date(2021, 1, 1),
        end_ref=date(2024, 6, 30),
        start_pred=date(2025, 1, 1),
        end_pred=date(2025 , 12, 1),
        show_details=True,
        head_n=10,




    )

