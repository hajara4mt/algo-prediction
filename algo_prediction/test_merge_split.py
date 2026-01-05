# algo_prediction/tests/test_merge_split.py

from datetime import date
import pandas as pd

from backend_gestion.adls_silver import AdlsSilverBackend
from algo_prediction.domain import RequestParams

from preprocessing.invoices import build_monthly_invoices
from preprocessing.usage_data import build_monthly_usage_factors
from preprocessing.months import build_month_year_invoice
from preprocessing.dju import get_degreedays_mentuel
from preprocessing.model_table import (
    build_model_table_for_pdl_fluid,
    split_train_test_like_r,
)

from modeling.decision import decide_training_strategy_like_r
from modeling.training import train_like_r
from modeling.dju_model import choose_best_hdd_cdd_like_r


def run_test(
    building_id: str,
    start_ref: date,
    end_ref: date,
    start_pred: date,
    end_pred: date,
) -> None:
    backend = AdlsSilverBackend()

    params = RequestParams(
        building_id=building_id,
        start_date_ref=start_ref,
        end_date_ref=end_ref,
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )

    messages: list[str] = []

    print("\n===== 1) get_site_info =====")
    site = backend.get_site_info(params)
    print(f"Building : {site.id_building_primaire} | {site.name} | station={site.weather_station}")

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
    print(f"fluids: {fluids}")
    print(f"pdls  : {pdls}")

    if not pdls or not fluids:
        print("⚠️  Pas de pdl ou pas de fluid détecté -> stop test")
        return

    # Test 1 pdl + 1 fluid (comme ton test actuel)
    pdl_id = pdls[0]
    fluid = fluids[0]
    print(f"\n--> On teste PDL={pdl_id} | fluid={fluid}")

    print("\n===== 3) build month_year_invoice =====")
    month_year_invoice = build_month_year_invoice(
        df_monthly_invoices=inv_monthly[inv_monthly["deliverypoint_id_primaire"] == pdl_id],
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )
    print(f"month_year_invoice ({len(month_year_invoice)}): {month_year_invoice[:6]} ... {month_year_invoice[-6:]}")

    print("\n===== 4) get_degreedays_mentuel (pivot) =====")
    dju_pivot = get_degreedays_mentuel(
        station_id=site.weather_station,
        month_year_invoice=month_year_invoice,
        messages=messages,
    )
    print(f"dju pivot: {dju_pivot.shape}")

    print("\n===== 5) get_usage_data (raw) -> monthly usage pivot =====")
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
        messages=messages,
    )

    if not usage_pivot.empty:
        # Some pipelines return YYYYMM (e.g. "202501") -> convert to "YYYY-MM"
        if usage_pivot["month_year"].astype(str).str.match(r"^\d{6}$").any():
            usage_pivot["month_year"] = (
                usage_pivot["month_year"].astype(str).str.slice(0, 4)
                + "-"
                + usage_pivot["month_year"].astype(str).str.slice(4, 6)
            )

    print(f"usage pivot: {usage_pivot.shape}")
    if usage_pivot.empty:
        print("usage pivot vide (OK)")
    else:
        print(usage_pivot.head(10))

    print("\n===== 6) build model_table (merge + interpolation usage) =====")
    model_table = build_model_table_for_pdl_fluid(
        df_invoices_monthly=inv_monthly,
        df_dju_monthly=dju_pivot,
        df_usage_monthly=usage_pivot,
        pdl_id=pdl_id,
        fluid=fluid,
        month_year_invoice=month_year_invoice,
        messages=messages,
    )
    print(f"model_table: {model_table.shape}")
    print(model_table.head(15))

    print("\n===== 7) split train/test =====")
    train, test = split_train_test_like_r(
        model_table=model_table,
        start_ref=start_ref,
        end_ref=end_ref,
        start_pred=start_pred,
        end_pred=end_pred,
        messages=messages,
    )

    print(f"train: {train.shape} | test: {test.shape}")
    print("\n--- TRAIN head ---")
    print(train.head(15))
    print("\n--- TEST head ---")
    print(test)

    print("\n===== 7bis) DJU CANDIDATE SELECTION (adjR2) =====")
    # choose_best_hdd_cdd_like_r peut renvoyer:
    #   - (best_hdd, best_cdd)
    #   - (best_hdd, best_cdd, hdd_scores, cdd_scores)
    res = choose_best_hdd_cdd_like_r(
        train=train,
        value_col="value",   # IMPORTANT : facture brute
    )

    if isinstance(res, tuple) and len(res) == 4:
        best_hdd, best_cdd, hdd_scores, cdd_scores = res
    else:
        best_hdd, best_cdd = res
        hdd_scores, cdd_scores = {}, {}

    print("\n--- HDD candidates (adjR2) ---")
    if not hdd_scores:
        print("No HDD scores (function returns only best_hdd/best_cdd or no usable HDD)")
    else:
        for k, v in sorted(hdd_scores.items(), key=lambda x: -x[1]):
            print(f"{k:8s} -> adjR2 = {v:.4f}")

    print("\n--- CDD candidates (adjR2) ---")
    if not cdd_scores:
        print("No CDD scores (function returns only best_hdd/best_cdd or no usable CDD)")
    else:
        for k, v in sorted(cdd_scores.items(), key=lambda x: -x[1]):
            print(f"{k:8s} -> adjR2 = {v:.4f}")

    print("\n--- SELECTED DJU ---")
    print("best_hdd:", best_hdd)
    print("best_cdd:", best_cdd)

    print("\n===== 8) TRAINING DECISION =====")
    decision = decide_training_strategy_like_r(
        train=train,
        test=test,
        fluid=fluid,
        pdl_id=pdl_id,
        messages=messages,
    )
    print("status:", decision["status"])

    # IMPORTANT: détecter les colonnes réellement présentes dans model_table/train/test
    standard_cols = {"month_year", "start", "end", "deliverypoint_id_primaire", "fluid", "value"}
    dju_cols = [c for c in model_table.columns if c.startswith("hdd") or c.startswith("cdd")]
    influencing_cols = [c for c in model_table.columns if c.startswith("fi_")]

    usage_cols = [
        c for c in model_table.columns
        if (c not in standard_cols) and (c not in dju_cols) and (c not in influencing_cols)
    ]

    print("\n===== 8bis) FEATURES DETECTED =====")
    print("dju_cols:", dju_cols)
    print("influencing_cols:", influencing_cols)
    print("usage_cols:", usage_cols)

    print("\n===== 9) TRAINING (MEAN / DJU + postprocess) =====")
    training_out = train_like_r(
        train=train,
        test=test,
        fluid=fluid,
        pdl_id=pdl_id,
        messages=messages,
        influencing_cols=influencing_cols,
        usage_cols=usage_cols,
    )

    print("\nmodel_coefficients:", training_out.get("model_coefficients"))

    acc = training_out.get("accuracy_reference_model")
    pred = training_out.get("predictive_consumption")

    print("\n--- accuracy_reference_model ---")
    print(acc if acc is not None else "None")

    print("\n--- predictive_consumption (head) ---")
    if pred is not None and hasattr(pred, "head"):
        print(pred.head(20))
    else:
        print(pred)

    print("\n===== Messages (warnings/errors style R) =====")
    for m in messages:
        print("-", m)


if __name__ == "__main__":
    run_test(
        building_id="building_022",  # <-- remplace par ton building
        start_ref=date(2022, 1, 1),
        end_ref=date(2024, 6, 30),
        start_pred=date(2025, 1, 1),
        end_pred=date(2025, 12, 31),
    )
