# backend_gestion/silver_results_writer.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import uuid
import pandas as pd

from algo_prediction.backend_gestion.adls_silver import write_parquet_to_adls, delete_adls_prefix

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


@dataclass(frozen=True)
class SilverPaths:
    base: str = "silver/resultats"
    predictions: str = "predictions_monthly"
    models: str = "models"


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _validate_month_str(df: pd.DataFrame) -> None:
    if "month_str" not in df.columns:
        raise ValueError("month_str manquant dans df (format obligatoire YYYY-MM).")

    bad = df["month_str"].astype(str).map(lambda s: MONTH_RE.match(s) is None)
    if bad.any():
        examples = df.loc[bad, "month_str"].head(5).tolist()
        raise ValueError(f"month_str invalide (attendu YYYY-MM). Exemples: {examples}")


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")


def persist_predictions_monthly(
    df_predictions: pd.DataFrame,
    id_building_primaire: str,
    run_id: str | None = None,
    created_at: pd.Timestamp | None = None,
    paths: SilverPaths = SilverPaths(),
) -> tuple[str, pd.Timestamp]:
    """
    Stocke silver/resultats/predictions_monthly/<building_id>/predictions.parquet (LATEST).
    -> mise à jour : on supprime le dossier du building puis on réécrit.
    """
    df = df_predictions.copy()

    required = [
        "deliverypoint_id_primaire",
        "fluid",
        "month_str",
        "real_consumption",
        "predictive_consumption",
    ]
    _ensure_columns(df, required)
    _validate_month_str(df)

    run_id = run_id or str(uuid.uuid4())
    created_at = created_at or _utc_now()

    df["id_building_primaire"] = id_building_primaire
    df["run_id"] = run_id
    df["created_at"] = created_at

    # Colonnes optionnelles IC95
    if "confidence_lower95" not in df.columns:
        df["confidence_lower95"] = pd.NA
    if "confidence_upper95" not in df.columns:
        df["confidence_upper95"] = pd.NA

    # Ordre des colonnes (contrat stable)
    out_cols = [
        "id_building_primaire",
        "deliverypoint_id_primaire",
        "fluid",
        "month_str",
        "real_consumption",
        "predictive_consumption",
        "confidence_lower95",
        "confidence_upper95",
        "run_id",
        "created_at",
    ]
    df = df[out_cols]

    # Nouveau chemin demandé : .../predictions_monthly/<building_id>/
    building_prefix = f"{paths.base}/{paths.predictions}/{id_building_primaire}"
    delete_adls_prefix(building_prefix)

    target = f"{building_prefix}/predictions.parquet"
    write_parquet_to_adls(df, target)

    return run_id, created_at


def persist_models(
    df_models: pd.DataFrame,
    id_building_primaire: str,
    run_id: str,
    created_at: pd.Timestamp,
    paths: SilverPaths = SilverPaths(),
) -> None:
    """
    Stocke silver/resultats/models/<building_id>/<fluid>/<deliverypoint_id_primaire>.parquet (LATEST).
    -> mise à jour : on supprime le dossier du building puis on réécrit tous les fichiers.
    """
    df = df_models.copy()

    required = [
        "deliverypoint_id_primaire",
        "fluid",
        "model_family",
        "chosen_hdd",
        "chosen_cdd",
        "status",
        "b_coefficient",
        "a_hdd",
        "a_cdd",
        "annual_consumption_reference",
        "ME",
        "RMSE",
        "MAE",
        "MPE",
        "MAPE",
        "R2",
        "adjR2",
    ]
    _ensure_columns(df, required)

    df["id_building_primaire"] = id_building_primaire
    df["run_id"] = run_id
    df["created_at"] = created_at

    out_cols = [
        "id_building_primaire",
        "deliverypoint_id_primaire",
        "fluid",
        "run_id",
        "created_at",
        "model_family",
        "chosen_hdd",
        "chosen_cdd",
        "status",
        "b_coefficient",
        "a_hdd",
        "a_cdd",
        "annual_consumption_reference",
        "ME",
        "RMSE",
        "MAE",
        "MPE",
        "MAPE",
        "R2",
        "adjR2",
    ]
    df = df[out_cols]

    # Nouveau chemin demandé : .../models/<building_id>/<fluid>/<deliverypoint>.parquet
    building_prefix = f"{paths.base}/{paths.models}/{id_building_primaire}"
    delete_adls_prefix(building_prefix)

    # Écrire un parquet par (fluid, deliverypoint)
    for (fluid, dp_id), df_one in df.groupby(["fluid", "deliverypoint_id_primaire"], dropna=False):
        fluid_prefix = f"{building_prefix}/{fluid}"
        target = f"{fluid_prefix}/{dp_id}.parquet"
        write_parquet_to_adls(df_one, target)
