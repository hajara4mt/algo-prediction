# algo_prediction/preprocessing/model_table.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

from algo_prediction.modeling.imputation import interpolation_missing_linear


@dataclass
class SplitResult:
    model_table: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    messages: List[str]


def _append_once(messages: List[str], msg: str) -> None:
    """Ajoute un message une seule fois (évite les doublons)."""
    if msg not in messages:
        messages.append(msg)


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    """Retourne le dernier jour du mois de ts (ts peut être n'importe quel jour du mois)."""
    m0 = pd.Timestamp(ts.year, ts.month, 1)
    return m0 + relativedelta(months=1) - pd.Timedelta(days=1)


def _month_range_yyyy_mm(start: date, end: date) -> List[str]:
    """Liste des mois entre start et end inclus, format 'YYYY-MM'."""
    start_ts = pd.Timestamp(start).replace(day=1)
    end_ts = pd.Timestamp(end).replace(day=1)
    months = pd.date_range(start=start_ts, end=end_ts, freq="MS")
    return [m.strftime("%Y-%m") for m in months]


def _ensure_month_year_format(df: pd.DataFrame, col: str, fmt: str = "%Y-%m") -> pd.DataFrame:
    """
    S'assure que df[col] existe en format 'YYYY-MM'.
    - Si col n'existe pas, on ne fait rien.
    - Si col est datetime, on format.
    - Si col est déjà string, on tente de parser puis reformat.
    """
    if df.empty or col not in df.columns:
        return df

    out = df.copy()

    if pd.api.types.is_datetime64_any_dtype(out[col]):
        out[col] = out[col].dt.strftime(fmt)
    else:
        parsed = pd.to_datetime(out[col], errors="coerce")
        mask_ok = parsed.notna()
        out.loc[mask_ok, col] = parsed.loc[mask_ok].dt.strftime(fmt)

    return out


def build_model_table_for_pdl_fluid(
    df_invoices_monthly: pd.DataFrame,
    df_dju_monthly: pd.DataFrame,
    df_usage_monthly: pd.DataFrame,
    pdl_id: str,
    fluid: str,
    month_year_invoice: List[str],
    messages: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reproduit la partie R :
      - retrieve_invoice_fluid_pdl + left_join(DJU) + left_join(influencing factors)
      - interpolation_missing(linear) sur les facteurs d’usage uniquement

    Table finale = 1 ligne par mois (mois attendus), avec :
      - identité PDL + fluid toujours présents
      - start/end toujours présents (bornes de mois), même si pas de facture
      - value présent seulement si facture existe (sinon NaN)
      - DJU / usage merged si dispo
    """
    if messages is None:
        messages = []

    # --- 0) Base : mois attendus + identité + bornes de mois (IMPORTANT)
    base = pd.DataFrame({"month_year": month_year_invoice})
    base = _ensure_month_year_format(base, "month_year", fmt="%Y-%m")

    base["start"] = pd.to_datetime(base["month_year"] + "-01")
    base["end"] = base["start"].apply(_month_end)
    base["deliverypoint_id_primaire"] = pdl_id
    base["fluid"] = fluid

    # --- 1) Filtrer invoices pour ce PDL + fluid
    inv = df_invoices_monthly.copy()
    if inv.empty:
        _append_once(messages, f"error_014: ALL INVOICE OF {pdl_id} ARE MISSING (empty invoices input)")
        model = base.copy()
    else:
        inv = _ensure_month_year_format(inv, "month_year", fmt="%Y-%m")

        inv_pf = inv[
            (inv["deliverypoint_id_primaire"] == pdl_id) &
            (inv["fluid"] == fluid)
        ].copy()

        if inv_pf.empty:
            _append_once(messages, f"error_014: ALL INVOICE OF {pdl_id} ARE MISSING for fluid={fluid}")
            model = base.copy()
        else:
            # doublons month_year -> garder dernière facture (stable)
            if "start" in inv_pf.columns:
                inv_pf = inv_pf.sort_values(["start"])
            inv_pf = inv_pf.drop_duplicates(subset=["month_year"], keep="last")

            # --- 2) Merge base + invoices (on ne veut pas perdre pdl/fluid)
            # On merge uniquement les colonnes invoice utiles
            keep_cols = [c for c in ["month_year", "start", "end", "value"] if c in inv_pf.columns]
            inv_pf_small = inv_pf[keep_cols].copy()

            model = base.merge(inv_pf_small, on="month_year", how="left", suffixes=("", "_inv"))

            # si invoice manque, conserver start/end de base
            if "start_inv" in model.columns:
                model["start"] = model["start_inv"].combine_first(model["start"])
                model.drop(columns=["start_inv"], inplace=True)
            if "end_inv" in model.columns:
                model["end"] = model["end_inv"].combine_first(model["end"])
                model.drop(columns=["end_inv"], inplace=True)

    # --- 3) Merge DJU
    dju = df_dju_monthly.copy()
    if not dju.empty and "month_year" in dju.columns:
        # get_degreedays_mentuel peut renvoyer month_year en YYYYMM -> convertir
        dju_my = dju["month_year"].astype(str)
        if dju_my.str.match(r"^\d{6}$").any():
            dju = dju.copy()
            dju["month_year"] = dju_my.str.slice(0, 4) + "-" + dju_my.str.slice(4, 6)

        dju = _ensure_month_year_format(dju, "month_year", fmt="%Y-%m")
        model = model.merge(dju, on="month_year", how="left")
    else:
        _append_once(messages, "note: DJU table is empty (no DJU merged)")

    # --- 4) Merge Usage factors
    usage = df_usage_monthly.copy()
    usage_cols: List[str] = []

    if not usage.empty and "month_year" in usage.columns:
        usage = _ensure_month_year_format(usage, "month_year", fmt="%Y-%m")
        usage_cols = [c for c in usage.columns if c != "month_year"]
        model = model.merge(usage, on="month_year", how="left")
    else:
        _append_once(messages, "note_012: ALL INFLUENCING FACTOR NOT FOUND OR VALUE of INFLUENCING FACTOR IS CONSTANT")

    # --- 5) Interpolation linéaire sur les colonnes usage uniquement (comme R)
    if usage_cols:
        model = model.sort_values("month_year").reset_index(drop=True)
        for col in usage_cols:
            model[col] = interpolation_missing_linear(model[col])

    # --- 6) Nettoyage/tri
    model = model.sort_values("month_year").reset_index(drop=True)

    # colonnes minimales attendues (optionnel)
    # (si tu veux assurer l’ordre)
    # base_cols = ["month_year", "start", "end", "value", "deliverypoint_id_primaire", "fluid"]
    # rest = [c for c in model.columns if c not in base_cols]
    # model = model[base_cols + rest]

    return model


def split_train_test_like_r(
    model_table: pd.DataFrame,
    start_ref: date,
    end_ref: date,
    start_pred: date,
    end_pred: date,
    messages: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split fidèle au R :

      index_ref <- which(invoice_start_date >= start_date_ref & invoice_end_date <= end_date_ref)
      train <- retrieve_invoice_fluid_pdl[index_ref,]

    Et test = mois de prédiction (month_to_predict), avec jointure sur la table globale.
    """
    if messages is None:
        messages = []


     # ✅ R RULE: prediction must be within ONE calendar year
    if start_pred.year != end_pred.year:
        # R message is error_000 but logger prints error_001 -> keep your convention if needed
        messages.append(
            "error_000 :  Model can predict only one calendar year. "
            "Please check start_date_pred and end_date_pred for your request !"
        )
        # Stop-like behavior: no test generated
        empty = model_table.iloc[0:0].copy()
        return empty.reset_index(drop=True), empty.reset_index(drop=True)

    df = model_table.copy()
    if df.empty:
        return df, df

    # start/end toujours présents (base), mais value peut être NaN
    df["start"] = pd.to_datetime(df.get("start"), errors="coerce")
    df["end"] = pd.to_datetime(df.get("end"), errors="coerce")

    # Train : on exige qu'il y ait une valeur réelle (facture)
    # (en R, train vient de retrieve_invoice_fluid_pdl => toujours invoice.consumption)
    has_value = df["value"].notna()

    mask_train = (
        has_value &
        df["start"].notna() & df["end"].notna() &
        (df["start"].dt.date >= start_ref) &
        (df["end"].dt.date <= end_ref)
    )
    train = df.loc[mask_train].copy()

    # Test : mois à prédire (qu’il y ait une facture ou non)
    pred_months = set(_month_range_yyyy_mm(start_pred, end_pred))
    test = df[df["month_year"].isin(pred_months)].copy()

    if train.empty:
        _append_once(messages, "note: train is empty for given reference period (no invoice values inside ref window)")
    if test.empty:
        _append_once(messages, "note: test is empty for given prediction period")

    return train.reset_index(drop=True), test.reset_index(drop=True)
