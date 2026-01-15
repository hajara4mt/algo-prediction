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
    if messages is None:
        messages = []

    # --- 0) Base mois attendus
    base = pd.DataFrame({"month_year": month_year_invoice})
    base = _ensure_month_year_format(base, "month_year", fmt="%Y-%m")

    base["start"] = pd.to_datetime(base["month_year"] + "-01")
    base["end"] = base["start"].apply(_month_end)
    base["deliverypoint_id_primaire"] = pdl_id
    base["fluid"] = fluid

    # --- 1) Filtrer invoices PDL + fluid
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
            # Assurer start/end en datetime
            if "start" in inv_pf.columns:
                inv_pf["start"] = pd.to_datetime(inv_pf["start"], errors="coerce")
            if "end" in inv_pf.columns:
                inv_pf["end"] = pd.to_datetime(inv_pf["end"], errors="coerce")

            before_rows = len(inv_pf)
            before_months = inv_pf["month_year"].nunique(dropna=True)

            # Debug: mois avec plusieurs lignes (cause fréquente du 45 vs 33)
            dup_counts = inv_pf["month_year"].value_counts()
            n_dup_months = int((dup_counts > 1).sum())
            max_dup = int(dup_counts.max()) if len(dup_counts) else 0
            if n_dup_months > 0:
                _append_once(
                    messages,
                    f"debug_invoices: {n_dup_months} months have multiple invoice rows (max rows in a month={max_dup})"
                )

            # ✅ CORRECTION R-LIKE:
            # R finit avec 1 ligne par mois (après prorata + agrégation) => on agrège ici par month_year
            # value = somme, start=min, end=max
            # (et on ignore les lignes sans value si besoin)
            inv_pf_agg = (
                inv_pf.groupby("month_year", as_index=False)
                .agg(
                    start=("start", "min"),
                    end=("end", "max"),
                    value=("value", "sum"),
                )
            )

            after_rows = len(inv_pf_agg)
            after_months = inv_pf_agg["month_year"].nunique(dropna=True)

            _append_once(
                messages,
                f"debug_invoices_agg: rows {before_rows}->{after_rows} ; unique months {before_months}->{after_months}"
            )

            # --- Merge base + invoices agrégées
            model = base.merge(inv_pf_agg, on="month_year", how="left", suffixes=("", "_inv"))

            # conserver start/end base si NA côté invoice
            if "start_inv" in model.columns:
                model["start"] = model["start_inv"].combine_first(model["start"])
                model.drop(columns=["start_inv"], inplace=True)
            if "end_inv" in model.columns:
                model["end"] = model["end_inv"].combine_first(model["end"])
                model.drop(columns=["end_inv"], inplace=True)

    # --- 3) Merge DJU
    dju = df_dju_monthly.copy()
    if not dju.empty and "month_year" in dju.columns:
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

    # --- 5) Interpolation linéaire sur usage uniquement
    if usage_cols:
        model = model.sort_values("month_year").reset_index(drop=True)
        for col in usage_cols:
            model[col] = interpolation_missing_linear(model[col])

    model = model.sort_values("month_year").reset_index(drop=True)

    n_val = int(model["value"].notna().sum()) if "value" in model.columns else 0
    _append_once(messages, f"debug_model_table: months_total={len(model)} months_with_value={n_val}")

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

    if start_pred > end_pred:
        raise ValueError("start_pred must be <= end_pred")

    df = model_table.copy()
    if df.empty:
        return df, df

    # start/end toujours présents (base), mais value peut être NaN
    df["start"] = pd.to_datetime(df.get("start"), errors="coerce")
    df["end"] = pd.to_datetime(df.get("end"), errors="coerce")

    # Train : on exige une valeur réelle (facture)
    has_value = df["value"].notna()

    mask_train = (
        has_value &
        df["start"].notna() & df["end"].notna() &
        (df["start"].dt.date >= start_ref) &
        (df["end"].dt.date <= end_ref)
    )
    train = df.loc[mask_train].copy()

    # Log R-like : nombre de mois utilisés pour la référence
    pdl = df["deliverypoint_id_primaire"].iloc[0] if "deliverypoint_id_primaire" in df.columns and len(df) else "unknown_pdl"
    fl = df["fluid"].iloc[0] if "fluid" in df.columns and len(df) else "unknown_fluid"
    _append_once(messages, f"note_annual_ref: {fl} PDL {pdl} was used {len(train)} months for ANNUAL REFERENCE")

    # Debug bornes effectives
    _append_once(messages, f"debug_ref_bounds: start_ref={start_ref} end_ref={end_ref}")

    if len(train) > 0:
        min_m = str(train["month_year"].min())
        max_m = str(train["month_year"].max())
        _append_once(messages, f"debug_ref_window: train month_year range = {min_m} -> {max_m}")

        # voir si des starts dupliqués (comme le check R sur invoice_start_date)
        if "start" in train.columns:
            dup_start = int(train["start"].duplicated().sum())
            if dup_start > 0:
                _append_once(messages, f"debug_duplicates: {dup_start} duplicated start dates found in train")

    # Test : mois à prédire (qu’il y ait une facture ou non)
    pred_months = set(_month_range_yyyy_mm(start_pred, end_pred))
    test = df[df["month_year"].isin(pred_months)].copy()

    # Logs test
    _append_once(messages, f"debug_pred_bounds: start_pred={start_pred} end_pred={end_pred} months={len(pred_months)}")
    if len(test) > 0:
        _append_once(messages, f"debug_test: months_in_test={len(test)} month_year range={test['month_year'].min()} -> {test['month_year'].max()}")
    else:
        _append_once(messages, "note: test is empty for given prediction period")

    if train.empty:
        _append_once(messages, "note: train is empty for given reference period (no invoice values inside ref window)")

    return train.reset_index(drop=True), test.reset_index(drop=True)
