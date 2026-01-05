# algo_prediction/preprocessing/months.py

from datetime import date
from typing import List

import pandas as pd


def build_month_year_invoice(
    df_monthly_invoices: pd.DataFrame,
    start_date_pred: date,
    end_date_pred: date,
) -> List[str]:
    """
    Construit la liste des mois "YYYY-MM" à couvrir pour les DJU & les usages,
    en suivant la logique du code R :

      month_year_invoice = (mois des factures) U (mois de la période de prédiction)

    Paramètres
    ----------
    df_monthly_invoices : DataFrame
        DataFrame de factures déjà agrégées mensuellement (une ligne par mois / PDL / fluide),
        typiquement la sortie de build_monthly_invoices(...) filtrée sur un PDL + fluide.
        Doit contenir une colonne 'start' (datetime/date) correspondant au début de la période.

    start_date_pred : date
        Date de début de la période de prédiction.

    end_date_pred : date
        Date de fin de la période de prédiction.

    Retour
    ------
    List[str]
        Liste triée de mois au format 'YYYY-MM', sans doublons.
    """
    # --- 1) Mois issus des factures (référence / historique) ---
    if "start" not in df_monthly_invoices.columns or df_monthly_invoices.empty:
        invoice_months: set[str] = set()
    else:
        # on s'assure que 'start' est bien au format datetime
        start_dates = pd.to_datetime(df_monthly_invoices["start"])

        # on ramène au 1er du mois, puis on formate en "YYYY-MM"
        invoice_months = {
            d.to_period("M").strftime("%Y-%m")
            for d in start_dates
        }

    # --- 2) Mois de la période de prédiction ---
    if start_date_pred is None or end_date_pred is None:
        pred_months: set[str] = set()
    else:
        # "MS" = Month Start, donc générer 1er de chaque mois
        pred_range = pd.date_range(start=start_date_pred, end=end_date_pred, freq="MS")
        pred_months = {d.strftime("%Y-%m") for d in pred_range}

    # --- 3) Union des deux ensembles ---
    all_months = invoice_months.union(pred_months)

    # --- 4) Liste triée ---
    month_year_invoice = sorted(all_months)

    return month_year_invoice
