# algo_prediction/preprocessing/dju.py

import pandas as pd
from typing import List, Optional

from algo_prediction.backend_gestion.adls_silver import read_parquet_as_df

DJU_COL_SPECS = [
    ("hdd", 10),
    ("hdd", 15),
    ("hdd", 18),
    ("cdd", 21),
    ("cdd", 24),
    ("cdd", 26),
]

DJU_COLUMNS = [f"{ind}{basis}" for ind, basis in DJU_COL_SPECS]


def get_degreedays_mentuel(
    station_id: str,
    month_year_invoice: List[str],  # ex: ["2025-01", "2025-02", ...]
    messages: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reproduit la logique R de la section 'GET DJU CORR. STATION METEO' en utilisant
    le parquet mensuel silver/degreedays/degreedays_monthly.parquet.

    - Filtre sur station_id
    - Pour chaque ref hdd10, hdd15, ..., cdd26:
        * Vérifie si des données existent
        * Log error_008 si aucune ligne pour cette ref
        * Contrôle les mois manquants vs month_year_invoice -> error_009
    - Construit un pivot: month_year | hdd10 | hdd15 | ... | cdd26
    - Si aucune ref n'est disponible -> error_010 + retourne DF vide
    """
    if messages is None:
        messages = []

    # 1) Lire le parquet Silver
    df = read_parquet_as_df("silver/degreedays/degreedays_monthly.parquet")

    if df.empty:
        messages.append("error_010: ALL DJU REFERENCE NOT FOUND (no degreedays data in silver)")
        return empty_dju_frame()

    # 2) Filtrer sur la station
    df = df[df["station_id"] == station_id].copy()

    if df.empty:
        messages.append(f"error_010: ALL DJU REFERENCE NOT FOUND FOR STATION {station_id}")
        return empty_dju_frame()

    # 3) On garde uniquement la période qui nous intéresse (month_year_invoice)
    # period_month est au format "YYYY-MM"
    df = df[df["period_month"].isin(month_year_invoice)].copy()

    if df.empty:
        messages.append(
            f"error_010: ALL DJU REFERENCE NOT FOUND FOR STATION {station_id} "
            f"ON REQUESTED MONTHS {month_year_invoice}"
        )
        return empty_dju_frame()

    # month_year = period_month (format "YYYY-MM" partout)
    df["month_year"] = df["period_month"]

    # 4) Traitement ref par ref (hdd10, hdd15, ..., cdd26)
    dju_frames = []
    dju_names = []

    for (indicator, basis) in DJU_COL_SPECS:
        ref_name = f"{indicator}{basis}"

        # filtrer cette ref
        mask_ref = (df["indicator"] == indicator) & (df["basis"] == basis)
        df_ref = df[mask_ref].copy()

        if df_ref.empty:
            # cas error_008 dans le R
            messages.append(
                f"error_008: Your request RETRIEVE_DJU does not return data for reference {ref_name}"
            )
            continue

        # On sait que nos données sont déjà mensuelles (1 ligne par station/mois/indicator/basis)
        tmp = df_ref[["month_year", "period_month", "value"]].copy()
        tmp = tmp.rename(columns={"value": ref_name})

        # Contrôle des mois manquants (error_009)
        missing = sorted(
            m for m in month_year_invoice
            if m not in tmp["period_month"].unique().tolist()
        )
        if missing:
            messages.append(
                f"error_009: {ref_name} has missing DJU data for months {missing}"
            )

        # On ne garde que month_year + ref_name
        tmp = tmp[["month_year", ref_name]].drop_duplicates()

        dju_frames.append(tmp)
        dju_names.append(ref_name)

    # 5) Si aucune référence n'a donné de données -> error_010
    if not dju_frames:
        messages.append("error_010: ALL DJU REFERENCE NOT FOUND")
        return empty_dju_frame()

    # 6) Join de toutes les ref sur month_year (équivalent de plyr::join_all)
    dju_merged = dju_frames[0]
    for other in dju_frames[1:]:
        dju_merged = dju_merged.merge(other, on="month_year", how="left")

    # 7) S'assurer que toutes les colonnes DJU_COLUMNS existent
    for col in DJU_COLUMNS:
        if col not in dju_merged.columns:
            dju_merged[col] = None

    # 8) Réordonner & trier
    dju_merged = dju_merged[["month_year"] + DJU_COLUMNS]
    dju_merged = dju_merged.sort_values("month_year").reset_index(drop=True)

    return dju_merged


def empty_dju_frame() -> pd.DataFrame:
    cols = ["month_year"] + DJU_COLUMNS
    return pd.DataFrame(columns=cols)
