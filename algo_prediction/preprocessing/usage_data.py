# algo_prediction/preprocessing/usage.py

from datetime import date
from typing import Optional, List

import pandas as pd


def prepare_usage_raw(
    df_usage: pd.DataFrame,
    building_id: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    if df_usage.empty:
        return df_usage

    df = df_usage.copy()
    df = df[df["id_building_primaire"] == building_id]

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]

    return df


def build_monthly_usage_factors(
    df_usage: pd.DataFrame,
    building_id: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    messages: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Construit les facteurs d'usage mensuels pour un bâtiment.
    Si aucun facteur n'est trouvable ou que tous sont constants,
    on ajoute un message similaire à :
      "ALL INFLUENCING FACTOR NOT FOUND OR VALUE of INFLUENCING FACTOR IS CONSTANT"
    """
    df = prepare_usage_raw(df_usage, building_id=building_id, start=start, end=end)

    if df.empty:
        # cas équivalent à: "pas de données d'influencing factor pour ce site"
        if messages is not None:
            messages.append(
                "note_012: ALL INFLUENCING FACTOR NOT FOUND OR VALUE of INFLUENCING FACTOR IS CONSTANT"
            )
        return pd.DataFrame(columns=["month_year"])

    df["month_year"] =  df["date"].dt.strftime("%Y-%m")

    monthly = (
        df.groupby(["month_year", "type"], as_index=False)["value"]
        .mean()
    )

    if monthly.empty:
        if messages is not None:
            messages.append(
                "note_012: ALL INFLUENCING FACTOR NOT FOUND OR VALUE of INFLUENCING FACTOR IS CONSTANT"
            )
        return pd.DataFrame(columns=["month_year"])

    pivot = (
        monthly.pivot(index="month_year", columns="type", values="value")
        .reset_index()
    )

    factor_cols = [col for col in pivot.columns if col != "month_year"]

    cols_to_keep = ["month_year"]
    for col in factor_cols:
        series = pivot[col]
        if series.dropna().empty:
            continue
        if series.dropna().std() != 0:
            cols_to_keep.append(col)
        # sinon: facteur constant -> on le retire, comme dans le R

    # Si au final on n’a gardé que month_year => aucun facteur exploitable
    if cols_to_keep == ["month_year"]:
        if messages is not None:
            messages.append(
                "note_012: ALL INFLUENCING FACTOR NOT FOUND OR VALUE of INFLUENCING FACTOR IS CONSTANT"
            )
        # on renvoie quand même une colonne month_year, vide ou non
        pivot = pivot[["month_year"]].copy()
    else:
        pivot = pivot[cols_to_keep]

    pivot = pivot.sort_values("month_year").reset_index(drop=True)

    return pivot
