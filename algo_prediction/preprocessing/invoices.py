# preprocessing/invoices.py

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


def prepare_invoices_raw(df_inv: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les factures brutes :
    - cast des dates en datetime
    - calcule la durée (en jours, inclusif)
    - nettoie éventuellement certaines valeurs
    """
    if df_inv.empty:
        return df_inv

    df = df_inv.copy()

    # S'assurer que start et end sont des datetime
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Durée de la facture en jours (inclusif, comme dans le R : +1)
    df["invoice_duration"] = (df["end"] - df["start"]).dt.days + 1

    # Optionnel : gérer des valeurs spéciales de "value" (ex: 9999)
    # df.loc[df["value"] == 9999, "value"] = pd.NA

    return df


def normalize_invoices_to_monthly(df_inv: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les factures en factures mensuelles :
      - si une facture est sur un seul mois => gardée telle quelle
      - si une facture couvre plusieurs mois => répartie par prorata journalier
      -

    Retourne un DataFrame avec :
      - start / end adaptés au mois
      - value = consommation du mois
      - month_year / month_year_end_date
      - invoice_duration (nb jours dans le mois)
    """
    if df_inv.empty:
        return df_inv

    df = df_inv.copy()

    # Helper pour savoir si une facture est multi-mois
    def is_multimonth_row(row):
        return (row["start"].year, row["start"].month) != (row["end"].year, row["end"].month)

    # Séparation en 2 groupes
    multi_mask = df.apply(is_multimonth_row, axis=1)
    df_multi = df.loc[multi_mask].copy()
    df_single = df.loc[~multi_mask].copy()

    # On va construire les factures proratisées dans une liste de dict/Series
    prorata_rows = []

    #### Le prorata ici est gardé de la meme facon que la R , c'est qui etaot prsent dans le code est reste le meme , la next amelioration consiste à arrete cette prorata , et faire une separation des données en les allignant avec  les Djus journaliers ! 


    for _, row in df_multi.iterrows():
        start = row["start"]
        end = row["end"]
        total_value = row["value"]
        duration = row["invoice_duration"]

        if duration <= 0 or pd.isna(total_value):
            # On ignore les cas bizarres
            continue

        # conso/jour
        value_per_day = total_value / duration

        # toutes les dates couvertes par la facture
        all_days = pd.date_range(start=start, end=end, freq="D")

        # pour chaque jour -> mois associé (début de mois)
        tmp = pd.DataFrame({"date": all_days})
        tmp["month_start"] = tmp["date"].values.astype("datetime64[M]")  # début de mois
        # conso ce jour-là
        tmp["value_day"] = value_per_day

        # on agrège par mois
        monthly = tmp.groupby("month_start", as_index=False)["value_day"].sum()
        monthly.rename(columns={"value_day": "value"}, inplace=True)

        for _, mrow in monthly.iterrows():
            month_start = mrow["month_start"]
            # fin de mois = début du mois suivant - 1 jour
            month_end = (month_start + relativedelta(months=1)) - pd.Timedelta(days=1)

            new_row = row.copy()
            new_row["start"] = month_start
            new_row["end"] = month_end
            new_row["value"] = mrow["value"]
            new_row["invoice_duration"] = (month_end - month_start).days + 1
            prorata_rows.append(new_row)

    # DataFrame avec les lignes proratisées
    if prorata_rows:
        df_prorata = pd.DataFrame(prorata_rows)
    else:
        df_prorata = pd.DataFrame(columns=df.columns)

    # Concat : factures sur 1 mois + factures multi-mois proratisées
    df_monthly = pd.concat([df_single, df_prorata], ignore_index=True)

    # Ajout de month_year et month_year_end_date comme dans le R
    df_monthly["month_year"] = df_monthly["start"].dt.strftime("%Y%m")
    df_monthly["month_year_end_date"] = df_monthly["end"].dt.strftime("%Y%m")

    return df_monthly


def aggregate_monthly_invoices(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les factures mensuelles par :
      - deliverypoint_id_primaire
      - fluid
      - month_year_end_date

    => une seule ligne par mois / PDL / fluid, avec :
       - value = somme des valeurs du mois
       - start = min(start)
       - end = max(end)
       - invoice_duration = nb jours entre start et end + 1
       - month_year = mois de start
    """
    if df_monthly.empty:
        return df_monthly

    df = df_monthly.copy()

    group_cols = ["deliverypoint_id_primaire", "fluid", "month_year_end_date"]


    # On ordonne pour avoir des choix stables dans le groupby
    df = df.sort_values(["deliverypoint_id_primaire", "fluid", "start"])

    agg_rows = []

    for (pdl_id, fluid, month_end_str), group in df.groupby(group_cols):
        # start = date la plus tôt
        start_min = group["start"].min()
        end_max = group["end"].max()
        value_sum = group["value"].sum()

        # On prend quelques colonnes "de référence" depuis la première ligne
        first = group.iloc[0]

        new_row = {
            "invoice_id_primaire": first.get("invoice_id_primaire"),
            "deliverypoint_id_primaire": pdl_id,
            "invoice_code": first.get("invoice_code"),
            "start": start_min,
            "end": end_max,
            "value": value_sum,
            "invoice_duration": (end_max - start_min).days + 1,
            "id_building_primaire": first.get("id_building_primaire"),
            "deliverypoint_code": first.get("deliverypoint_code"),
            "deliverypoint_number": first.get("deliverypoint_number"),
            "fluid": fluid,
            "fluid_unit": first.get("fluid_unit"),
            "month_year": start_min.strftime("%Y%m"),
            "month_year_end_date": end_max.strftime("%Y%m"),
        }

        agg_rows.append(new_row)

    df_agg = pd.DataFrame(agg_rows)

    # Optionnel : trier le résultat
    df_agg = df_agg.sort_values(
        ["deliverypoint_id_primaire", "fluid", "start"]
    ).reset_index(drop=True)

    return df_agg


def build_monthly_invoices(df_inv: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet pour passer de factures brutes (get_invoices)
    à une conso mensuelle agrégée par PDL / fluid.

    Étapes :
      1) préparation des colonnes de base
      2) normalisation au mois (prorata pour factures multi-mois)
      3) agrégation mensuelle finale
    """
    if df_inv.empty:
        return df_inv

    df_prep = prepare_invoices_raw(df_inv)
    df_monthly = normalize_invoices_to_monthly(df_prep)
    df_agg = aggregate_monthly_invoices(df_monthly)

    return df_agg
