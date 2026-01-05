# backend_gestion/adls_silver.py

from io import BytesIO
from datetime import date

from azure.storage.filedatalake import DataLakeServiceClient
import pandas as pd

from algo_prediction.config import (
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_FILESYSTEM,
)
from algo_prediction.domain import RequestParams, SiteInfo
from algo_prediction.backend_gestion.base import BackendBase
from azure.core.exceptions import ResourceNotFoundError



# -----------------------------
# Helpers de connexion ADLS
# -----------------------------

def _get_datalake_client() -> DataLakeServiceClient:
    """
    Crée un client DataLakeServiceClient à partir
    des infos définies dans config.py.
    """
    account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"
    return DataLakeServiceClient(
        account_url=account_url,
        credential=AZURE_STORAGE_ACCOUNT_KEY,
    )


def _read_parquet_from_adls(path: str) -> pd.DataFrame:
    """
    Lit un fichier Parquet dans ADLS (filesystem = AZURE_STORAGE_FILESYSTEM)
    et renvoie un DataFrame pandas.
    """
    service_client = _get_datalake_client()
    fs_client = service_client.get_file_system_client(AZURE_STORAGE_FILESYSTEM)
    file_client = fs_client.get_file_client(path)

    download = file_client.download_file()
    data = download.readall()

    # lecture du parquet en mémoire
    df = pd.read_parquet(BytesIO(data))
    return df


def read_parquet_as_df(path: str) -> pd.DataFrame:
    """
    Fonction publique utilisée par preprocessing.dju, preprocessing.usage, etc.
    """
    return _read_parquet_from_adls(path)

# -----------------------------
# Backend concret ADLS Silver
# -----------------------------

class AdlsSilverBackend(BackendBase):
    """
    Implémentation du backend qui lit les données
    dans la zone silver d'ADLS.
    """

    # pour l'instant on hardcode le chemin, on pourra
    # le déplacer dans config.py plus tard si besoin
    BUILDING_SILVER_PATH = "silver/building/building.parquet"
    DELIVERYPOINT_SILVER_PATH = "silver/deliverypoint/deliverypoint.parquet"
    INVOICE_SILVER_PATH = "silver/invoice/invoice.parquet"
    USAGE_DATA_SILVER_PATH = "silver/usage_data/usage_data.parquet"
    DEGREEDAYS_SILVER_PATH = "silver/degreedays/degreedays_monthly.parquet"

    
    

    def get_site_info(self, params: RequestParams) -> SiteInfo:
        """
        Récupère les infos du bâtiment dans silver/building/building.parquet
        et les mappe vers un objet SiteInfo.
        """
        df = _read_parquet_from_adls(self.BUILDING_SILVER_PATH)

        # filtre sur le bâtiment demandé
        mask = df["id_building_primaire"] == params.building_id
        rows = df.loc[mask]

        if rows.empty:
            raise ValueError(
                f"Bâtiment {params.building_id} introuvable dans {self.BUILDING_SILVER_PATH}"
            )

        r = rows.iloc[0]

        # création de l'objet SiteInfo à partir des colonnes du parquet
        site = SiteInfo(
            id_building_primaire=r["id_building_primaire"],
            platform_code=r.get("platform_code"),
            building_code=r.get("building_code"),
            name=r.get("name"),
            latitude=r.get("latitude"),
            longitude=r.get("longitude"),
            organisation=r.get("organisation"),
            address=r.get("address"),
            city=r.get("city"),
            zipcode=r.get("zipcode"),
            country=r.get("country"),
            typology=r.get("typology"),
            geographical_area=r.get("geographical_area"),
            occupant=r.get("occupant"),
            surface=r.get("surface"),
            reference_period_start=r.get("reference_period_start"),
            reference_period_end=r.get("reference_period_end"),
            weather_station=r.get("weather_station"),
            received_at=r.get("received_at"),
        )
        return site

    # Les autres méthodes du backend seront ajoutées plus tard.
    # Pour l'instant, on les laisse en NotImplemented pour rester simple.

    def get_invoices(self, params: RequestParams) -> pd.DataFrame:
         # 1. Lire les Parquet deliverypoint & invoice
        dp_df = _read_parquet_from_adls(self.DELIVERYPOINT_SILVER_PATH)
        inv_df = _read_parquet_from_adls(self.INVOICE_SILVER_PATH)

        # 2. Filtrer les deliverypoints du building
        mask_dp = dp_df["id_building_primaire"] == params.building_id
        dp_building = dp_df.loc[mask_dp]

        if dp_building.empty:
            # Aucun PDL pour ce building -> pas de factures
            return pd.DataFrame()

        pdl_ids = dp_building["deliverypoint_id_primaire"].unique()

        # 3. Filtrer les factures pour ces PDL
        inv_building = inv_df.loc[inv_df["deliverypoint_id_primaire"].isin(pdl_ids)].copy()

        if inv_building.empty:
            # Pas de factures pour ces PDL
            return inv_building

        # 4. Ajouter fluid et fluid_unit via un join
        inv_building = inv_building.merge(
            dp_building[
                [
                    "deliverypoint_id_primaire",
                    "id_building_primaire",
                    "deliverypoint_code",
                    "deliverypoint_number",
                    "fluid",
                    "fluid_unit",
                ]
            ],
            on="deliverypoint_id_primaire",
            how="left",
        )

        # 5. S'assurer que les dates sont bien au format datetime
        inv_building["start"] = pd.to_datetime(inv_building["start"])
        inv_building["end"] = pd.to_datetime(inv_building["end"])

        # On laisse la suite du traitement (prorata, agrégation, etc.)
        # à un module de preprocessing, pour garder ce backend "simple".
        return inv_building



    def get_usage_data(self, params: RequestParams, start, end) -> pd.DataFrame:
        df = _read_parquet_from_adls(self.USAGE_DATA_SILVER_PATH)

        # S'assurer que la colonne date est bien en datetime/date
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Filtre sur le building
        mask_building = df["id_building_primaire"] == params.building_id

        # Filtre sur la plage de dates
        mask_period = (df["date"] >= start) & (df["date"] <= end)

        df_building = df.loc[mask_building & mask_period].copy()

        # On renvoie tel quel, sans pivot pour l'instant
        return df_building
    

    def get_degreedays(self, station_id: str, start, end) -> pd.DataFrame:
        df = _read_parquet_from_adls(self.DEGREEDAYS_SILVER_PATH)

        # filtre sur la station météo
        mask_station = df["station_id"] == station_id
        df_station = df.loc[mask_station].copy()

        if df_station.empty:
            # aucune donnée DJU pour cette station
            return df_station

        # period_month est de type "YYYY-MM" -> on le convertit en date (1er du mois)
        df_station["period_month"] = pd.to_datetime(
            df_station["period_month"] + "-01"
        ).dt.date

        # on veut tous les mois entre start et end (inclus)
        # on se ramène au 1er du mois pour comparer

        start_month = date(start.year, start.month, 1)
        end_month = date(end.year, end.month, 1)

        mask_period = (df_station["period_month"] >= start_month) & (
            df_station["period_month"] <= end_month
        )

        df_period = df_station.loc[mask_period].copy()

        # tri pour être propre
        df_period = df_period.sort_values(
            ["period_month", "indicator", "basis"]
        )

        return df_period 
    

# -----------------------------
# Helpers d'écriture ADLS
# -----------------------------

def _get_fs_client():
    service_client = _get_datalake_client()
    return service_client.get_file_system_client(AZURE_STORAGE_FILESYSTEM)


def write_parquet_to_adls(df: pd.DataFrame, path: str) -> None:
    """
    Écrit un DataFrame pandas en Parquet dans ADLS à l'emplacement `path`.
    """
    fs_client = _get_fs_client()
    file_client = fs_client.get_file_client(path)

    bio = BytesIO()
    # nécessite pyarrow en dépendance (souvent déjà le cas avec pandas)
    df.to_parquet(bio, index=False)
    bio.seek(0)

    # upload overwrite=True pour remplacer le fichier si existe
    file_client.upload_data(bio.getvalue(), overwrite=True)



def delete_adls_prefix(prefix_path: str) -> None:
    """
    Supprime récursivement tout ce qu'il y a sous un 'dossier' (prefix) ADLS.
    Si le path n'existe pas -> ne fait rien (idempotent).
    """
    fs_client = _get_fs_client()

    try:
        paths_iter = fs_client.get_paths(path=prefix_path, recursive=True)
        paths = list(paths_iter)
    except ResourceNotFoundError:
        # Rien à supprimer : dossier inexistant (premier run)
        return

    # supprimer d'abord les fichiers
    for p in paths:
        if not p.is_directory:
            try:
                fs_client.delete_file(p.name)
            except ResourceNotFoundError:
                pass

    # supprimer ensuite les dossiers (du plus profond au plus haut)
    dirs = sorted([p.name for p in paths if p.is_directory], key=len, reverse=True)
    for d in dirs:
        try:
            fs_client.delete_directory(d)
        except ResourceNotFoundError:
            pass
        except Exception:
            # parfois le dossier racine est déjà supprimé / non vide -> on ignore
            pass

    # supprimer le dossier racine du prefix (si existant)
    try:
        fs_client.delete_directory(prefix_path)
    except ResourceNotFoundError:
        pass
    except Exception:
        pass
