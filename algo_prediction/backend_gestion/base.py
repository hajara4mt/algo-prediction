# backend_gestion/base.py

from abc import ABC, abstractmethod
from datetime import date
from typing import Protocol
import pandas as pd

from algo_prediction.domain import RequestParams, SiteInfo


class BackendBase(ABC):
    """
    Interface de base pour un backend de données.

    Une implémentation (ex: AdlsSilverBackend) devra
    fournir toutes ces méthodes pour que l'algo
    puisse récupérer les données dont il a besoin.
    """

    @abstractmethod
    def get_site_info(self, params: RequestParams) -> SiteInfo:
        """
        Récupère les infos du bâtiment (surface, occupants, station météo, etc.)
        à partir de la silver `building`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_invoices(self, params: RequestParams) -> pd.DataFrame:
        """
        Récupère toutes les factures (invoice) associées au bâtiment :
        - join deliverypoint + invoice
        - pas encore de prorata ni d'agrégation, juste les lignes brutes.
        """
        raise NotImplementedError

    @abstractmethod
    def get_usage_data(self, params: RequestParams, start: date, end: date) -> pd.DataFrame:
        """
        Récupère les usage_data (facteurs d'influence) pour le bâtiment
        entre start et end.
        """
        raise NotImplementedError

    @abstractmethod
    def get_degreedays(self, station_id: str, start: date, end: date) -> pd.DataFrame:
        """
        Récupère les DJU mensuels pour une station météo donnée
        entre start et end à partir du Parquet silver/degreedays.
        """
        raise NotImplementedError
