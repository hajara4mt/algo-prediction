# backend_gestion/base.py

from abc import ABC, abstractmethod
from datetime import date
from typing import Protocol
import pandas as pd

from algo_prediction.domain import RequestParams, SiteInfo


class BackendBase(ABC):
    

    @abstractmethod
    def get_site_info(self, params: RequestParams) -> SiteInfo:
       
        raise NotImplementedError

    @abstractmethod
    def get_invoices(self, params: RequestParams) -> pd.DataFrame:
       
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
