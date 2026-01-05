# domain.py

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class RequestParams:
    """
    Paramètres d'entrée de l'algo de prédiction.

    Ils seront construits à partir du JSON que ton API reçoit :
      - id_building_primaire
      - start_date_ref / end_date_ref
      - start_date_pred / end_date_pred
    """
    building_id: str              # correspond à id_building_primaire
    start_date_ref: date
    end_date_ref: date
    start_date_pred: date
    end_date_pred: date
    # Pour l'instant on ne gère pas les périodes de récidive,
    # donc on ne les met pas ici. On pourra les ajouter plus tard si besoin.


@dataclass
class SiteInfo:
    """
    Représentation Python d'un bâtiment, telle qu'elle vient du Parquet
    silver/building.

    On garde tous les champs utiles pour l'algo (surface, occupants,
    station météo, etc.) + quelques métadonnées.
    """
    id_building_primaire: str
    platform_code: Optional[str]
    building_code: Optional[str]
    name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    organisation: Optional[str]
    address: Optional[str]
    city: Optional[str]
    zipcode: Optional[str]
    country: Optional[str]
    typology: Optional[str]
    geographical_area: Optional[int]
    occupant: Optional[int]
    surface: Optional[float]
    reference_period_start: Optional[date]
    reference_period_end: Optional[date]
    weather_station: Optional[str]
    received_at: Optional[datetime]
