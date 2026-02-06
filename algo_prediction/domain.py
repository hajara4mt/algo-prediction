# domain.py

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class RequestParams:
    
    building_id: str              
    start_date_ref: date
    end_date_ref: date
    start_date_pred: date
    end_date_pred: date
    

@dataclass
class SiteInfo:
    
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
