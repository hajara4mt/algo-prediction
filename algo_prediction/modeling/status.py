# modeling/status.py

from enum import Enum


class TrainStatus(str, Enum):
    """
    Statuts métier alignés avec le code R
    """
    NO_REFERENCE_DATA = "note_000"
    TOO_FEW_OBSERVATIONS = "note_001"
    OK_ANNUAL_REFERENCE = "note_annual_ref"
