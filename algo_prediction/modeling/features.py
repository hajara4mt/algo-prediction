# modeling/features.py

from __future__ import annotations
from typing import List, Set
import pandas as pd

DJU_CANDIDATES = ["hdd10", "hdd15", "hdd18", "cdd21", "cdd24", "cdd26"]

META_COLS: Set[str] = {
    "month_year", "start", "end",
    "deliverypoint_id_primaire", "deliverypoint_code",
    "deliverypoint_number", "fluid",
    "invoice_id_primaire", "invoice_code",
    "month_year_end_date",
    "value",
}

def detect_usage_factor_cols(df: pd.DataFrame) -> List[str]:
    
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if c in DJU_CANDIDATES:
            continue
        # on garde uniquement colonnes numériques (ou convertibles)
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            cols.append(c)
    return cols
