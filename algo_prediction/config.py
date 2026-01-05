# algo_prediction/config.py (ou predictive_model/config.py)

from dataclasses import dataclass
import os

# Charger .env seulement si python-dotenv est installé (dev/local)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

AZURE_STORAGE_ACCOUNT_NAME = "stfenixforecast"   # OK si tu veux le garder en clair
AZURE_STORAGE_FILESYSTEM = "fenixlake"           # OK si tu veux le garder en clair

# ✅ secret uniquement via env/.env
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

if not AZURE_STORAGE_ACCOUNT_KEY:
    raise RuntimeError(
        "AZURE_STORAGE_ACCOUNT_KEY manquante. "
        "Définis-la en variable d'environnement ou dans un fichier .env (local)."
    )

@dataclass
class SilverPaths:
    building: str = "silver/building/building.parquet"
    deliverypoint: str = "silver/deliverypoint/deliverypoint.parquet"
    invoice: str = "silver/invoice/invoice.parquet"
    usage_data: str = "silver/usage_data/usage_data.parquet"
    degreedays: str = "silver/degreedays/degreedays_monthly.parquet"

SILVER_PATHS = SilverPaths()
