# modeling/decision.py

from typing import List, Dict, Optional
import pandas as pd

from algo_prediction.modeling.status import TrainStatus


def decide_training_strategy_like_r(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fluid: str,
    pdl_id: str,
    messages: Optional[List[str]] = None,
) -> Dict:
    """
    Reproduction fidèle du code R :

    - note_000 : pas de données, ou toutes NA, ou toutes zéro
    - note_001 : moins de 6 observations
    - note_annual_ref : OK pour modèle
    """
    if messages is None:
        messages = []

    # =========================
    # CASE 1 — Aucune donnée
    # =========================
    if train.empty:
        messages.append(
            f"note_000: {fluid} PDL {pdl_id}: no reference data"
        )
        return {
            "status": TrainStatus.NO_REFERENCE_DATA,
            "train": train,
            "test": test,
            "messages": messages,
        }

    y = pd.to_numeric(train["value"], errors="coerce")

    # =========================
    # CASE 2 — Toutes NA
    # =========================
    if y.isna().all():
        messages.append(
            f"note_000: {fluid} PDL {pdl_id}: all reference invoice are missing (NA)"
        )
        return {
            "status": TrainStatus.NO_REFERENCE_DATA,
            "train": train,
            "test": test,
            "messages": messages,
        }

    # =========================
    # CASE 3 — Toutes à zéro
    # =========================
    non_na = y.dropna()
    if not non_na.empty and (non_na == 0).all():
        messages.append(
            f"note_000: {fluid} PDL {pdl_id}: all reference invoice are null (zero)"
        )
        return {
            "status": TrainStatus.NO_REFERENCE_DATA,
            "train": train,
            "test": test,
            "messages": messages,
        }

    # =========================
    # CASE 4 — < 6 observations
    # =========================
    if len(train) < 6:
        messages.append(
            f"note_001: {fluid} PDL {pdl_id}: historical data has only {len(train)} OBSERVATIONS"
        )
        return {
            "status": TrainStatus.TOO_FEW_OBSERVATIONS,
            "train": train,
            "test": test,
            "messages": messages,
        }

    # =========================
    # CASE 5 — OK modèle
    # =========================
    messages.append(
        f"note_annual_ref: {fluid} PDL {pdl_id} was used {len(train)} months for ANNUAL REFERENCE"
    )

    return {
        "status": TrainStatus.OK_ANNUAL_REFERENCE,
        "train": train,
        "test": test,
        "messages": messages,
    }
