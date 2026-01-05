# tests/test_degreedays.py

from datetime import date
from typing import List

import pandas as pd

from preprocessing.dju import get_degreedays_mentuel


def build_month_year_invoice(start: date, end: date) -> List[str]:
    """
    Construit une liste de mois au format 'YYYY-MM'
    entre start et end inclus.
    Équivalent de month_year_invoice dans le R.
    """
    # on utilise pandas juste pour simplifier la génération des mois
    months = pd.date_range(start=start, end=end, freq="MS")  # MS = Month Start
    return [d.strftime("%Y-%m") for d in months]


def main():
    # 🔹 1) Paramètres de test
    station_id = "SBRJ"  # adapte avec une station existante chez toi

    # période sur laquelle tu veux tester les DJU
    start = date(2024, 6, 1)
    end = date(2025, 5, 31)

    # 🔹 2) Construire la liste des mois attendus (month_year_invoice en R)
    month_year_invoice = build_month_year_invoice(start, end)
    print("Mois demandés (month_year_invoice) :", month_year_invoice)

    # 🔹 3) Appel de la fonction avec un buffer de messages
    messages: List[str] = []

    df_dju = get_degreedays_mentuel(
        station_id=station_id,
        month_year_invoice=month_year_invoice,
        messages=messages,
    )

    # 🔹 4) Affichage des résultats
    print("\n===== RESULTAT DJU PIVOTE =====")
    print(df_dju)

    print("\n===== MESSAGES (notes / erreurs) =====")
    for msg in messages:
        print("-", msg)


if __name__ == "__main__":
    main()
