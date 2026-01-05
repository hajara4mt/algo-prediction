from datetime import date

from backend_gestion.adls_silver import AdlsSilverBackend
from algo_prediction.domain import RequestParams
from preprocessing.invoices import build_monthly_invoices
from preprocessing.months import build_month_year_invoice
from preprocessing.dju import get_degreedays_mentuel


def main():
    # 1) Paramètres de test
    building_id = "building_010"  # <-- mets un building qui existe chez toi

    # Les dates ref ne sont pas nécessaires pour month_year_invoice,
    # mais on les met ici car ton API les aura de toute façon
    start_ref = date(2024, 1, 1)
    end_ref = date(2024, 12, 31)

    # Période de prédiction demandée
    start_pred = date(2025, 1, 1)
    end_pred = date(2025, 12, 31)

    params = RequestParams(
        building_id=building_id,
        start_date_ref=start_ref,
        end_date_ref=end_ref,
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )

    backend = AdlsSilverBackend()

    # 2) Site info -> station météo
    site = backend.get_site_info(params)
    station_id = site.weather_station
    print("\n===== TEST site info =====")
    print("Building:", site.id_building_primaire, "-", site.name)
    print("Weather station:", station_id)

    # 3) Factures brutes -> factures mensuelles agrégées
    df_inv_raw = backend.get_invoices(params)
    print("\n===== TEST invoices raw =====")
    print("Nb factures brutes:", len(df_inv_raw))

    df_inv_monthly = build_monthly_invoices(df_inv_raw)
    print("\n===== TEST invoices monthly =====")
    print("Nb lignes mensuelles:", len(df_inv_monthly))
    print("Mois présents (invoices):", sorted(df_inv_monthly["month_year"].unique().tolist())[:10], "...")

    # 4) Construire month_year_invoice (factures + prédiction)
    month_year_invoice = build_month_year_invoice(
        df_monthly_invoices=df_inv_monthly,
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )
    print("\n===== TEST month_year_invoice =====")
    print("Nb mois:", len(month_year_invoice))
    print("First 12:", month_year_invoice[:12])
    print("Last 12 :", month_year_invoice[-12:])

    # 5) DJU mensuels pour ces mois
    messages = []
    dju = get_degreedays_mentuel(
        station_id=station_id,
        month_year_invoice=month_year_invoice,
        messages=messages,
    )

    print("\n===== TEST get_degreedays_mentuel =====")
    print("Nb lignes DJU:", len(dju))
    print("Colonnes:", dju.columns.tolist())
    print(dju.head(5))

    print("\n===== MESSAGES =====")
    for m in messages:
        print("-", m)


if __name__ == "__main__":
    main()
