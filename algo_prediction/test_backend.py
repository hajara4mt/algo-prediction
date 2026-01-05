from datetime import date

from algo_prediction.domain import RequestParams
from backend_gestion.adls_silver import AdlsSilverBackend


def main():
    # TODO : remplace par un id_building_primaire réel présent dans ton parquet
    BUILDING_ID = "building_010"  # <-- à adapter

    # Dates de test (adapte aussi si besoin)
    start_ref = date(2020, 1, 1)
    end_ref = date(2020, 12, 31)
    start_pred = date(2025, 1, 1)
    end_pred = date(2025, 12, 31)

    params = RequestParams(
        building_id=BUILDING_ID,
        start_date_ref=start_ref,
        end_date_ref=end_ref,
        start_date_pred=start_pred,
        end_date_pred=end_pred,
    )

    backend = AdlsSilverBackend()

    print("===== TEST get_site_info =====")
    try:
        site = backend.get_site_info(params)
        print("Building ID     :", site.id_building_primaire)
        print("Name            :", site.name)
        print("City            :", site.city)
        print("Country         :", site.country)
        print("Surface (m2)    :", site.surface)
        print("Occupants       :", site.occupant)
        print("Geo area        :", site.geographical_area)
        print("Weather station :", site.weather_station)
        print()
    except Exception as e:
        print("Erreur get_site_info :", e)
        return

    print("===== TEST get_invoices =====")
    try:
        df_inv = backend.get_invoices(params)
        print("Nb factures trouvées :", len(df_inv))
        if not df_inv.empty:
            print("Colonnes :", df_inv.columns.tolist())
            print(df_inv.head())
            print("Fluids présents :", df_inv["fluid"].unique())
        print()
    except Exception as e:
        print("Erreur get_invoices :", e)

    print("===== TEST get_usage_data =====")
    try:
        # on prend une plage large pour voir ce qui existe
        df_usage = backend.get_usage_data(
            params,
            start=date(2018, 1, 1),
            end=date(2025, 12, 31),
        )
        print("Nb lignes usage_data :", len(df_usage))
        if not df_usage.empty:
            print("Colonnes :", df_usage.columns.tolist())
            print(df_usage.head())
            print("Types présents :", df_usage["type"].unique())
        print()
    except Exception as e:
        print("Erreur get_usage_data :", e)

    print("===== TEST get_degreedays =====")
    try:
        station_id = site.weather_station
        if not station_id:
            print("⚠️ Aucune station météo définie pour ce building, test DJU ignoré.")
        else:
            df_dju = backend.get_degreedays(
                station_id=station_id,
                start=start_ref,
                end=end_pred,
            )
            print(f"Station météo testée : {station_id}")
            print("Nb lignes DJU :", len(df_dju))
            if not df_dju.empty:
                print("Colonnes :", df_dju.columns.tolist())
                print(df_dju.head(10))
                print("Indicators :", df_dju["indicator"].unique())
                print("Bases :", df_dju["basis"].unique())
        print()
    except Exception as e:
        print("Erreur get_degreedays :", e)


if __name__ == "__main__":
    main()
