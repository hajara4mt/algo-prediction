# Algo Prediction

## 📋 Project Overview

**Algo Prediction** est une implémentation Python de l'algorithme R `predictive_consumption_modelisation` développé par Energisme pour la modélisation prédictive de consommation énergétique des bâtiments, conçu pour être déployé sur Azure Function App.

Le service prend en entrée un identifiant de bâtiment et deux périodes : une période de référence (données historiques) et une période de prédiction. Il récupère automatiquement depuis Azure Data Lake Storage (ADLS) les factures énergétiques, les Degrés-Jours Unifiés (DJU) de la station météo associée, et les éventuels facteurs d'influence (occupation, surface, etc.).

L'algorithme original, écrit en R, utilise les données historiques de facturation énergétique (gaz, électricité) combinées aux Degrés-Jours Unifiés (DJU) pour construire un modèle de régression linéaire capable de prédire les consommations futures. Cette version Python a été conçue pour reproduire fidèlement le comportement de l'algorithme R, fonction par fonction, afin de garantir des résultats identiques tout en permettant un déploiement cloud-native sur Azure Function App. Chaque étape du pipeline R a son équivalent Python documenté, avec une attention particulière portée aux détails d'implémentation.

Les résultats (prédictions mensuelles, coefficients du modèle, métriques de performance, outliers détectés) sont persistés en format Parquet sur ADLS et retournés en JSON via l'API HTTP.

---

## 📁 Repository Structure

```
algo_prediction/
│
├── domain.py                    # Dataclasses RequestParams, SiteInfo
├── config.py                    # Variables d'environnement ADLS
│
├── algo_services/
│   └── run_algo_services.py     # Pipeline principal (≈ R: predictive_consumption_modelisation)
│
├── backend_gestion/
│   ├── base.py                  # Interface abstraite BackendBase
│   ├── adls_silver.py           # Lecture/écriture ADLS Parquet
│   └── silver_results_writer.py # Persistance résultats (predictions, models)
│
├── preprocessing/
│   ├── invoices.py              # Agrégation mensuelle factures (≈ R: prorata processing)
│   ├── months.py                # Génération liste month_year (≈ R: month_year_invoice)
│   ├── dju.py                   # Récupération DJU mensuels (≈ R: retrieve_dju_data)
│   ├── usage_data.py            # Pivot facteurs d'usage (≈ R: retrieve_influencing_factor)
│   └── model_table.py           # Construction table modèle + split train/test (≈ R: index_ref)
│
└── modeling/
    ├── status.py                # Enum TrainStatus (NO_DATA, TOO_FEW, OK)
    ├── decision.py              # Stratégie d'entraînement (≈ R: control missing/zero data)
    ├── imputation.py            # Imputation valeurs manquantes (≈ R: ranking_method)
    ├── outliers.py              # Détection anomalies (≈ R: ts_anomaly_detection)
    ├── postprocess.py           # Pipeline Y: missing → outliers → best Y (≈ R: lignes 1160-1268)
    ├── dju_model.py             # Régression DJU + sélection HDD/CDD (≈ R: lm + which.max)
    ├── mean_model.py            # Modèle moyenne simple si n < 6 (≈ R: note_001 branch)
    ├── metrics.py               # Métriques régression (≈ R: forecast::accuracy)
    └── training.py              # Orchestration entraînement (≈ R: boucle fluid/pdl)
```

---

## 🏗️ Architecture du Code

### Vue Globale

```
                                    ┌─────────────────────┐
                                    │   Azure Function    │
                                    │   HTTP Trigger      │
                                    └──────────┬──────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         algo_services/run_algo_services.py                   │
│                         run_building_and_persist()                           │
│                         Point d'entrée principal                             │
└──────────────────────────────────────────────────────────────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  backend_gestion/   │    │   preprocessing/    │    │     modeling/       │
│                     │    │                     │    │                     │
│  • adls_silver.py   │───▶│  • invoices.py      │───▶│  • training.py      │
│  • silver_results_  │    │  • dju.py           │    │  • postprocess.py   │
│    writer.py        │◀───│  • usage_data.py    │    │  • outliers.py      │
│                     │    │  • model_table.py   │    │  • dju_model.py     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                                                     │
           ▼                                                     ▼
┌─────────────────────┐                               ┌─────────────────────┐
│    ADLS Gen2        │                               │   JSON Response     │
│    (Parquet)        │                               │   (API Output)      │
└─────────────────────┘                               └─────────────────────┘
```

### Flux de Données

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│  building_id, start_ref, end_ref, start_pred, end_pred                     │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         1. DATA RETRIEVAL                                   │
│  backend_gestion/adls_silver.py                                            │
├────────────────────────────────────────────────────────────────────────────┤
│  get_site_info()      → Infos bâtiment + station météo                     │
│  get_invoices()       → Factures brutes                                    │
│  get_usage_data()     → Facteurs d'usage                                   │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         2. PREPROCESSING                                    │
│  preprocessing/                                                             │
├────────────────────────────────────────────────────────────────────────────┤
│  build_monthly_invoices()           → Agrégation mensuelle                 │
│  get_degreedays_mentuel()           → DJU par station                      │
│  build_monthly_usage_factors()      → Pivot facteurs                       │
│  build_model_table_for_pdl_fluid()  → Table finale                         │
│  split_train_test_like_r()          → Séparation train/test                │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         3. MODELING                                         │
│  modeling/                                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  train_like_r()                                                      │  │
│  │  Orchestration principale                                            │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                         │
│          ┌───────────────────────┼───────────────────────┐                 │
│          ▼                       ▼                       ▼                 │
│  ┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│  │ decision.py   │     │ postprocess.py  │     │ dju_model.py    │        │
│  │ Stratégie     │     │ build_y_like_r  │     │ Régression      │        │
│  └───────────────┘     └────────┬────────┘     └─────────────────┘        │
│                                 │                                          │
│                    ┌────────────┴────────────┐                             │
│                    ▼                         ▼                             │
│           ┌───────────────┐        ┌─────────────────┐                     │
│           │ imputation.py │        │  outliers.py    │                     │
│           │ ranking_method│        │ ts_anomaly_     │                     │
│           │ _like_r()     │        │ detection_like_r│                     │
│           └───────────────┘        └─────────────────┘                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         4. OUTPUT                                           │
│  backend_gestion/silver_results_writer.py                                  │
├────────────────────────────────────────────────────────────────────────────┤
│  persist_predictions_monthly()  → Parquet predictions                      │
│  persist_models()               → Parquet models                           │
│                                                                            │
│  Return JSON: results, models, outliers_details, outliers_notes            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Dépendances entre Modules

```
training.py
    ├── decision.py          (decide_training_strategy_like_r)
    ├── postprocess.py       (build_y_like_r)
    │       ├── imputation.py    (ranking_method_like_r)
    │       ├── outliers.py      (ts_anomaly_detection_like_r)
    │       └── dju_model.py     (r2_and_adj_r2)
    ├── dju_model.py         (run_best_dju_model_like_r, choose_best_hdd_cdd_like_r)
    │       └── metrics.py       (regression_metrics)
    └── mean_model.py        (run_mean_model_like_r)
```

### Boucle Principale (run_building_and_persist)

```
Pour chaque PDL (point de livraison):
    Pour chaque FLUID (gaz, elec, ...):
        │
        ├── 1. Filtrer factures pour ce PDL + FLUID
        ├── 2. Construire model_table (factures + DJU + usage)
        ├── 3. Split train / test
        ├── 4. train_like_r()
        │       ├── Si NO_DATA        → skip
        │       ├── Si TOO_FEW (n<6)  → mean_model
        │       └── Si OK             → postprocess + dju_model
        ├── 5. Collecter predictions + model_coefficients + outliers
        └── 6. Append aux résultats globaux

Persister résultats ADLS
Retourner JSON
```

---

## 📦 Requirements

```txt
# Core Data
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0

# Azure
azure-functions>=1.17.0
azure-storage-file-datalake>=12.14.0
azure-identity>=1.15.0

# Statistical / Modeling
statsmodels>=0.14.0
scipy>=1.11.0

# Date utilities
python-dateutil>=2.8.0
```

---

## ⚙️ Configuration

### Variables d'Environnement

| Variable | Description |
|----------|-------------|
| `ADLS_ACCOUNT_NAME` | Nom du compte Azure Data Lake Storage |
| `ADLS_ACCOUNT_KEY` | Clé d'accès au compte ADLS |
| `ADLS_CONTAINER_NAME` | Nom du container ADLS |

### Sources de Données ADLS (Silver)

| Chemin | Description |
|--------|-------------|
| `silver/building/building.parquet` | Infos bâtiments (station météo) |
| `silver/deliverypoint/deliverypoint.parquet` | Points de livraison |
| `silver/invoice/invoice.parquet` | Factures énergétiques |
| `silver/degreedays/degreedays_monthly.parquet` | DJU mensuels |
| `silver/usage_data/usage_data.parquet` | Facteurs d'usage |

---

## 🚀 Usage

### Appel Direct (Python)

```python
from datetime import date
from algo_prediction.algo_services.run_algo_services import run_building_and_persist

result = run_building_and_persist(
    building_id="BUILDING_001",
    start_ref=date(2022, 1, 1),
    end_ref=date(2024, 12, 31),
    start_pred=date(2025, 1, 1),
    end_pred=date(2025, 12, 31),
)
```

### Appel API (HTTP)

```bash
curl -X POST "https://<function-app>.azurewebsites.net/api/predict" \
  -H "Content-Type: application/json" \
  -H "x-functions-key: <YOUR_KEY>" \
  -d '{
    "building_id": "BUILDING_001",
    "start_ref": "2022-01-01",
    "end_ref": "2024-12-31",
    "start_pred": "2025-01-01",
    "end_pred": "2025-12-31"
  }'
```

### Réponse

```json
{
  "id_building_primaire": "BUILDING_001",
  "run_id": "a1b2c3d4-...",
  "created_at": "2025-02-11T10:30:00+00:00",
  "results": [
    {
      "deliverypoint_id_primaire": "PDL_001",
      "fluid": "GAZ",
      "month_str": "2025-01",
      "real_consumption": null,
      "predictive_consumption": 12500.5,
      "confidence_lower95": 10200.3,
      "confidence_upper95": 14800.7
    }
  ],
  "models": [
    {
      "deliverypoint_id_primaire": "PDL_001",
      "fluid": "GAZ",
      "chosen_hdd": "hdd18",
      "chosen_cdd": null,
      "b_coefficient": 1500.2,
      "a_hdd": 45.3,
      "adjR2": 0.92
    }
  ],
  "outliers_details": [...],
  "outliers_notes": [...]
}
```

---

## 🔗 Correspondance R / Python

| Étape | Fonction R | Fonction Python |
|-------|------------|-----------------|
| Pipeline principal | `predictive_consumption_modelisation()` | `run_building_and_persist()` |
| Imputation missing | `ranking_method()` | `ranking_method_like_r()` |
| Détection outliers | `ts_anomaly_detection()` | `ts_anomaly_detection_like_r()` |
| Sélection DJU | `which.max(accuracy_dju_hdd)` | `choose_best_hdd_cdd_like_r()` |
| Régression | `lm()` | `np.linalg.lstsq()` |
| Métriques | `forecast::accuracy()` | `regression_metrics()` |
| Quantiles IQR | `quantile(type=7)` | `_quantile_type7()` |
| Seasonal adjustment | `forecast::mstl() + seasadj()` | `STL() + seasonal` |
| Smoothing | `supsmu()` | `lowess()` |

### Note sur l'Alignement

L'implémentation Python reproduit fidèlement ~95% du comportement R. La seule différence notable concerne la fonction de lissage :

- **R** : `supsmu()` (Friedman's Super Smoother avec cross-validation)
- **Python** : `lowess()` avec span fixe optimisé (0.25 pour n<40)

Cette différence peut occasionnellement produire des variations mineures dans la détection d'outliers pour les cas limites.
