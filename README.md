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

### Pipeline Principal

| R (`predictive_consumption_modelisation`) | Python (`run_building_and_persist`) |
|-------------------------------------------|-------------------------------------|
| `for (fluid in fluids) { for (pdl in pdls) {...} }` | `for pdl_id in pdls: for fluid in fluids:` |
| `retrieve_invoice` → GET backend | `backend.get_invoices()` → ADLS Parquet |
| `retrieve_dju_data` → GET backend | `get_degreedays_mentuel()` → ADLS Parquet |
| `retrieve_influencing_factor` → GET backend | `build_monthly_usage_factors()` → ADLS Parquet |
| `index_ref <- which(start >= start_ref & end <= end_ref)` | `split_train_test_like_r()` |
| `train <- data.frame(retrieve_invoice[index_ref, ])` | `train, test = split_train_test_like_r(...)` |

---

### Sélection Optimal DJU

| R (lignes 1143-1158) | Python (`choose_best_hdd_cdd_like_r`) |
|----------------------|---------------------------------------|
| `accuracy_dju_hdd <- sapply(dju_ref_hdd, function(x){` | `for col in hdd_cols:` |
| `  model <- summary(lm(invoice.consumption ~ x, data=train))` | `  _, adj = r2_and_adj_r2(y, X @ beta, p)` |
| `  model$adj.r.squared` | `  hdd_scores[col] = adj` |
| `})` | |
| `names(which.max(accuracy_dju_hdd))` | `best_hdd = max(hdd_scores, key=hdd_scores.get)` |

---

### Traitement Valeurs Manquantes

| R (lignes 1160-1190) | Python (`build_y_like_r` + `ranking_method_like_r`) |
|----------------------|-----------------------------------------------------|
| `number_of_gaps <- sum(is.na(train$invoice.consumption))/nrow(train)` | `gap_ratio = df["is_missing"].mean()` |
| `if (number_of_gaps >= 0.2) { note_003 }` | `if gap_ratio >= 0.2: messages.append("note_003...")` |
| `train$is_missing <- is.na(train$invoice.consumption)` | `df["is_missing"] = y_raw.isna()` |
| `if (sum(train$is_missing) != 0) { note_004 }` | `if df["is_missing"].sum() > 0: messages.append("note_004...")` |

#### ranking_method

| R (`ranking_method`, lignes 448-475) | Python (`ranking_method_like_r`) |
|--------------------------------------|----------------------------------|
| `linear_interpolation <- interpolation_missing(x, "linear")` | `linear = interpolation_missing_linear(s)` |
| `Kalman_StructTS <- na_Kalman_Smooth(x, "StructTS")` | `kalman = kalman_smooth_structts_like(s)` |
| `if (period > 1 && length(x) > 2*period) {` | `if period > 1 and len(s) > 2 * period:` |
| `  season_stl_loess <- forecast::na.interp(x)` | `  season = seasonal_stl_loess_like(s, period)` |
| `}` | |
| `weighted_combination <- rowMeans(combination)` | `df["weighted_combination"] = df.mean(axis=1)` |

#### Refit DJU sur Missing

| R (lignes 1182-1183) | Python (`_predict_dju_fitted`) |
|----------------------|--------------------------------|
| `fit <- lm(consumption_imputation ~ HDD + CDD, data=train)` | `beta = np.linalg.lstsq(X_fit, y_fit)` |
| `train$consumption_imputation[is_missing] <- fit$fitted.values[is_missing]` | `df.loc[is_missing, "consumption_imputation"] = fitted[is_missing]` |

---

### Détection Outliers

| R (`ts_anomaly_detection`, lignes 382-434) | Python (`ts_anomaly_detection_like_r`) |
|--------------------------------------------|----------------------------------------|
| `n <- length(x)` | `n = len(x)` |
| `freq <- frequency(x)` | `period = 12` |
| `if (nmiss > 0) { xx <- forecast::na.interp(x) }` | `xx = _na_interp_ts_like(x, period)` |
| `if (freq > 1 && n > 2*freq) {` | `if period > 1 and n > 2 * period:` |
| `  fit <- forecast::mstl(xx, robust=TRUE)` | `  stl = STL(xx, period, robust=True).fit()` |
| `  strength <- 1 - var(rem)/var(detrend)` | `  strength = 1 - var(rem) / var(detrend)` |
| `  if (strength >= 0.6) { xx <- seasadj(fit) }` | `  if strength >= 0.6: xx = xx - seasonal` |
| `}` | |
| `mod <- supsmu(tt, xx)` | `smooth = lowess(xx, tt, frac=0.25, it=0)` |
| `resid <- xx - mod$y` | `resid = xx - smooth` |
| `resid.q <- quantile(resid, c(0.25, 0.75))` | `q1 = _quantile_type7(resid, 0.25)` |
| `iqr <- diff(resid.q)` | `iqr = q3 - q1` |
| `limits <- resid.q + thres * iqr * c(-1, 1)` | `low = q1 - thres * iqr` ; `high = q3 + thres * iqr` |
| `outliers <- which(resid < limits[1] \| resid > limits[2])` | `out_mask = (resid < low) \| (resid > high)` |
| `if (iterate > 1) { tsoutliers(x, iterate=1) }` | `for pass_num in range(1, iterate + 1):` |

---

### Correction Outliers

| R (lignes 1204-1226) | Python (`build_y_like_r`) |
|----------------------|---------------------------|
| `ts_data <- ts(train$consumption_imputation, frequency=12)` | `res = ts_anomaly_detection_like_r(df["consumption_imputation"])` |
| `anomaly_detection <- ts_anomaly_detection(ts_data, thres=3)` | |
| `train$is_anomaly <- is.na(anomaly_detection$x)` | `df["is_anomaly"] = res.outlier_mask` |
| `if (sum(train$is_anomaly) != 0) { note_005 }` | `if out_mask.sum() > 0: messages.append("note_005...")` |
| `ts_data <- ts(anomaly_detection$x, frequency=12)` | `base = df["consumption_imputation"].copy()` |
| `# (x avec NA aux outliers)` | `base.loc[out_mask] = np.nan` |
| `missing_imputation <- ranking_method(ts_data, period=12)` | `corr = ranking_method_like_r(base, period=12)` |
| `train$consumption_correction <- missing_imputation$weighted_combination` | `df["consumption_correction"] = corr["weighted_combination"]` |
| `fit <- lm(consumption_correction ~ HDD + CDD, data=train)` | `fitted = _predict_dju_fitted(df, "consumption_correction", ~is_anomaly)` |
| `train$consumption_correction[is_anomaly] <- fit$fitted.values[is_anomaly]` | `df.loc[is_anomaly, "consumption_correction"] = fitted[is_anomaly]` |

---

### Règle des Zéros

| R (lignes 1235-1258) | Python (`build_y_like_r`) |
|----------------------|---------------------------|
| `train0 <- train[which(train$consumption_imputation != 0),]` | `df_wo0 = df[df["consumption_imputation"] != 0]` |
| `accuracy_ref_invoice0 <- sapply(ref_invoice, ...)` | `s_wo0 = _score_adj_r2(df_wo0, "consumption_imputation")` |
| `accuracy_ref_invoice <- sapply(ref_invoice, ...)` | `s_with0 = _score_adj_r2(df, "consumption_imputation")` |
| `if (accuracy_ref_invoice0[1] >= accuracy_ref_invoice[1]) {` | `if s_wo0 >= s_with0:` |
| `  train <- train0` | `  df = df_wo0` |
| `  note_006: "WITHOUT ZEROS selected"` | `  messages.append("note_006...")` |
| `} else { note_007: "WITH CORRECTED ZEROS selected" }` | `else: messages.append("note_007...")` |

---

### Sélection Best Y

| R (lignes 1261-1268) | Python (`build_y_like_r`) |
|----------------------|---------------------------|
| `ref_invoice <- c("consumption_imputation", "consumption_correction")` | `s_imp = _score_adj_r2(df, "consumption_imputation")` |
| `accuracy_ref_invoice <- sapply(ref_invoice, function(x){` | `s_cor = _score_adj_r2(df, "consumption_correction")` |
| `  model <- summary(lm(x ~ HDD + CDD, data=train))` | |
| `  model$adj.r.squared` | |
| `})` | |
| `names(which.max(accuracy_ref_invoice))` | `best_y = "imputation" if s_imp >= s_cor else "correction"` |
| `note_008: "xxx was selected as the best outcome Y"` | `messages.append("note_008: {best_y} selected...")` |

---

### Modèle Final & Prédictions

| R (lignes 1286-1302) | Python (`run_best_dju_model_like_r`) |
|----------------------|--------------------------------------|
| `groupvars <- c(optimal_dju_name, name_influencing_factor)` | `features = [best_hdd, best_cdd] + influencing_cols` |
| `fit <- lm(best_Y ~ groupvars, data=train)` | `beta = np.linalg.lstsq(X_train, y_train)` |
| `model_coefficients <- list(` | `model_coefficients = {` |
| `  a_coefficient = fit$coefficients[-1],` | `  "a_coefficient.hdd": beta[1],` |
| `  b_coefficient = fit$coefficients[1]` | `  "b_coefficient": beta[0]` |
| `)` | `}` |
| `accuracy <- forecast::accuracy(y_true, y_pred)` | `metrics = regression_metrics(y_true, y_pred)` |
| `R2 <- summary(fit)$r.squared` | `r2, adj_r2 = r2_and_adj_r2(y, yhat, p)` |
| `pred <- predict(fit, test, interval="confidence")` | `y_pred = X_test @ beta` |
| `confidence_lower95 = pred$lwr` | `ci = t_crit * se` → `lower = y_pred - ci` |
| `confidence_upper95 = pred$upr` | `upper = y_pred + ci` |

---

### Messages / Notes

| Code R | Code Python | Description |
|--------|-------------|-------------|
| `note_000` | `TrainStatus.NO_REFERENCE_DATA` | Aucune donnée de référence |
| `note_001` | `TrainStatus.TOO_FEW_OBSERVATIONS` | Moins de 6 observations → modèle moyenne |
| `note_003` | `"note_003: MISSING > 20%"` | Plus de 20% de valeurs manquantes |
| `note_004` | `"note_004: MISSING data occurred"` | Présence de valeurs manquantes |
| `note_005` | `"note_005: ANOMALIES data occurred"` | Outliers détectés |
| `note_006` | `"note_006: WITHOUT ZEROS selected"` | Données sans zéros sélectionnées |
| `note_007` | `"note_007: WITH CORRECTED ZEROS"` | Données avec zéros corrigés |
| `note_008` | `"note_008: {Y} selected as best Y"` | Meilleur Y sélectionné |
| `note_009` | `"debug_postprocess_dju: best_hdd=..."` | Meilleur DJU sélectionné |

---

### Note sur l'Alignement

L'implémentation Python reproduit fidèlement **~95%** du comportement R. La seule différence notable concerne la fonction de lissage dans `ts_anomaly_detection` :

| Aspect | R | Python |
|--------|---|--------|
| Fonction | `supsmu()` | `lowess()` |
| Span | Cross-validation automatique | Fixe: 0.25 (n<40), 0.20 (n<100), 0.15 (n≥100) |
| Itérations robustes | Non (supsmu n'en a pas) | `it=0` (désactivées pour matcher R) |

Cette différence peut occasionnellement produire des variations mineures dans la détection d'outliers pour les valeurs proches des bornes IQR.
