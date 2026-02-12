# Usecases de Test — Algo Prediction

## Objectif

Ce document décrit les **104 cas de test** pour valider le code Python de prédiction de consommation énergétique. L'objectif est la **fidélité au comportement R** : le Python doit reproduire exactement le même comportement que l'algorithme R `predictive_consumption_modelisation`, y compris sur les cas limites non gérés.

---

## Légende des Statuts

| Statut | Signification |
|--------|---------------|
| ✅ | Pertinent — comportement clairement défini |
| ⚠️ | À documenter — comportement existe mais pas de garde-fou (fidélité R) |

---

## Organisation par Couche du Pipeline

```
1. ENTRÉE (Bâtiment, Factures, DJU, Facteurs)
       ↓
2. PREPROCESSING (Prorata, Agrégation, Construction table)
       ↓
3. SPLIT (Train/Test)
       ↓
4. IMPUTATION (Missing values)
       ↓
5. OUTLIERS (Détection, Correction)
       ↓
6. SÉLECTION Y (Imputation vs Correction, Zéros)
       ↓
7. RÉGRESSION (Sélection DJU, Modèle, Métriques)
       ↓
8. OUTPUT (Prédictions, Écriture ADLS)
```

---

# COUCHE 1 : DONNÉES D'ENTRÉE

## 1.1 Bâtiment & Structure

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 1 | Bâtiment sans aucun delivery point | ✅ | Boucle PDL vide → résultat vide, pas d'erreur | `run_algo_services.py` |
| 2 | Bâtiment avec un seul PDL, un seul fluide | ✅ | Cas nominal minimal, traitement complet | `run_algo_services.py` |
| 3 | Bâtiment avec plusieurs PDL, plusieurs fluides | ✅ | Chaque combinaison PDL×fluid traitée indépendamment | `run_algo_services.py` |
| 4 | Station météo inexistante dans fichier DJU | ✅ | DJU vide → `note: DJU table is empty` → fallback mean model | `dju.py` |
| 5 | Station météo change en cours de période | ⚠️ | **Non géré en R** — une seule station utilisée | `adls_silver.py` |
| 6 | Bâtiment inexistant (building_id absent) | ✅ | Erreur ou SiteInfo vide selon implémentation backend | `adls_silver.py` |
| 7 | Champs obligatoires null (station météo null) | ✅ | `raise ValueError("Pas de weather_station")` | `run_algo_services.py:47` |

---

## 1.2 Factures

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 8 | Aucune facture sur période ref | ✅ | `note_000` → `NO_REFERENCE_DATA` | `decision.py` |
| 9 | Factures toutes à consommation nulle | ✅ | `note_000` → `NO_REFERENCE_DATA` (tous zéros = pas de données) | `decision.py` |
| 10 | Consommations négatives | ⚠️ | **Non filtré en R** — traitées normalement, peuvent être outliers via IQR | `invoices.py`, `outliers.py` |
| 11 | Factures qui se chevauchent | ✅ | Prorata journalier répartit la conso sur chaque jour | `invoices.py:normalize_invoices_to_monthly()` |
| 12 | Factures avec trous (jan-mars, puis juillet) | ✅ | `note_004` + imputation par ranking_method | `postprocess.py` |
| 13 | Une seule facture sur toute la période | ✅ | n=1 < 6 → `TOO_FEW_OBSERVATIONS` → modèle moyenne | `decision.py` |
| 14 | Exactement 5 observations | ✅ | n=5 < 6 → `TOO_FEW_OBSERVATIONS` → modèle moyenne | `decision.py` |
| 15 | Exactement 6 observations | ✅ | n=6 ≥ 6 → `OK_ANNUAL_REFERENCE` → modèle DJU | `decision.py` |
| 17 | Dates mal formées (start > end) | ⚠️ | **Non validé en R** — durée négative, prorata faussé silencieusement | `invoices.py:108` |
| 18 | Factures dupliquées (même PDL, période, montant) | ✅ | `dedup_invoices_like_r()` garde la première occurrence | `invoices.py:54-57` |
| 19 | Consommation extrêmement élevée (outlier 10x) | ✅ | Détecté par IQR → `note_005` | `outliers.py` |
| 20a | Missing exactement 19.9% | ✅ | Pas de `note_003` (seuil non atteint) | `postprocess.py:103` |
| 20b | Missing exactement 20% | ✅ | `note_003: MISSING > 20%` (seuil atteint) | `postprocess.py:103` |

---

## 1.3 Valeur Sentinelle 9999

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 74a | Une facture à 9999 | ✅ | Convertie en NA → imputée | `invoices.py:28` |
| 74b | Plusieurs factures à 9999 | ✅ | Toutes converties en NA → seuil 20% recalculé | `invoices.py` + `postprocess.py` |
| 74c | Toutes factures à 9999 | ✅ | Toutes NA → `note_000` (tout missing) | `decision.py` |
| 74d | 9999.0 (float) vs 9999 (int) | ✅ | Comparaison Python `== 9999` gère les deux | `invoices.py` |
| 74e | 9998 ou 10000 (test négatif) | ✅ | **Ne doit PAS** être filtré (seul 9999 est sentinelle) | `invoices.py` |

---

## 1.4 DJU

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 21 | DJU tous à zéro (été pur) | ✅ | Variance nulle → adj R² = -inf → DJU non sélectionné | `dju_model.py` |
| 22 | DJU manquants sur partie de la ref | ✅ | Lignes exclues du fit (mask sur NA) | `dju_model.py:284-286` |
| 23 | DJU manquants sur toute la prédiction | ✅ | `note: some months have missing predictors → NA` | `dju_model.py:322-326` |
| 24 | DJU avec valeurs négatives | ⚠️ | **Non filtré en R** — traités normalement dans régression | `dju_model.py` |
| 25 | Uniquement HDD ou uniquement CDD | ✅ | `best_hdd` ou `best_cdd` = None, régression sur l'autre | `dju_model.py:170-171` |
| 26 | DJU identiques tous les mois (variance=0) | ✅ | adj R² = -inf → colonne ignorée | `dju_model.py` |

---

## 1.5 Facteurs d'Usage

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 27 | Aucun facteur d'usage | ✅ | `note_012: ALL INFLUENCING FACTOR NOT FOUND` → régression DJU seul | `model_table.py:175` |
| 28 | Facteurs partiellement renseignés | ✅ | Interpolation linéaire sur chaque colonne | `model_table.py:178-181` |
| 29 | Facteurs tous identiques (variance=0) | ✅ | `note_011` par facteur + `note_012` si tous constants | `dju_model.py` |
| 30 | Facteurs aberrants (occupation 500%) | ⚠️ | **Non filtré en R** — point de levier dans régression | `dju_model.py` |
| 31 | Facteurs absents sur période prédiction | ✅ | `note: test missing required columns` → return None → fallback mean | `dju_model.py:309-312` |

---

# COUCHE 2 : PÉRIODES & PARAMÈTRES

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 32 | Période ref dans le futur | ✅ | train.empty → `note_000` | `split_train_test_like_r()` |
| 33 | Période pred dans le passé | ✅ | Cas valide — permet comparaison avec réel | - |
| 34 | Ref et pred se chevauchent | ⚠️ | **Non vérifié en R** — data leakage silencieux, R² gonflé | `split_train_test_like_r()` |
| 35 | Période ref très courte (3 mois) | ✅ | n=3 < 6 → `TOO_FEW_OBSERVATIONS` | `decision.py` |
| 36 | Période ref très longue (10 ans) | ✅ | Test de performance — traitement normal | - |
| 37 | Prédiction d'un seul mois | ✅ | Cas valide minimal | - |
| 38 | start_ref > end_ref | ⚠️ | **Non validé en R** — comportement indéfini | `model_table.py` |
| 39 | Dates format inattendu / timezone | ⚠️ | `pd.to_datetime(..., errors="coerce")` → NaT si invalide | - |

---

# COUCHE 3 : PREPROCESSING

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 40 | Tous mois manquants après agrégation | ✅ | `debug_model_table: months_with_value=0` → `note_000` | `model_table.py:186` |
| 41 | Test set vide | ✅ | `note: test is empty for given prediction period` | `model_table.py:262` |
| 42 | Train set vide | ✅ | `note: train is empty for given reference period` → `note_000` | `model_table.py:264-265` |
| 43 | Colonnes entièrement NaN | ✅ | `mask_train.sum() < 6` → return None → fallback mean | `dju_model.py:288-290` |
| 75 | Champs manquants entre lignes factures | ✅ | `.get()` avec valeurs par défaut | `invoices.py:196-211` |
| 76 | Facture couvrant 3+ mois | ✅ | Prorata journalier sur tous les mois couverts | `invoices.py:102-138` |
| 77 | Facture d'un seul jour (duration=1) | ✅ | Pas multi-mois → gardée telle quelle | `invoices.py:91` |
| 91 | Doublons avec consommations différentes | ✅ | `keep="first"` → première valeur gardée | `invoices.py:54-57` |
| 92 | Facture commence le 15 du mois | ✅ | Prorata répartit sur les jours effectifs | `invoices.py:102-138` |

---

# COUCHE 4 : IMPUTATION

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 44 | Toutes valeurs manquantes | ✅ | `notna().sum() < 3` → série retournée inchangée | `imputation.py:21` |
| 45 | Une seule valeur non manquante | ✅ | `notna().sum() < 3` → série retournée inchangée | `imputation.py:21` |
| 46 | 3 méthodes divergent fortement | ✅ | Moyenne pondérée = `df.mean(axis=1)` | `imputation.py:115` |
| 47 | Série trop courte (n < 24) | ✅ | `n <= 2 * period` → fallback interpolation linéaire | `imputation.py:70-72` |
| 79 | Interpolation avec < 3 points non-NA | ✅ | Retourne série inchangée | `imputation.py:21` |
| 88 | 3 méthodes divergent → moyenne aberrante | ✅ | Comportement attendu de la moyenne | `imputation.py:115` |

---

# COUCHE 5 : OUTLIERS

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 48 | Aucun outlier détecté | ✅ | Pas de `note_005`, `consumption_correction = raw` | `postprocess.py:178-180` |
| 49 | Tous points = outliers | ✅ | Tous détectés → correction par ranking_method | `outliers.py` |
| 50 | Outlier en première position | ✅ | Détecté normalement par lowess + IQR | `outliers.py` |
| 51 | Outlier en dernière position | ✅ | Détecté normalement par lowess + IQR | `outliers.py` |
| 52 | Outliers consécutifs (3 mois) | ✅ | Chacun détecté individuellement | `outliers.py` |
| 53 | Valeur exactement sur borne IQR | ✅ | Test `<` et `>` (pas `<=` `>=`) → sur la borne = pas outlier | `outliers.py:57-58` |
| 54 | Série très courte (n < 20) | ✅ | Span adapté + fallback Theil-Sen si n ≤ 12 | `outliers.py:148-162` |
| 81 | Série constante (variance=0) | ✅ | `is.constant` check → return empty outliers | `outliers.py:101-103` |
| 82 | Strength exactement = 0.6 | ✅ | `strength >= 0.6` → seasonal adjustment appliqué | `outliers.py:124` |
| 83 | 2ème passe trouve plus d'outliers | ✅ | `iterate=2` dans le code | `outliers.py` |
| 97 | Imputation crée pic → détecté outlier | ✅ | Comportement attendu du pipeline séquentiel | `postprocess.py` |
| 98 | Outlier sur point déjà imputé | ✅ | Double correction possible | `postprocess.py:163-176` |

---

# COUCHE 6 : SÉLECTION Y

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 61 | Égalité parfaite adj R² (imp vs corr) | ✅ | `s_imp >= s_cor` → **imputation gagne** en cas d'égalité | `postprocess.py:202` |
| 62 | Sans zéros → seulement 3 obs | ⚠️ | **Non vérifié en R** — régression avec 3 obs (pas de recheck seuil 6) | `postprocess.py:190-192` |
| 63 | Toutes consommations = 0 | ✅ | `df_wo0` vide → `s_wo0 = -inf` → zéros conservés | `postprocess.py:187` |
| 80 | R² raw > imputé → < 6 obs après drop NA | ⚠️ | **Non vérifié en R** — régression avec < 6 obs | `postprocess.py:124-126` |
| 87 | Sans zéros réduit train à < 6 obs | ⚠️ | **Non vérifié en R** — même comportement que #62 | `postprocess.py:190-192` |
| 99 | Égalité adj R² imp vs corr | ✅ | Imputation gagne (>=) | `postprocess.py:202` |

---

# COUCHE 7 : RÉGRESSION DJU

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 56 | Tous HDD ont adj R² négatif | ✅ | `best_hdd = None` → régression sur CDD seul ou fallback mean | `dju_model.py:170` |
| 57 | R² > 0.99 (overfitting potentiel) | ⚠️ | **Non signalé en R** — stocké tel quel | `dju_model.py` |
| 58 | R² négatif | ✅ | Possible, stocké tel quel | `dju_model.py:107` |
| 59 | Matrice singulière (colonnes colinéaires) | ✅ | `np.linalg.lstsq(..., rcond=None)` gère via pseudo-inverse | `dju_model.py` |
| 60 | IC 95% plus large que prédiction | ✅ | Possible, pas de garde-fou | `dju_model.py` |
| 84 | Prédictions partiellement NA | ✅ | `note: some months have missing predictors → NA` | `dju_model.py:322-326` |
| 85 | Prédictions négatives | ⚠️ | **Pas de garde-fou en R** — modèle linéaire peut prédire < 0 | `dju_model.py` |
| 86 | IC lower95 < 0 | ⚠️ | **Pas de garde-fou en R** — peut être négatif | `dju_model.py` |
| 93 | HDD et CDD tous adj R² négatif | ✅ | `return None` → fallback mean model | `dju_model.py:256-258` |
| 94 | DJU avec NA après jointure | ✅ | Lignes exclues du fit | `dju_model.py:284-286` |
| 95 | Tous facteurs constants (sd=0) | ✅ | `note_011`/`note_012`, régression DJU seul | `dju_model.py` |
| 96 | Facteur absent en pred | ✅ | `return None` → fallback mean model | `dju_model.py:309-312` |
| 100 | Plus de variables que d'observations | ✅ | `np.linalg.lstsq` pseudo-inverse (adj R² sera bizarre) | `dju_model.py` |
| 101 | DJU identiques sur toute la période | ✅ | Variance=0 → adj R² = -inf → non sélectionné | `dju_model.py` |
| 102 | Interpolation crée corrélation spurieuse | ⚠️ | **Pas de détection en R** — comportement possible | - |
| 105 | Test DJU hors range train (extrapolation) | ⚠️ | **Pas de warning en R** — extrapolation silencieuse | `dju_model.py` |

---

# COUCHE 8 : OUTPUT

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 64 | Chemin ADLS inexistant | ✅ | Azure crée automatiquement le chemin | `adls_silver.py` |
| 65 | NaN/Infinity dans JSON | ✅ | `json.dumps(..., default=str)` convertit en string | `run_algo_services.py` |
| 66 | Réponse JSON volumineuse (50+ PDL) | ✅ | Test de performance | - |
| 67 | Exécution concurrente même bâtiment | ⚠️ | **Non géré en R** — last-write-wins | `silver_results_writer.py` |

---

# COUCHE TRANSVERSALE : ROBUSTESSE

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 68 | Parquet corrompu/vide | ✅ | `pd.read_parquet()` lève exception | `adls_silver.py` |
| 69 | Schéma Parquet différent (colonne renommée) | ⚠️ | KeyError si colonne attendue manquante | - |
| 70 | Caractères spéciaux dans IDs | ✅ | UTF-8 géré par Pandas | - |
| 71 | Très gros bâtiment (20 ans, 15 PDL) | ✅ | Test de performance | - |
| 72 | ADLS vide (aucun fichier silver) | ✅ | `ResourceNotFoundError` | `adls_silver.py` |
| 78 | geographical_area = NULL | ⚠️ | Dépend logique pivot usage_data | `usage_data.py` |

---

# MULTI-PDL

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 89 | 2 PDL : un DJU, un moyenne | ✅ | Chaque PDL traité indépendamment | `run_algo_services.py` |
| 90 | Un PDL cause erreur fatale | ✅ | Avec try/except : continue aux autres PDL | `run_algo_services.py` |

---

# MULTI-ANNÉES (Spécifique Python)

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 73a | Prédiction 1 année | ✅ | Cas nominal | - |
| 73b | Prédiction 2 années | ✅ | **Autorisé en Python** (bloqué en R par error_000) | - |
| 73c | Prédiction 3+ années | ✅ | Test de robustesse/performance | - |
| 73d | Décembre → Janvier (2 ans) | ✅ | 2 mois sur 2 années différentes | - |

---

# TRAIN/TEST EDGE CASES

| # | Usecase | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|--------|--------------------------|----------------|
| 103 | Factures existent mais hors période ref | ✅ | train.empty → `note_000` | `split_train_test_like_r()` |
| 104 | Toutes factures hors range DJU | ✅ | Colonnes DJU toutes NA → `mask_train.sum() < 6` | `dju_model.py` |
| 106 | Test vide (start_pred > end_pred) | ✅ | `ValueError` ou liste mois vide | `model_table.py:213` |

---

# RÉSUMÉ STATISTIQUE

| Catégorie | ✅ Pertinent | ⚠️ Fidélité R (pas de garde-fou) |
|-----------|-------------|----------------------------------|
| Bâtiment & Structure | 6 | 1 |
| Factures | 12 | 2 |
| Sentinelle 9999 | 5 | 0 |
| DJU | 5 | 1 |
| Facteurs d'usage | 4 | 1 |
| Périodes | 5 | 3 |
| Preprocessing | 9 | 0 |
| Imputation | 6 | 0 |
| Outliers | 12 | 0 |
| Sélection Y | 3 | 3 |
| Régression DJU | 12 | 5 |
| Output | 3 | 1 |
| Robustesse | 5 | 1 |
| Multi-PDL | 2 | 0 |
| Multi-années | 4 | 0 |
| Train/Test | 3 | 0 |
| **TOTAL** | **84** | **18** |

---

# NOTES IMPORTANTES

## Fidélité R = Objectif

Les cas marqués ⚠️ ne sont **pas des bugs** — ils reproduisent le comportement exact du code R qui ne valide pas ces situations. Le Python doit se comporter de la même manière.

## Cas Retirés

- **#16** : Unités incohérentes (MWh vs kWh) — hors scope
- **#55** : Meilleur HDD = CDD — impossible par définition
- **#109** : Changement de fluide — hors scope

## Actions Préalables

Avant d'exécuter les tests, s'assurer que :

1. **Sentinelle 9999** : Code décommenté dans `invoices.py:28`
2. **Try/except PDL** : Ajouté dans `run_algo_services.py` pour le cas #90

---

*Document généré le 2025-02-12*
*Basé sur l'analyse du code R `predictive_consumption_modelisation` (code_algo.R)*
