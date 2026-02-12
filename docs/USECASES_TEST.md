# Usecases de Test — Algo Prediction

## Objectif

Ce document décrit les **107 cas de test** pour valider le code Python de prédiction de consommation énergétique. L'objectif est la **fidélité au comportement R** : le Python doit reproduire exactement le même comportement que l'algorithme R `predictive_consumption_modelisation`, y compris sur les cas limites non gérés.

---

## Légende

### Statuts

| Statut | Signification |
|--------|---------------|
| ✅ | Pertinent — comportement clairement défini |
| ⚠️ | À documenter — comportement existe mais pas de garde-fou (fidélité R) |

### Priorités

| Priorité | Signification | Critère |
|----------|---------------|---------|
| **P0** | Critique | Bloque le pipeline ou fausse les résultats |
| **P1** | Important | Comportement à valider en priorité |
| **P2** | Nice to have | Edge case rare en production |

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

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 1 | Bâtiment sans aucun delivery point | P1 | ✅ | Boucle PDL vide → résultat vide, pas d'erreur | `run_algo_services.py` |
| 2 | Bâtiment avec un seul PDL, un seul fluide | P0 | ✅ | Cas nominal minimal, traitement complet | `run_algo_services.py` |
| 3 | Bâtiment avec plusieurs PDL, plusieurs fluides | P0 | ✅ | Chaque combinaison PDL×fluid traitée indépendamment | `run_algo_services.py` |
| 4 | Station météo inexistante dans fichier DJU | P1 | ✅ | DJU vide → `note: DJU table is empty` → fallback mean model | `dju.py` |
| 5 | Station météo change en cours de période | P2 | ⚠️ | **Non géré en R** — une seule station utilisée | `adls_silver.py` |
| 6 | Bâtiment inexistant (building_id absent) | P1 | ✅ | Erreur ou SiteInfo vide selon implémentation backend | `adls_silver.py` |
| 7 | Champs obligatoires null (station météo null) | P0 | ✅ | `raise ValueError("Pas de weather_station")` | `run_algo_services.py:47` |
| 8 | PDL actifs vs inactifs dans la période | P1 | ✅ | Seuls les PDL avec factures dans la période ref sont traités | `run_algo_services.py` |

---

## 1.2 Factures

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 9 | Aucune facture sur période ref | P0 | ✅ | `note_000` → `NO_REFERENCE_DATA` | `decision.py` |
| 10 | Factures toutes à consommation nulle | P1 | ✅ | `note_000` → `NO_REFERENCE_DATA` (tous zéros = pas de données) | `decision.py` |
| 11 | Consommations négatives | P2 | ⚠️ | **Non filtré en R** — traitées normalement, peuvent être outliers via IQR | `invoices.py`, `outliers.py` |
| 12 | Factures qui se chevauchent | P1 | ✅ | Prorata journalier répartit la conso sur chaque jour | `invoices.py:normalize_invoices_to_monthly()` |
| 13 | Factures avec trous (jan-mars, puis juillet) | P0 | ✅ | `note_004` + imputation par ranking_method | `postprocess.py` |
| 14 | Une seule facture sur toute la période | P0 | ✅ | n=1 < 6 → `TOO_FEW_OBSERVATIONS` → modèle moyenne | `decision.py` |
| 15 | Exactement 5 observations | P0 | ✅ | n=5 < 6 → `TOO_FEW_OBSERVATIONS` → modèle moyenne | `decision.py` |
| 16 | Exactement 6 observations | P0 | ✅ | n=6 ≥ 6 → `OK_ANNUAL_REFERENCE` → modèle DJU | `decision.py` |
| 17 | Dates mal formées (start > end) | P1 | ⚠️ | **Non validé en R** — durée négative, prorata faussé silencieusement | `invoices.py:108` |
| 18 | Factures dupliquées (même PDL, période, montant) | P1 | ✅ | `dedup_invoices_like_r()` garde la première occurrence | `invoices.py:54-57` |
| 19 | Consommation extrêmement élevée (outlier 10x) | P1 | ✅ | Détecté par IQR → `note_005` | `outliers.py` |
| 20 | Missing exactement 19.9% | P2 | ✅ | Pas de `note_003` (seuil non atteint) | `postprocess.py:103` |
| 21 | Missing exactement 20% | P1 | ✅ | `note_003: MISSING > 20%` (seuil atteint) | `postprocess.py:103` |
| 22 | Valeur NaN explicite dans colonne value | P1 | ✅ | Traitée comme missing → imputation | `postprocess.py` |
| 23 | conso=0 ET durée=0 (division par zéro prorata) | P0 | ⚠️ | **Non protégé en R** — division par zéro → NaN/Inf propagé | `invoices.py:113` |

---

## 1.3 Valeur Sentinelle 9999

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 24 | Une facture à 9999 | P0 | ✅ | Convertie en NA → imputée | `invoices.py:28` |
| 25 | Plusieurs factures à 9999 | P1 | ✅ | Toutes converties en NA → seuil 20% recalculé | `invoices.py` + `postprocess.py` |
| 26 | Toutes factures à 9999 | P1 | ✅ | Toutes NA → `note_000` (tout missing) | `decision.py` |
| 27 | 9999.0 (float) vs 9999 (int) | P2 | ✅ | Comparaison Python `== 9999` gère les deux | `invoices.py` |
| 28 | 9998 ou 10000 (test négatif) | P2 | ✅ | **Ne doit PAS** être filtré (seul 9999 est sentinelle) | `invoices.py` |

---

## 1.4 DJU

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 29 | DJU tous à zéro (été pur) | P1 | ✅ | Variance nulle → adj R² = -inf → DJU non sélectionné | `dju_model.py` |
| 30 | DJU manquants sur partie de la ref | P1 | ✅ | Lignes exclues du fit (mask sur NA) | `dju_model.py:284-286` |
| 31 | DJU manquants sur toute la prédiction | P1 | ✅ | `note: some months have missing predictors → NA` | `dju_model.py:322-326` |
| 32 | DJU avec valeurs négatives | P2 | ⚠️ | **Non filtré en R** — traités normalement dans régression | `dju_model.py` |
| 33 | Uniquement HDD ou uniquement CDD | P1 | ✅ | `best_hdd` ou `best_cdd` = None, régression sur l'autre | `dju_model.py:170-171` |
| 34 | DJU identiques tous les mois (variance=0) | P1 | ✅ | adj R² = -inf → colonne ignorée | `dju_model.py` |
| 35 | DJU partiels (certains seuils manquants) | P1 | ✅ | Seuls les seuils présents sont testés pour best HDD/CDD | `dju_model.py:167-171` |

---

## 1.5 Facteurs d'Usage

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 36 | Aucun facteur d'usage | P1 | ✅ | `note_012: ALL INFLUENCING FACTOR NOT FOUND` → régression DJU seul | `model_table.py:175` |
| 37 | Facteurs partiellement renseignés | P1 | ✅ | Interpolation linéaire sur chaque colonne | `model_table.py:178-181` |
| 38 | Facteurs tous identiques (variance=0) | P1 | ✅ | Facteurs constants retirés silencieusement + `note_012` si tous constants (pas de note_011 par facteur) | `usage_data.py:79-91` |
| 39 | Facteurs aberrants (occupation 500%) | P2 | ⚠️ | **Non filtré en R** — point de levier dans régression | `dju_model.py` |
| 40 | Facteurs absents sur période prédiction | P1 | ✅ | `note: test missing required columns` → return None → fallback mean | `dju_model.py:309-312` |

---

# COUCHE 2 : PÉRIODES & PARAMÈTRES

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 41 | Période ref dans le futur | P1 | ✅ | train.empty → `note_000` | `split_train_test_like_r()` |
| 42 | Période pred dans le passé | P2 | ✅ | Cas valide — permet comparaison avec réel | - |
| 43 | Ref et pred se chevauchent | P1 | ⚠️ | **Non vérifié en R** — data leakage silencieux, R² gonflé | `split_train_test_like_r()` |
| 44 | Période ref très courte (3 mois) | P0 | ✅ | n=3 < 6 → `TOO_FEW_OBSERVATIONS` | `decision.py` |
| 45 | Période ref très longue (10 ans) | P2 | ✅ | Test de performance — traitement normal | - |
| 46 | Prédiction d'un seul mois | P1 | ✅ | Cas valide minimal | - |
| 47 | start_ref > end_ref | P1 | ⚠️ | **Non validé en R** — comportement indéfini | `model_table.py` |
| 48 | Dates format inattendu / timezone | P2 | ⚠️ | `pd.to_datetime(..., errors="coerce")` → NaT si invalide | - |

---

# COUCHE 3 : PREPROCESSING

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 49 | Tous mois manquants après agrégation | P0 | ✅ | `debug_model_table: months_with_value=0` → `note_000` | `model_table.py:186` |
| 50 | Test set vide | P1 | ✅ | `note: test is empty for given prediction period` | `model_table.py:262` |
| 51 | Train set vide | P0 | ✅ | `note: train is empty for given reference period` → `note_000` | `model_table.py:264-265` |
| 52 | Colonnes entièrement NaN | P1 | ✅ | `mask_train.sum() < 6` → return None → fallback mean | `dju_model.py:288-290` |
| 53 | Champs manquants entre lignes factures | P2 | ✅ | `.get()` avec valeurs par défaut | `invoices.py:196-211` |
| 54 | Facture couvrant 3+ mois | P1 | ✅ | Prorata journalier sur tous les mois couverts | `invoices.py:102-138` |
| 55 | Facture d'un seul jour (duration=1) | P2 | ✅ | Pas multi-mois → gardée telle quelle | `invoices.py:91` |
| 56 | Doublons avec consommations différentes | P1 | ✅ | `keep="first"` → première valeur gardée | `invoices.py:54-57` |
| 57 | Facture commence le 15 du mois | P1 | ✅ | Prorata répartit sur les jours effectifs | `invoices.py:102-138` |

---

# COUCHE 4 : IMPUTATION

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 58 | Toutes valeurs manquantes | P0 | ✅ | `notna().sum() < 3` → série retournée inchangée | `imputation.py:21` |
| 59 | Une seule valeur non manquante | P1 | ✅ | `notna().sum() < 3` → série retournée inchangée | `imputation.py:21` |
| 60 | 3 méthodes divergent fortement | P2 | ✅ | Moyenne pondérée = `df.mean(axis=1)` | `imputation.py:115` |
| 61 | Série trop courte (n < 24) | P1 | ✅ | `n <= 2 * period` → fallback interpolation linéaire | `imputation.py:70-72` |
| 62 | Interpolation avec < 3 points non-NA | P1 | ✅ | Retourne série inchangée | `imputation.py:21` |
| 63 | 3 méthodes divergent → moyenne aberrante | P2 | ✅ | Comportement attendu de la moyenne | `imputation.py:115` |

---

# COUCHE 5 : OUTLIERS

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 64 | Aucun outlier détecté | P1 | ✅ | Pas de `note_005`, `consumption_correction = raw` | `postprocess.py:178-180` |
| 65 | Tous points = outliers | P2 | ✅ | Tous détectés → correction par ranking_method | `outliers.py` |
| 66 | Outlier en première position | P1 | ✅ | Détecté normalement par lowess + IQR | `outliers.py` |
| 67 | Outlier en dernière position | P1 | ✅ | Détecté normalement par lowess + IQR | `outliers.py` |
| 68 | Outliers consécutifs (3 mois) | P1 | ✅ | Chacun détecté individuellement | `outliers.py` |
| 69 | Valeur exactement sur borne IQR | P2 | ✅ | Test `<` et `>` (pas `<=` `>=`) → sur la borne = pas outlier | `outliers.py:57-58` |
| 70 | Série très courte (n < 20) | P1 | ✅ | Span adapté + fallback Theil-Sen si n ≤ 12 | `outliers.py:148-162` |
| 71 | Série constante (variance=0) | P1 | ✅ | `is.constant` check → return empty outliers | `outliers.py:101-103` |
| 72 | Strength exactement = 0.6 | P2 | ✅ | `strength >= 0.6` → seasonal adjustment appliqué | `outliers.py:124` |
| 73 | 2ème passe trouve plus d'outliers | P2 | ✅ | `iterate=2` dans le code | `outliers.py` |
| 74 | Imputation crée pic → détecté outlier | P2 | ✅ | Comportement attendu du pipeline séquentiel | `postprocess.py` |
| 75 | Outlier sur point déjà imputé | P2 | ✅ | Double correction possible | `postprocess.py:163-176` |
| 76 | Borne span lowess : n=39 vs n=40 | P2 | ✅ | **Logique Python** : n<40 → span=0.6 ; n≥40 → span adaptatif (0.5-0.7) | `outliers.py:155-162` |

---

# COUCHE 6 : SÉLECTION Y

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 77 | Égalité parfaite adj R² (imp vs corr) | P1 | ✅ | `s_imp >= s_cor` → **imputation gagne** en cas d'égalité | `postprocess.py:202` |
| 78 | Sans zéros → seulement 3 obs | P1 | ⚠️ | **Non vérifié en R** — régression avec 3 obs (pas de recheck seuil 6) | `postprocess.py:190-192` |
| 79 | Toutes consommations = 0 | P1 | ✅ | `df_wo0` vide → `s_wo0 = -inf` → zéros conservés | `postprocess.py:187` |
| 80 | R² raw > imputé → < 6 obs après drop NA | P2 | ⚠️ | **Non vérifié en R** — régression avec < 6 obs | `postprocess.py:124-126` |

---

# COUCHE 7 : RÉGRESSION DJU

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 81 | Tous HDD ont adj R² négatif | P1 | ✅ | `best_hdd = None` → régression sur CDD seul ou fallback mean | `dju_model.py:170` |
| 82 | R² > 0.99 (overfitting potentiel) | P2 | ⚠️ | **Non signalé en R** — stocké tel quel | `dju_model.py` |
| 83 | R² négatif | P1 | ✅ | Possible, stocké tel quel | `dju_model.py:107` |
| 84 | Matrice singulière (colonnes colinéaires) | P2 | ✅ | `np.linalg.lstsq(..., rcond=None)` gère via pseudo-inverse | `dju_model.py` |
| 85 | IC 95% plus large que prédiction | P2 | ✅ | Possible, pas de garde-fou | `dju_model.py` |
| 86 | Prédictions partiellement NA | P1 | ✅ | `note: some months have missing predictors → NA` | `dju_model.py:322-326` |
| 87 | Prédictions négatives | P1 | ⚠️ | **Pas de garde-fou en R** — modèle linéaire peut prédire < 0 | `dju_model.py` |
| 88 | IC lower95 < 0 | P2 | ⚠️ | **Pas de garde-fou en R** — peut être négatif | `dju_model.py` |
| 89 | HDD et CDD tous adj R² négatif | P0 | ✅ | `return None` → fallback mean model | `dju_model.py:256-258` |
| 90 | DJU avec NA après jointure | P1 | ✅ | Lignes exclues du fit | `dju_model.py:284-286` |
| 91 | Tous facteurs constants (sd=0) | P1 | ✅ | Facteurs retirés + `note_012`, régression DJU seul | `usage_data.py:79-91` |
| 92 | Facteur absent en pred | P1 | ✅ | `return None` → fallback mean model | `dju_model.py:309-312` |
| 93 | Plus de variables que d'observations | P2 | ✅ | `np.linalg.lstsq` pseudo-inverse (adj R² sera bizarre) | `dju_model.py` |
| 94 | DJU identiques sur toute la période | P1 | ✅ | Variance=0 → adj R² = -inf → non sélectionné | `dju_model.py` |
| 95 | Interpolation crée corrélation spurieuse | P2 | ⚠️ | **Pas de détection en R** — comportement possible | - |
| 96 | Test DJU hors range train (extrapolation) | P2 | ⚠️ | **Pas de warning en R** — extrapolation silencieuse | `dju_model.py` |

---

# COUCHE 8 : OUTPUT

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 97 | Chemin ADLS inexistant | P2 | ✅ | Azure crée automatiquement le chemin | `adls_silver.py` |
| 98 | NaN/Infinity dans JSON | P1 | ✅ | `json.dumps(..., default=str)` convertit en string | `run_algo_services.py` |
| 99 | Réponse JSON volumineuse (50+ PDL) | P2 | ✅ | Test de performance | - |
| 100 | Exécution concurrente même bâtiment | P2 | ⚠️ | **Non géré en R** — last-write-wins | `silver_results_writer.py` |

---

# COUCHE TRANSVERSALE : ROBUSTESSE

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 101 | Parquet corrompu/vide | P1 | ✅ | `pd.read_parquet()` lève exception | `adls_silver.py` |
| 102 | Schéma Parquet différent (colonne renommée) | P1 | ⚠️ | KeyError si colonne attendue manquante | - |
| 103 | Caractères spéciaux dans IDs | P2 | ✅ | UTF-8 géré par Pandas | - |
| 104 | Très gros bâtiment (20 ans, 15 PDL) | P2 | ✅ | Test de performance | - |
| 105 | ADLS vide (aucun fichier silver) | P1 | ✅ | `ResourceNotFoundError` | `adls_silver.py` |
| 106 | geographical_area = NULL | P2 | ⚠️ | Dépend logique pivot usage_data | `usage_data.py` |

---

# MULTI-PDL

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 107 | 2 PDL : un DJU, un moyenne | P0 | ✅ | Chaque PDL traité indépendamment | `run_algo_services.py` |
| 108 | Un PDL cause erreur fatale | P0 | ✅ | Avec try/except : continue aux autres PDL | `run_algo_services.py` |

---

# TRAIN/TEST EDGE CASES

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 109 | Factures existent mais hors période ref | P0 | ✅ | train.empty → `note_000` | `split_train_test_like_r()` |
| 110 | Toutes factures hors range DJU | P1 | ✅ | Colonnes DJU toutes NA → `mask_train.sum() < 6` | `dju_model.py` |
| 111 | Test vide (start_pred > end_pred) | P1 | ✅ | `ValueError` ou liste mois vide | `model_table.py:213` |

---

# MULTI-ANNÉES (Spécifique Python)

| # | Usecase | Prio | Statut | Comportement Attendu (R) | Fichier Python |
|---|---------|------|--------|--------------------------|----------------|
| 112a | Prédiction 1 année | P0 | ✅ | Cas nominal | `main_test.py` |
| 112b | Prédiction 2 années | P1 | ✅ | **Autorisé en Python** (bloqué en R par error_000) | `main_test.py` |
| 112c | Prédiction 3+ années | P2 | ✅ | Test de robustesse/performance | `main_test.py` |
| 112d | Décembre → Janvier (2 ans) | P1 | ✅ | 2 mois sur 2 années différentes | `main_test.py` |

---

# RÉSUMÉ STATISTIQUE

| Catégorie | Total | P0 | P1 | P2 | ✅ | ⚠️ |
|-----------|-------|----|----|----|----|-----|
| Bâtiment & Structure | 8 | 2 | 5 | 1 | 7 | 1 |
| Factures | 15 | 5 | 7 | 3 | 13 | 2 |
| Sentinelle 9999 | 5 | 1 | 2 | 2 | 5 | 0 |
| DJU | 7 | 0 | 6 | 1 | 6 | 1 |
| Facteurs d'usage | 5 | 0 | 4 | 1 | 4 | 1 |
| Périodes | 8 | 1 | 4 | 3 | 5 | 3 |
| Preprocessing | 9 | 2 | 5 | 2 | 9 | 0 |
| Imputation | 6 | 1 | 3 | 2 | 6 | 0 |
| Outliers | 13 | 0 | 7 | 6 | 13 | 0 |
| Sélection Y | 4 | 0 | 3 | 1 | 2 | 2 |
| Régression DJU | 16 | 1 | 9 | 6 | 11 | 5 |
| Output | 4 | 0 | 2 | 2 | 3 | 1 |
| Robustesse | 6 | 0 | 3 | 3 | 4 | 2 |
| Multi-PDL | 2 | 2 | 0 | 0 | 2 | 0 |
| Train/Test | 3 | 1 | 2 | 0 | 3 | 0 |
| Multi-années | 4 | 1 | 2 | 1 | 4 | 0 |
| **TOTAL** | **115** | **17** | **64** | **34** | **97** | **18** |

---

# NOTES IMPORTANTES

## Fidélité R = Objectif

Les cas marqués ⚠️ ne sont **pas des bugs** — ils reproduisent le comportement exact du code R qui ne valide pas ces situations. Le Python doit se comporter de la même manière.

## Note sur la Détection d'Outliers (supsmu vs lowess)

La détection d'outliers peut présenter des **écarts mineurs** entre R et Python :

| Aspect | R | Python |
|--------|---|--------|
| Algorithme smoothing | `supsmu` (Friedman's Super Smoother) | `lowess` (statsmodels) |
| Choix du span | **Automatique** (adaptatif interne) | **Manuel** : span=0.6 pour n<40, adaptatif sinon |
| Comportement | Smooth plus "plat" sur petites séries | Calibré pour matcher le comportement R |

**Conséquence** : ~5% des cas peuvent avoir une détection d'outliers légèrement différente. Le cas #76 teste la **logique interne Python** (bornes de span), pas la fidélité R.

## Cas Retirés (de la numérotation originale)

- **Ancien #16** : Unités incohérentes (MWh vs kWh) — hors scope
- **Ancien #55** : Meilleur HDD = CDD — impossible par définition
- **Ancien #109** : Changement de fluide — hors scope
- **Ancien #87** : Doublon de #78 (sans zéros < 6 obs)
- **Ancien #99** : Doublon de #77 (égalité adj R²)

## Cas Ajoutés

| Nouveau # | Description |
|-----------|-------------|
| 8 | PDL actifs/inactifs dans la période |
| 22 | NaN explicite dans colonne value |
| 23 | Division par zéro dans prorata (conso=0, durée=0) |
| 35 | DJU partiels (certains seuils manquants) |
| 76 | Bornes spans lowess n=39 vs n=40 (**logique Python, pas R**) |

## Correction Cas #38 (ex-29)

Le code Python **filtre bien** les facteurs constants (std=0) dans `usage_data.py:79-91`, mais :
- Pas de `note_011` par facteur (contrairement à ce qui était indiqué)
- Seulement `note_012` quand **tous** les facteurs sont constants
- Comportement conforme à R

## Actions Préalables

Avant d'exécuter les tests, s'assurer que :

1. **Sentinelle 9999** : Code décommenté dans `invoices.py:28`
2. **Try/except PDL** : Ajouté dans `run_algo_services.py` pour le cas #108

---

*Document mis à jour le 2025-02-12*
*Basé sur l'analyse du code R `predictive_consumption_modelisation` (code_algo.R)*
