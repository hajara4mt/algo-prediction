# algo_prediction/modeling/imputation.py - V2 (Match R's ranking_method)
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Pour Kalman StructTS
try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    HAS_UNOBSERVED = True
except ImportError:
    HAS_UNOBSERVED = False

# Pour STL (na.interp)
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except ImportError:
    HAS_STL = False


# ==============================================================================
# 1. LINEAR INTERPOLATION (R's approx with rule=2)
# ==============================================================================
def interpolation_missing_linear(x: pd.Series) -> pd.Series:
    """
    R's interpolation_missing(x, option="linear")
    
    Équivalent de: approx(index, x[index], all_index, rule=2)$y
    - Interpolation linéaire entre points connus
    - rule=2 = extrapolation constante aux bords (répète la dernière valeur connue)
    """
    s = pd.to_numeric(x, errors="coerce").astype(float)
    
    if not s.isna().any():
        return s
    
    n_valid = s.notna().sum()
    if n_valid < 2:
        # Pas assez de points pour interpoler
        return s.fillna(s.mean()) if n_valid > 0 else s
    
    # Interpolation linéaire (comme R's approx)
    result = s.interpolate(method="linear")
    
    # Extrapolation constante aux bords (comme R's rule=2)
    # - Forward fill pour le début (si NA au début)
    # - Backward fill pour la fin (si NA à la fin)
    result = result.bfill().ffill()
    
    return result


# ==============================================================================
# 2. KALMAN SMOOTHING avec StructTS (R's na_Kalman_Smooth)
# ==============================================================================
# ==============================================================================
# 2. KALMAN SMOOTHING avec StructTS (R's na_Kalman_Smooth) - VERSION CORRIGÉE
# ==============================================================================
def kalman_smooth_structts_like(x: pd.Series, period: int = 12) -> pd.Series:
    """
    R's na_Kalman_Smooth(x, model="StructTS") - VERSION 100% ALIGNÉE
    
    R's StructTS comportement:
    - Si frequency == 1: modèle "level" (random walk)
    - Si frequency > 1: modèle "BSM" (Basic Structural Model = level + trend + seasonal)
    
    Étapes R:
    1. Si premier point NA, l'interpole d'abord avec interpolation linéaire
    2. Ajuste StructTS (choisit automatiquement le modèle selon frequency)
    3. Applique KalmanSmooth pour obtenir les états lissés
    4. Impute les NA avec: états_lissés × matrice_observation (mod$Z)
    """
    s = pd.to_numeric(x, errors="coerce").astype(float)
    
    if not s.isna().any():
        return s
    
    n = len(s)
    n_valid = s.notna().sum()
    
    if n_valid < 3:
        return interpolation_missing_linear(s)
    
    if not HAS_UNOBSERVED:
        return interpolation_missing_linear(s)
    
    miss_mask = s.isna()
    s_work = s.copy()
    
    # =========================================================================
    # ÉTAPE 1: R - Si le premier point est NA, interpoler ce point uniquement
    # R fait: data[1] <- interpolation_missing(data, option = "linear")[1]
    # =========================================================================
    if pd.isna(s_work.iloc[0]):
        # R interpole toute la série puis prend juste le premier élément
        temp_interp = interpolation_missing_linear(s_work)
        s_work.iloc[0] = temp_interp.iloc[0]
    
    # =========================================================================
    # ÉTAPE 2: Remplir temporairement les NA pour permettre le fit
    # R passe les données avec NA à StructTS qui les gère en interne
    # En Python, on doit pré-remplir pour éviter les erreurs de fit
    # =========================================================================
    s_filled = interpolation_missing_linear(s_work)
    
    try:
        # =====================================================================
        # ÉTAPE 3: Choisir le modèle selon la frequency (comme R's StructTS)
        # R: StructTS choisit automatiquement:
        #   - frequency == 1 → "level" model
        #   - frequency > 1  → "BSM" model (level + trend + seasonal)
        # =====================================================================
        
        use_seasonal = (period > 1) and (n > 2 * period)
        
        if use_seasonal:
            # R's BSM: Basic Structural Model avec saisonnalité
            # level='local linear trend' + seasonal=period
            model = UnobservedComponents(
                s_filled.values,
                level='local linear trend',
                seasonal=period,
                stochastic_seasonal=True,  # R's BSM a une saisonnalité stochastique
            )
        else:
            # R's "level" model pour données non-saisonnières
            # Ou si pas assez de données pour la saisonnalité
            model = UnobservedComponents(
                s_filled.values,
                level='local linear trend',
            )
        
        # =====================================================================
        # ÉTAPE 4: Fit avec MLE (comme R's StructTS)
        # =====================================================================
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = model.fit(disp=False, maxiter=500, method='powell')
        
        # =====================================================================
        # ÉTAPE 5: Kalman Smoothing - obtenir les états lissés
        # R: kal <- KalmanSmooth(data, mod, nit=-1)
        #    karima <- erg[missindx, ] %*% as.matrix(mod$Z)
        # 
        # En Python: smoothed_state contient les états, on prend le level (index 0)
        # Pour BSM: smoothed_state[0,:] = level
        # =====================================================================
        smoothed_level = results.smoothed_state[0, :]
        
        # =====================================================================
        # ÉTAPE 6: Remplacer uniquement les NA originaux
        # =====================================================================
        result = s.copy()
        result.loc[miss_mask] = smoothed_level[miss_mask.values]
        
        # Vérifier qu'il n'y a pas de valeurs aberrantes
        if result.isna().any() or np.isinf(result).any():
            return interpolation_missing_linear(s)
        
        return result
        
    except Exception:
        # Fallback en cas d'erreur de convergence
        return interpolation_missing_linear(s)

# ==============================================================================
# 3. SEASONAL STL LOESS (R's forecast::na.interp)
# ==============================================================================
def seasonal_stl_loess_like(x: pd.Series, period: int = 12) -> pd.Series:
    """
    R's forecast::na.interp(x)
    
    Pour données saisonnières, R fait:
    1. Remplir les NA temporairement (interpolation linéaire)
    2. STL décomposition: seasonal + trend + remainder
    3. Série désaisonnalisée = trend + remainder
    4. Remettre NA aux positions originales dans la série désaisonnalisée
    5. Interpoler la série désaisonnalisée
    6. Résultat = désaisonnalisé_interpolé + seasonal
    
    Cela préserve le pattern saisonnier tout en interpolant intelligemment.
    """
    s = pd.to_numeric(x, errors="coerce").astype(float)
    n = len(s)
    
    if not s.isna().any():
        return s
    
    n_valid = s.notna().sum()
    if n_valid < 3:
        return interpolation_missing_linear(s)
    
    # Vérifier conditions saisonnières (comme R)
    if period <= 1 or n <= 2 * period:
        return interpolation_missing_linear(s)
    
    if not HAS_STL:
        return interpolation_missing_linear(s)
    
    miss_mask = s.isna()
    
    try:
        # Étape 1: Remplir temporairement pour permettre STL
        s_filled = interpolation_missing_linear(s)
        
        # Étape 2: STL décomposition (comme R's stl avec robust=TRUE)
        stl = STL(s_filled.values, period=period, robust=True).fit()
        
        seasonal = stl.seasonal
        trend = stl.trend
        remainder = stl.resid
        
        # Étape 3: Série désaisonnalisée
        deseasoned = trend + remainder
        
        # Étape 4: Remettre NA aux positions originales
        deseasoned_series = pd.Series(deseasoned, index=s.index)
        deseasoned_series.loc[miss_mask] = np.nan
        
        # Étape 5: Interpoler la série désaisonnalisée
        deseasoned_interp = interpolation_missing_linear(deseasoned_series)
        
        # Étape 6: Résultat = désaisonnalisé interpolé + saisonnier
        result_values = deseasoned_interp.values + seasonal
        
        # Remplacer uniquement les NA originaux
        result = s.copy()
        result.loc[miss_mask] = result_values[miss_mask.values]
        
        return result
        
    except Exception:
        return interpolation_missing_linear(s)


# ==============================================================================
# 4. RANKING METHOD (R's ranking_method)
# ==============================================================================
def ranking_method_like_r(x: pd.Series, period: int = 12) -> pd.DataFrame:
    """
    R's ranking_method() - Combine 3 méthodes d'imputation
    
    R fait:
    1. linear_interpolation = interpolation_missing(x, option="linear")
    2. Kalman_StructTS = na_Kalman_Smooth(x, model="StructTS")
    3. season_stl_loess = forecast::na.interp(x) [si saisonnier]
    4. weighted_combination = rowMeans(toutes les méthodes)
    
    Retourne un DataFrame avec toutes les colonnes.
    """
    s = pd.to_numeric(x, errors="coerce").astype(float)
    n = len(s)
    
    # 1. Linear interpolation (toujours)
    linear = interpolation_missing_linear(s)
    
    # 2. Kalman StructTS (toujours)
    kalman = kalman_smooth_structts_like(s , period=period)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        "linear_interpolation": linear,
        "kalman_structts": kalman,
    })
    
    # 3. Seasonal STL (seulement si assez de données)
    # R: if (period > 1 && length(x) > 2 * period)
    if period > 1 and n > 2 * period:
        season = seasonal_stl_loess_like(s, period=period)
        df["season_stl_loess"] = season
    
    # 4. Weighted combination = moyenne des colonnes (comme R's rowMeans)
    df["weighted_combination"] = df.mean(axis=1)
    
    return df
