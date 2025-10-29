# Copyright (c) 2025  Feklin Dmitry (FeklinDN@gmail.com)
import os
os.environ['SCIPY_ARRAY_API'] = '1'
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.signal import detrend, periodogram, find_peaks
from scipy.special import gammaln
from typing import List, Dict, Any, Tuple, Optional
import itertools

import warnings

# Trying to import statsmodels to make it an optional dependency
try:
    from statsmodels.tsa.stattools import adfuller, acf, kpss
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    # Create a "dummy" so that the code does not crash in stationarity() if statsmodels is missing
    class InterpolationWarning(Warning): pass
    print("Warning: statsmodels library not found. Some functions will have limited functionality.")
    print("Install it via 'pip install statsmodels' for full features.")

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: pywavelets library not found. Spectral analysis will be limited.")
    print("Install it via 'pip install PyWavelets' for full features.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures library not found. Structural break analysis will use a basic method.")
    print("Install it via 'pip install ruptures' for full features.")

try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    print("Warning: nolds library not found. Nonlinear/fractal analysis will be unavailable.")
    print("Install it via 'pip install nolds' for full features.")

try:
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_STL_AVAILABLE = True
except ImportError:
    STATSMODELS_STL_AVAILABLE = False
    print("Warning: STL decomposition not available. Update statsmodels for this feature.")

try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False
    print("Warning: stumpy library not found. Matrix Profile analysis will be unavailable.")
    print("Install it via 'pip install stumpy' for this feature.")

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("Warning: hmmlearn library not found. HMM segmentation will be unavailable.")
    print("Install it via 'pip install hmmlearn' for this feature.")

try:
    from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
except ImportError:
    # If sklearn isn't available, create "dummy" programs so the code doesn't crash.
    class ConvergenceWarning(Warning): pass
    class UndefinedMetricWarning(Warning): pass

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch library not found. Volatility analysis (GARCH) will be unavailable.")
    print("Install it via 'pip install arch' for this feature.")

try:
    from pyts.approximation import SymbolicAggregateApproximation
    PYTS_AVAILABLE = True
except ImportError:
    PYTS_AVAILABLE = False
    print("Warning: pyts library not found. Symbolic analysis (SAX) will be unavailable.")
    print("Install it via 'pip install pyts' for this feature.")

try:
    from PyEMD import EMD
    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False
    print("Warning: PyEMD library not found. Empirical Mode Decomposition will be unavailable.")
    print("Install it via 'pip install EMD-signal' for this feature.")

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset
    TSLEARN_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TSLEARN_AVAILABLE = False
    print("Warning: tslearn library not found or failed to import due to dependency issues.")
    print(f"         (Reason: {e})")
    print("         Time series clustering will be unavailable.")
    print("         To fix, try running: 'pip install --upgrade scipy scikit-learn tslearn'")

try:
    import sympy as sp
    from sympy.printing.pretty.pretty import pretty_print
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: sympy library not found. P-recursive and nonlinear recurrence formula analysis will be unavailable.")
    print("Install it via 'pip install sympy' for this feature.")
    # Create a dummy function if sympy is missing so the code doesn't crash
    def pretty_print(arg): print(arg)

# ===================================================================
# 1. LOCAL DYNAMICS (Pattern of Change)
# ===================================================================
def local_dynamics(seq: np.ndarray, cycle_detection_lag: int = 15) -> Dict[str, Any]:
    """
    Analyzes local transitions between elements, including high-order moments.
    """
    # A minimum of 5 points are required to calculate the jerk (diff^3) and its variance
    if len(seq) < 5:
        return {
            "pattern_type": "undetermined", "operator_type": "undetermined",
            "structure_of_changes": "too_short", "diff_variance": None,
            "acceleration_variance": None, "jerk_variance": None
        }

    # === Basic and higher derivatives ===
    diffs = np.diff(seq)                # Speed
    diff2 = np.diff(diffs)              # Acceleration
    diff3 = np.diff(diff2)              # Jerk

    # === Determining the operator type (additive/multiplicative) ===
    # Use the median for robustness to outliers
    median_val = np.median(np.abs(seq[:-1]))
    median_val = 1e-9 if median_val == 0 else median_val

    var_abs_diff = np.var(diffs)
    var_rel_diff = np.var(diffs / (seq[:-1] + 1e-9)) # Avoiding division by zero
    operator_type = "additive" if var_abs_diff < var_rel_diff else "multiplicative"

    # === Defining the structure of changes ===
    var_diffs = np.var(diffs)
    if var_diffs < 1e-9:
        structure_type = "constant" # Linear growth/decrease or constant
    else:
        # Check if there is a cycle in the differences themselves (for example, [1, 2, -1, 1, 2, -1])
        # This is a more complex pattern than the simple cyclicality of the series itself.
        if STATSMODELS_AVAILABLE and len(diffs) > cycle_detection_lag * 2:
            try:
                diff_acf = acf(diffs, nlags=cycle_detection_lag, fft=True)[1:]
                if len(diff_acf) > 0:
                    peak_lag = np.argmax(diff_acf)
                    # Threshold 0.5 is a fairly strong signal of periodicity
                    if diff_acf[peak_lag] > 0.5:
                        structure_type = f"periodic (T={peak_lag + 1})"
                    else:
                        structure_type = "stochastic_or_complex"
                else:
                    structure_type = "stochastic_or_complex"
            except Exception:
                structure_type = "stochastic_or_complex"
        else:
            structure_type = "stochastic_or_complex"

    # === Definition of a generalized pattern type ===
    # Counting how often the direction of movement changes
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    if np.all(diffs >= 0) or np.all(diffs <= 0):
        p_type = "monotonic"
    elif sign_changes / len(diffs) > 0.4: # Often fluctuates
        p_type = "oscillating"
    else: # Changes direction, but rarely
        p_type = "piecewise"

    return {
        "pattern_type": p_type,
        "operator_type": operator_type,
        "structure_of_changes": structure_type,
        "diff_variance": float(var_diffs),
        "acceleration_variance": float(np.var(diff2)),
        "jerk_variance": float(np.var(diff3))
    }

# ===================================================================
# 2. Dependency Model
# ===================================================================
def _analyze_index_dependency(seq: np.ndarray) -> Dict[str, Any]:
    """(Auxiliary) Determines the best functional form of y=f(n)."""
    n = len(seq)
    if n < 5: 
        return {"model_type": "too_short", "mse": float('inf'), "params": {}}
        
    x = np.arange(n, dtype=float)
    errors, params = {}, {}

    # Linear model
    try:
        p = np.polyfit(x, seq, 1)
        errors["linear"] = np.mean((seq - np.poly1d(p)(x)) ** 2)
        params["linear"] = {"a": p[0], "b": p[1]}
    except Exception:
        pass

    # Quadratic model
    try:
        p = np.polyfit(x, seq, 2)
        errors["quadratic"] = np.mean((seq - np.poly1d(p)(x)) ** 2)
        params["quadratic"] = {"a": p[0], "b": p[1], "c": p[2]}
    except Exception:
        pass
        
    # Cubic model
    if n > 3:
        try:
            p = np.polyfit(x, seq, 3)
            errors["cubic"] = np.mean((seq - np.poly1d(p)(x)) ** 2)
            params["cubic"] = {"a": p[0], "b": p[1], "c": p[2], "d": p[3]}
        except Exception:
            pass

    # Exponential model
    try:
        # Find the first positive element and fit on the cut
        # This allows you to handle sequences starting with 0 or negative numbers.
        positive_indices = np.where(seq > 0)[0]
        if len(positive_indices) > n / 2: # Require that at least half of the series be positive.
            start_idx = positive_indices[0]
            if n - start_idx >= 3: # Require a minimum of 3 points for a fit
                seq_slice = seq[start_idx:]
                x_slice = x[start_idx:]
                
                # Using a logarithm for linearization
                log_seq_slice = np.log(seq_slice)
                
                p = np.polyfit(x_slice, log_seq_slice, 1)
                
                # Forecast over the entire range of x
                y_pred_exp = np.exp(p[1]) * np.exp(p[0] * x)
                errors["exponential"] = np.mean((seq - y_pred_exp) ** 2)
                params["exponential"] = {"a": np.exp(p[1]), "b": p[0]}
    except Exception:
        pass

    # Harmonic model
    try:
        def harmonic_func(x_arg, A, w, phi, C): 
            return A * np.sin(w * x_arg + phi) + C
        
        fft_freqs = np.fft.fftfreq(n, d=1)
        fft_vals = np.abs(np.fft.fft(seq - np.mean(seq)))
        dominant_freq_idx = np.argmax(fft_vals[1:n//2]) + 1 if n > 2 else 1
        freq_guess = fft_freqs[dominant_freq_idx] * 2 * np.pi
        p0 = [(np.max(seq) - np.min(seq)) / 2, freq_guess, 0, np.mean(seq)]
        
        popt, _ = curve_fit(harmonic_func, x, seq, p0=p0, maxfev=10000)
        errors["harmonic"] = np.mean((seq - harmonic_func(x, *popt)) ** 2)
        params["harmonic"] = {"Amplitude": popt[0], "Frequency": popt[1], "Phase": popt[2], "Offset": popt[3]}
    except Exception:
        pass

    if not errors: 
        return {"model_type": "undetermined", "mse": float('inf'), "params": {}}
        
    best_fit = min(errors, key=errors.get)
    return {"model_type": best_fit, "mse": float(errors[best_fit]), "params": params.get(best_fit, {})}

def _analyze_value_dependency(seq: np.ndarray, max_order: int = 5) -> Dict[str, Any]:
    """(Auxiliary) Analyzes AR(p) relationships using AIC for model selection."""
    n = len(seq)
    max_order = min(max_order, n // 4)
    if n < 10 or max_order < 1: 
        return {"model_type": "autoregressive", "order": 0, "mse": float('inf'), "coeffs": []}
    
    best_aic = float('inf')
    best_order = 0
    best_coeffs = []
    best_mse = float('inf')

    for p in range(1, max_order + 1):
        y = seq[p:]
        X = np.array([np.roll(seq, i) for i in range(1, p + 1)]).T[p:]
        X = np.hstack([X, np.ones((len(y), 1))])
        
        if X.shape[0] < X.shape[1]: continue
        
        try:
            coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            if rank < X.shape[1]: continue

            k = p + 1 
            rss = np.sum(residuals**2) if residuals.size > 0 else 0
            
            if len(y) == 0: continue

            # Correctly handle the case of perfect fit (rss=0).
            # `log(0)` could cause a problem, now assign -inf,
            # which guarantees that this model will be chosen as the best.
            if rss < 1e-9:
                aic = -float('inf')
            else:
                aic = len(y) * np.log(rss / len(y)) + 2 * k
            
            if aic < best_aic:
                best_aic = aic
                best_order = p
                best_coeffs = coeffs.tolist()
                best_mse = (rss / len(y)) if len(y) > 0 else float('inf')
        except np.linalg.LinAlgError:
            continue
            
    return {
        "model_type": "autoregressive", 
        "order": best_order, 
        "mse": float(best_mse), 
        "coeffs": best_coeffs
    }

def dependency_model_analysis(seq: np.ndarray) -> Dict[str, Any]:
    """
    Defines the basic generation law, returning information about all candidate models.
    Replaced Berlekamp-Massey with a more robust exact recurrence detection
    based on the high-precision AR model results.
    """
    seq_float64 = seq.astype(np.float64)
    if not np.all(np.isfinite(seq_float64)):
        return {
            "dependency_type": "undetermined",
            "comment": "Sequence contains non-finite values."
        }

    variance = np.var(seq_float64)
    if variance < 1e-9: 
        variance = 1.0 # Avoid division by zero for constants

    idx_model = _analyze_index_dependency(seq_float64)
    val_model = _analyze_value_dependency(seq_float64)
    
    idx_mse_ratio = idx_model['mse'] / variance
    val_mse_ratio = val_model['mse'] / variance

    # --- Store information about all candidates ---
    candidates = {
        "index_dependent": {
            "type": idx_model.get('model_type', 'undetermined'),
            "params": idx_model.get('params', {}),
            "performance": {"mse_ratio": float(idx_mse_ratio)}
        },
        "value_dependent": {
            "type": "autoregressive",
            "order": val_model.get('order', 0),
            "coeffs": val_model.get('coeffs', []),
            "performance": {"mse_ratio": float(val_mse_ratio)}
        }
    }
    
    # --- Logic for choosing the best model ---
    threshold = 0.05
    idx_is_good = idx_mse_ratio < threshold
    val_is_good = val_mse_ratio < threshold

    dependency_type = "stochastic_or_complex"
    if idx_is_good and val_is_good:
        if abs(idx_mse_ratio - val_mse_ratio) < 0.01:
            dependency_type = "mixed"
        else:
            dependency_type = "index_dependent" if idx_mse_ratio < val_mse_ratio else "value_dependent"
    elif idx_is_good:
        dependency_type = "index_dependent"
    elif val_is_good:
        dependency_type = "value_dependent"

    # ===================================================================
    # BLOCK: Exact Recurrence Analysis (Replaces Berlekamp-Massey)
    # ===================================================================
    berlekamp_massey_result = {}

    is_integer_like = np.allclose(seq_float64, np.round(seq_float64))
    # A perfect fit is an MSE extremely close to zero relative to the sequence's variance.
    is_perfect_fit = val_mse_ratio < 1e-12 and val_model.get('order', 0) > 0

    # If the standard AR model finds a near-perfect fit for an integer-like sequence,
    # we treat it as an exact integer recurrence and present it clearly.
    if is_integer_like and is_perfect_fit:
        try:
            order = val_model['order']
            # AR coeffs from lstsq are [c1, c2, ..., cp, const]. We want the recurrence coeffs.
            # Round them to the nearest integer as they should be exact.
            float_coeffs = val_model['coeffs'][:-1] # Exclude the constant term
            recurrence_coeffs = [int(round(c)) for c in float_coeffs]
            
            # Check for trivial case (e.g., all zero coefficients)
            if any(recurrence_coeffs):
                terms = []
                for i, c in enumerate(recurrence_coeffs):
                    if c == 0: continue
                    
                    term = f"a(n-{i+1})"
                    if c == -1:
                        terms.append(f"- {term}")
                    elif c == 1:
                        terms.append(f"+ {term}")
                    else:
                        terms.append(f"{c:+}*{term}")

                formula_str = " ".join(terms).lstrip('+ ')

                berlekamp_massey_result = {
                    "model_type": "exact_linear_recurrence",
                    "comment": "An exact integer linear recurrence was inferred from the high-precision autoregressive model.",
                    "order": order,
                    "coeffs": recurrence_coeffs,
                    "formula_str": f"a(n) = {formula_str}"
                }
                dependency_type = "value_dependent (exact)"
        except Exception as e:
            berlekamp_massey_result = {"error": f"Exact recurrence analysis failed: {e}"}

    best_model_key = dependency_type.split(" ")[0]
    result = {
        "dependency_type": dependency_type,
        "best_model": candidates.get(best_model_key, {"type": "none"}),
        "model_candidates": candidates,
        "berlekamp_massey_analysis": berlekamp_massey_result
    }
    
    return result

# ===================================================================
# 3. Value Bounds
# ===================================================================
def value_bounds(seq: np.ndarray) -> Dict[str, Any]:
    if len(seq) == 0: return {"bounds": [None, None], "bound_width": 0, "bound_type": "empty"}
    min_val, max_val = float(np.min(seq)), float(np.max(seq))
    bound_type = "closed" if np.isfinite(min_val) and np.isfinite(max_val) else "unbounded"
    return {"bounds": [min_val, max_val], "bound_width": max_val - min_val, "bound_type": bound_type}

# ===================================================================
# 4. Cyclicity & Seasonality
# ===================================================================
def cyclicity_and_seasonality(seq: np.ndarray, max_lag: int = 40, seasonal_lags: List[int] = None) -> Dict[str, Any]:
    n = len(seq)
    max_lag = min(max_lag, n // 2)
    if not STATSMODELS_AVAILABLE or n < max_lag * 2 or max_lag <= 1: # Increased the threshold to <= 1
        return {"dominant_period": 0, "strength": 0.0, "seasonal_periods": {}}
    
    try:
        autocorr = acf(detrend(seq), nlags=max_lag, fft=True)[1:]
        
        # Checking for an empty autocorrelation array
        if len(autocorr) == 0:
            return {"dominant_period": 0, "strength": 0.0, "seasonal_periods": {}}

        # Split assignment to avoid UnboundLocalError
        dominant_period_idx = np.argmax(autocorr)
        strength = autocorr[dominant_period_idx]
        
        dominant_period = dominant_period_idx + 1 if strength >= 0.3 else 0
        strength = float(strength) if strength >= 0.3 else 0.0
        seasonal_periods = {lag: float(autocorr[lag - 1]) for lag in (seasonal_lags or []) if lag <= max_lag and lag > 0}
        
        return {"dominant_period": int(dominant_period), "strength": strength, "seasonal_periods": seasonal_periods}
    except Exception as e:
        return {"error": f"Analysis step 'cyclicity_and_seasonality' failed: {e}"}

# ===================================================================
# 5. Statistical Moments
# ===================================================================
def statistical_moments(seq: np.ndarray) -> Dict[str, Any]:
    if len(seq) < 2: return {"mean": None, "variance": None, "skewness": None, "kurtosis": None}
    return {"mean": float(np.mean(seq)), "variance": float(np.var(seq)), "skewness": float(stats.skew(seq)), "kurtosis": float(stats.kurtosis(seq))}

# ===================================================================
# 6. STATIONARITY
# ===================================================================
def stationarity(seq: np.ndarray) -> Dict[str, Any]:
    """Tests the stationarity of a series using the Dickey-Fuller and KPSS tests."""
    if not STATSMODELS_AVAILABLE or len(seq) < 20:
        return {"adf_p_value": None, "kpss_p_value": None, "conclusion": "Not enough data or statsmodels missing"}
    
    adf_p, kpss_p = None, None
    adf_stationary, kpss_stationary = None, None

    # Use the 'ct' (constant + trend) regression for greater robustness to trends.
    # This helps avoid false positive "Stationary" results for growing series.
    try:
        # For ADF, 'ct' tests for stationarity around the deterministic trend.
        adf_p = adfuller(seq, regression='ct')[1]
        adf_stationary = adf_p < 0.05
    except Exception:
        try: # Fallback to 'c' if 'ct' causes an error (e.g. on constant data)
            adf_p = adfuller(seq, regression='c')[1]
            adf_stationary = adf_p < 0.05
        except Exception:
            pass
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            # For KPSS, 'ct' tests the null hypothesis of stationarity around the trend.
            kpss_p = kpss(seq, regression='ct', nlags="auto")[1]
            kpss_stationary = kpss_p >= 0.05
    except Exception:
        try: # Rollback to 'c'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                kpss_p = kpss(seq, regression='c', nlags="auto")[1]
                kpss_stationary = kpss_p >= 0.05
        except Exception:
            pass

    conclusion = "Undetermined"
    if adf_stationary is not None and kpss_stationary is not None:
        if adf_stationary and kpss_stationary: conclusion = "Stationary"
        elif not adf_stationary and not kpss_stationary: conclusion = "Non-stationary (unit root)"
        elif not adf_stationary and kpss_stationary: conclusion = "Trend-stationary"
        else: conclusion = "Difference-stationary"

    return {"adf_p_value": float(adf_p) if adf_p is not None else None, "is_stationary_adf": adf_stationary, "kpss_p_value": float(kpss_p) if kpss_p is not None else None, "is_stationary_kpss": kpss_stationary, "conclusion": conclusion}

# ===================================================================
# 7. Autocorrelation and Memory
# ===================================================================
def autocorrelation_memory(seq: np.ndarray, max_lag: int = 40) -> Dict[str, Any]:
    n = len(seq)
    max_lag = min(max_lag, n // 2)
    if not STATSMODELS_AVAILABLE or n < max_lag * 2 or max_lag == 0: return {"memory_depth": 0, "memory_strength": 0.0, "coeffs": []}
    autocorr, confint = acf(seq, nlags=max_lag, fft=True, alpha=0.05)
    try: first_insignificant = np.where(autocorr[1:] < confint[1:, 1] - autocorr[1:])[0][0] + 1
    except IndexError: first_insignificant = max_lag
    memory_strength = np.sum(np.abs(autocorr[1:first_insignificant]))
    return {"memory_depth": int(first_insignificant), "memory_strength": float(memory_strength), "coeffs": [float(c) for c in autocorr]}

# ===================================================================
# 8. Structural Breaks
# ===================================================================
def structural_breaks(seq: np.ndarray, model: str = "Pelt", num_breaks: int = 5) -> Dict[str, Any]:
    """
    Detects points where the statistical properties of a series change.
    Added KernelCPD support for non-linear changes.
    """
    n = len(seq)
    if not RUPTURES_AVAILABLE or n < 20:
        return {"break_points": [], "comment": "Not enough data or 'ruptures' library missing"}
    
    model_map = {
        "Pelt": rpt.Pelt(model="rbf"),
        "Binseg": rpt.Binseg(model="rbf"),
        "BottomUp": rpt.BottomUp(model="rbf"),
        "KernelCPD": rpt.KernelCPD(kernel="rbf")
    }
    
    algo = model_map.get(model)
    if not algo:
        return {"break_points": [], "comment": f"Unknown model: {model}. Available: Pelt, Binseg, BottomUp, KernelCPD."}

    try:
        algo.fit(seq)
        if model in ["Pelt", "KernelCPD"]:
            penalty = np.log(n) * np.std(seq)**2 # Typical penalty for RBF/Kernel
            breaks = algo.predict(pen=penalty)
        else:
            breaks = algo.predict(n_bkps=num_breaks)
            
        breaks.pop()
        return {"break_points": [int(b) for b in breaks], "model_used": model}
    except Exception as e:
        return {"break_points": [], "comment": f"Ruptures analysis failed: {e}"}

# ===================================================================
# 9. ANOMALIES
# ===================================================================
def anomalies(seq: np.ndarray, value_k: float = 3.5, shape_k: float = 4.0) -> Dict[str, Any]:
    """Analyzes anomalies in values and shapes, adapting to monotonous series."""
    if len(seq) < 4: return {"outliers_idx": [], "shape_anomalies_idx": [], "num_anomalies": 0}
    
    # For "mostly" monotonic positive series, use a logarithm to find deviations
    # from the multiplicative trend, not the trend itself. This stabilizes variance.
    diffs_for_check = np.diff(seq)
    is_mostly_monotonic = False
    if len(diffs_for_check) > 0:
        is_mostly_monotonic = (np.sum(diffs_for_check >= 0) / len(diffs_for_check) > 0.8 or 
                               np.sum(diffs_for_check <= 0) / len(diffs_for_check) > 0.8)

    is_monotonic_positive = is_mostly_monotonic and np.min(seq) > 0
    
    data_for_value_analysis = seq
    if is_monotonic_positive and len(seq) > 10 and np.all(seq > 0): # Check that all values are positive for log
        try:
            data_for_value_analysis = np.log(seq)
        except (RuntimeWarning, FloatingPointError):
            pass # Remain with the original data if the logarithm fails

    median, mad = np.median(data_for_value_analysis), np.median(np.abs(data_for_value_analysis - np.median(data_for_value_analysis)))
    mad = 1e-9 if mad == 0 else mad
    outliers_idx = np.where(np.abs(0.6745 * (data_for_value_analysis - median) / mad) > value_k)[0]

    # The analysis of shape (sharp breaks) is always carried out on the original data,
    # but for exponential series, it's better to analyze the diff of the log.
    data_for_shape_analysis = np.diff(seq)
    if is_monotonic_positive and len(seq) > 10 and np.all(seq > 0):
        try:
            # For exponential series, log-differences are more stable
            data_for_shape_analysis = np.diff(np.log(seq))
        except (RuntimeWarning, FloatingPointError):
            pass

    second_diffs = np.diff(data_for_shape_analysis)
    if len(second_diffs) == 0:
        shape_anomalies_idx = np.array([])
    else:
        median_sd, mad_sd = np.median(second_diffs), np.median(np.abs(second_diffs - np.median(second_diffs)))
        mad_sd = 1e-9 if mad_sd == 0 else mad_sd
        shape_anomalies_idx = np.where(np.abs(0.6745 * (second_diffs - median_sd) / mad_sd) > shape_k)[0] + 1
    
    all_anomalies_idx = sorted(list(set(outliers_idx.tolist() + shape_anomalies_idx.tolist())))
    return {"outliers_idx": outliers_idx.tolist(), "shape_anomalies_idx": shape_anomalies_idx.tolist(), "all_anomalies_idx": all_anomalies_idx, "num_anomalies": len(all_anomalies_idx)}

# ===================================================================
# 10. Entropy Analysis
# ===================================================================
def entropy_analysis(seq: np.ndarray, m: int = 2, num_bins: int = 10) -> Dict[str, Any]:
    """
    Estimates the base complexity and predictability of a sequence using Shannon entropy and sample entropy.
    """
    n = len(seq)
    if n < 20:
        return {"shannon_entropy": None, "sample_entropy": None, "comment": "Sequence too short"}

    # --- Shannon Entropy ---
    # Estimates the "uncertainty" of the distribution of values.
    try:
        counts, _ = np.histogram(seq, bins=num_bins)
        # Remove zero bins to avoid log(0)
        counts = counts[counts > 0]
        shannon_entropy = stats.entropy(counts / n, base=2)
    except Exception:
        shannon_entropy = None

    # --- Sample Entropy ---
    # Estimates the likelihood of new patterns emerging. Higher values ​​= more unpredictable patterns.
    def _sampen(L, m_arg, r):
        N = len(L)
        B = 0.0
        A = 0.0
        
        # Create templates of length m and m+1
        x = np.array([L[i : i + m_arg] for i in range(N - m_arg)])
        x_plus = np.array([L[i : i + m_arg + 1] for i in range(N - m_arg - 1)])

        # Counting pairs of similar patterns
        for i in range(len(x_plus)):
            # Chebyshev distance between x[i] and all subsequent x[j]
            diffs = np.max(np.abs(x[i] - x[i+1:]), axis=1)
            B += np.sum(diffs < r)

        for i in range(len(x_plus)):
            # The same for patterns of length m+1
            diffs_plus = np.max(np.abs(x_plus[i] - x_plus[i+1:]), axis=1)
            A += np.sum(diffs_plus < r)
            
        return -np.log(A / B) if B > 0 and A > 0 else 0.0

    try:
        std_dev = np.std(seq)
        if std_dev > 1e-9:
            r_val = 0.2 * std_dev
            sample_entropy = _sampen(seq, m, r_val)
        else:
            sample_entropy = 0.0 # For a constant series, the entropy is 0
    except Exception:
        sample_entropy = None

    return {
        "shannon_entropy": float(shannon_entropy) if shannon_entropy is not None else None,
        "sample_entropy": float(sample_entropy) if sample_entropy is not None else None
    }

# ===================================================================
# 11. FLOWS AND THEIR STRUCTURE
# ===================================================================
def stream_structure_analysis(seq: np.ndarray, base_meta_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the full `dependency_model` report, including all candidates, and residual analysis for maximum accuracy.
    """
    n = len(seq)
    if n < 30:
        return {"is_interleaved": False, "num_streams": 1, "detection_reason": "Sequence too short", "sub_stream_analysis": []}
    
    # --- Evidence #1: Analysis of residuals from the BEST index-dependent model ---
    # This method is the most reliable if it works.
    dep_model_report = base_meta_report.get('dependency_model', {})
    idx_candidate = dep_model_report.get('model_candidates', {}).get('index_dependent', {})
    
    if idx_candidate and idx_candidate.get('performance', {}).get('mse_ratio', 1.0) < 0.05:
        try:
            # --- CALCULATION OF REMAINS ---
            model_type = idx_candidate.get('type')
            params = idx_candidate.get('params', {})
            x = np.arange(n, dtype=np.longdouble)
            y_pred = None
            residuals = None

            if model_type == 'exponential' and params:
                y_pred = params.get('a', 0) * np.exp(params.get('b', 0) * x)
                # For multiplicative models, analyze the ratio (residuals in a log scale)
                # Protection against division by zero and against the logarithm of a zero/negative number
                safe_y_pred = y_pred + 1e-9
                safe_seq = seq + 1e-9
                residuals = np.log(safe_seq) - np.log(safe_y_pred)

            elif model_type in ['linear', 'quadratic', 'cubic'] and params:
                # np.poly1d requires float64, so use a safe conversion
                p_coeffs_float = [float(v) for v in params.values()]
                y_pred = np.poly1d(p_coeffs_float)(x)
                # For additive models - the difference
                residuals = seq - y_pred
            # ---

            if residuals is not None and len(residuals) == n and STATSMODELS_AVAILABLE:
                autocorr_residuals = acf(residuals, nlags=min(20, n // 2), fft=True)[1:]
                peaks, _ = find_peaks(autocorr_residuals, height=0.3, distance=2)
                
                if len(peaks) > 0:
                    # Finding the strongest peak
                    peak_lag = peaks[np.argmax(autocorr_residuals[peaks])] + 1
                    
                    # Launching subflow analysis
                    sub_stream_reports = [
                        analyze_sequence_full(list(seq[i::peak_lag]), is_sub_stream=True)
                        for i in range(peak_lag)
                        if len(seq[i::peak_lag]) > 15
                    ]

                    return {
                        "is_interleaved": True, "num_streams": peak_lag,
                        "detection_reason": f"Periodic signal at lag {peak_lag} found in the residuals of a strong '{model_type}' model.",
                        "sub_stream_analysis": sub_stream_reports
                    }
        except Exception:
            # If the calculation of residuals or ACF fails, simply move on to the next piece of evidence
            pass

    # --- Evidence #2: Testing the candidate AR model (even if it didn't win) ---
    val_candidate = dep_model_report.get('model_candidates', {}).get('value_dependent', {})
    if val_candidate:
        ar_order = val_candidate.get('order', 0)
        ar_mse_ratio = val_candidate.get('performance', {}).get('mse_ratio', 1.0)

        if ar_order > 1 and ar_mse_ratio < 0.05:
            # Launching subflow analysis
            sub_stream_reports = [
                analyze_sequence_full(list(seq[i::ar_order]), is_sub_stream=True)
                for i in range(ar_order)
                if len(seq[i::ar_order]) > 15
            ]

            return {
                "is_interleaved": True, "num_streams": ar_order,
                "detection_reason": f"Hypothesis based on a high-order ({ar_order}) and highly accurate (MSE ratio: {ar_mse_ratio:.4f}) AR model candidate.",
                "sub_stream_analysis": sub_stream_reports
            }
        
    # If none of the strong evidence works, we return a negative result.
    return {"is_interleaved": False, "num_streams": 1, "detection_reason": "No strong evidence of interleaving found in model residuals or AR candidates.", "sub_stream_analysis": []}

# ===================================================================
# 12. NONLINEAR AND FRACTAL METRICS
# ===================================================================
def nonlinear_fractal_analysis(seq: np.ndarray) -> Dict[str, Any]:
    """
    Assesses the complexity, randomness, and fractal properties of a series.
    Thresholds are granular for each metric.
    """
    if not NOLDS_AVAILABLE:
        return {"comment": "Nolds library not available."}
    
    n = len(seq)
    results = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Hurst/DFA can provide an estimate of >50 points (although less accurate)
        try: results['hurst_exponent'] = float(nolds.hurst_rs(seq)) if n >= 50 else None
        except Exception: results['hurst_exponent'] = None
        try: results['dfa_exponent'] = float(nolds.dfa(seq)) if n >= 50 else None
        except Exception: results['dfa_exponent'] = None
        # Correlation/Lyapunov require more data to construct the attractor
        try: results['correlation_dimension'] = float(nolds.corr_dim(seq, emb_dim=10)) if n >= 100 else None
        except Exception: results['correlation_dimension'] = None
        try: results['lyapunov_exponent'] = float(nolds.lyap_r(seq, emb_dim=10)) if n >= 100 else None
        except Exception: results['lyapunov_exponent'] = None

    # Permutation Entropy is less demanding
    try:
        if n < 30: raise ValueError("Too short")
        def _pe(ts, m=3, d=1):
            perms = [tuple(np.array(ts[i:i+(m-1)*d+1:d]).argsort()) for i in range(len(ts)-(m-1)*d)]
            counts = np.unique(perms, return_counts=True, axis=0)[1]
            return stats.entropy(counts / len(perms), base=2)
        results['permutation_entropy'] = float(_pe(seq))
    except Exception: results['permutation_entropy'] = None

    # MFDFA is also demanding in terms of length
    try:
        if n > 100:
            lag = np.unique(np.logspace(0.5, 2.0, 15).astype(int))
            q_range = np.linspace(-5, 5, 11); q_range = q_range[q_range != 0]
            h_q_range = nolds.mfdfa(seq, lag=lag, q=q_range, order=1)
            results['multifractal_spectrum_width'] = float(np.max(h_q_range) - np.min(h_q_range))
        else:
            results['multifractal_spectrum_width'] = None
    except Exception: results['multifractal_spectrum_width'] = None

    return results

# ===================================================================
# 13. SPECTRAL ANALYSIS
# ===================================================================
def spectral_analysis(seq: np.ndarray) -> Dict[str, Any]:
    """
    Frequency domain analysis: PSD, wavelets and extended spectral metrics.
    """
    n = len(seq)
    if n < 20:
        return {"power_spectral_density": {}, "wavelet_analysis": {}, "comment": "Too short for spectral analysis"}
    
    results = {}
    
    # === Power Spectral Density (PSD) Analysis ===
    try:
        freqs, psd = periodogram(seq)
        psd_sum = np.sum(psd)
        
        if psd_sum > 1e-9:
            psd_norm = psd / psd_sum
            dominant_freq_idx = np.argmax(psd)
            
            # Spectral centroid (center of mass of the spectrum)
            centroid = np.sum(freqs * psd_norm)
            # Spectral spread (standard deviation of the spectrum)
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
            # Spectral entropy (a measure of the "flatness" or complexity of a spectrum)
            spectral_entropy = stats.entropy(psd_norm)
            
            results['power_spectral_density'] = {
                "dominant_frequency": float(freqs[dominant_freq_idx]),
                "power_at_dominant_freq": float(psd[dominant_freq_idx]),
                "spectral_entropy": float(spectral_entropy),
                "spectral_centroid": float(centroid),
                "spectral_spread": float(spread)
            }
        else:
            results['power_spectral_density'] = {"comment": "Signal has zero power"}

    except Exception as e:
        results['power_spectral_density'] = {"error": f"PSD calculation failed: {e}"}

    # === Wavelet analysis ===
    if PYWT_AVAILABLE:
        try:
            wavelet = 'db4'
            # Calculate the maximum possible level and limit it for stability.
            max_level = pywt.dwt_max_level(data_len=n, filter_len=pywt.Wavelet(wavelet).dec_len)
            level = min(max_level, 5) # Limit to 5 levels
            
            if level > 0:
                coeffs = pywt.wavedec(seq, wavelet, level=level)
                # Energy at each level of detail + approximation
                energy = [np.sum(np.square(c)) for c in coeffs]
                total_energy = np.sum(energy)
                
                if total_energy > 1e-9:
                    energy_distribution = [float(e / total_energy * 100) for e in energy]
                else:
                    energy_distribution = [0.0] * (level + 1)

                results['wavelet_analysis'] = {
                    "wavelet_family": wavelet,
                    "decomposition_levels": level,
                    "energy_distribution_percent": energy_distribution
                }
            else:
                 results['wavelet_analysis'] = {"comment": "Sequence too short for wavelet decomposition"}
        except Exception as e:
            results['wavelet_analysis'] = {"error": f"Wavelet analysis failed: {e}"}
    else:
        results['wavelet_analysis'] = {"comment": "PyWavelets library not available"}

    return results

# ===================================================================
# 14. CLASSIFICATION OF PATTERNS
# ===================================================================
def classify_pattern(meta: Dict[str, Any], seq: np.ndarray) -> str:
    """Assigns a class to a sequence based on the entire set of metaparameters."""
    try:
        dm = meta.get('dependency_model', {})
        
        # --- PRIORITIZED CHECK FOR EXACT RECURRENCE ---
        # This check must come before the length check to correctly classify short, exact sequences.
        if dm.get('dependency_type') == 'value_dependent (exact)':
            bma_analysis = dm.get('berlekamp_massey_analysis', {})
            if bma_analysis and 'formula_str' in bma_analysis:
                return f"Exact Linear Recurrence: {bma_analysis['formula_str']}"
        
        # --- Standard length check for all other, less certain patterns ---
        if len(seq) < 20:
            return "Indeterminate (Too Short)"
            
        cyc = meta.get('cyclicity_and_seasonality', {})
        st = meta.get('stationarity', {})
        nfa = meta.get('nonlinear_fractal_analysis', {})
        
        if cyc.get('strength', 0) > 0.6: 
            return f"Strongly Cyclic (T={cyc.get('dominant_period')})"
        
        dep_type = dm.get('dependency_type')
        # The best_model can now be directly under dependency_model for exact types
        best_model = dm.get('best_model', {}) if dm.get('best_model') else dm

        # Provide a more detailed description for value-dependent models.
        if 'value_dependent' in dep_type: # Catches both 'value_dependent' and 'value_dependent (exact)'
            order = best_model.get('order', 0)
            if order > 0:
                coeffs = best_model.get('coeffs', [])
                # For approximate models, show the floating point equation
                if coeffs and dep_type != 'value_dependent (exact)':
                    ar_coeffs = coeffs[:-1]
                    terms = [f"{c:+.2f}*a(n-{i+1})" for i, c in enumerate(ar_coeffs)]
                    recurrence_str = " ".join(terms).lstrip("+ ")
                    return f"Linear Recurrence (Order {order}): a(n) ≈ {recurrence_str}"
                return f"Autoregressive (Order {order})"

        if dep_type == "index_dependent" and best_model.get('type') == 'exponential':
            growth_rate = best_model.get('params', {}).get('b', 0)
            return "Exponential Growth" if growth_rate > 0 else "Exponential Decay"

        if dep_type == "index_dependent": 
            return f"Functional Trend ({best_model.get('type')})"
            
        hurst = nfa.get('hurst_exponent')
        if hurst is not None:
            if hurst > 0.75: return "Strongly Persistent / Long Memory"
            if hurst < 0.25: return "Strongly Anti-persistent / Mean-reverting"
        
        if dep_type == "mixed": return "Mixed Deterministic Pattern"
        
        if st.get('conclusion') == 'Stationary': return "Stochastic (Stationary)"
        if st.get('conclusion') == 'Non-stationary (unit root)': return "Stochastic (Random Walk-like)"
        
        return "Complex / Non-stationary"
    except Exception:
        return "Classification Error"

# ===================================================================
# 15. DECOMPOSITION
# ===================================================================
def decomposition_analysis(seq: np.ndarray) -> Dict[str, Any]:
    """
    Performs STL decomposition of a series into trend, seasonality, and residuals.
    Transition to an adaptive length threshold based on the period.
    """
    n = len(seq)
    if not STATSMODELS_STL_AVAILABLE or not STATSMODELS_AVAILABLE or n < 15:
        return {"comment": "Not enough data or required statsmodels components missing."}
    
    try:
        # Automatic period detection
        autocorr = acf(seq, nlags=min(n // 2, 40), fft=True)[1:]
        if len(autocorr) > 0:
            potential_period = np.argmax(autocorr) + 1
            # If the ACF peak is weak or the period is too short, STL is not applicable
            if autocorr[potential_period-1] < 0.25 or potential_period <= 1:
                return {"comment": "No clear seasonality detected for STL."}
        else:
            return {"comment": "Could not determine seasonality period."}
        
        # Adaptive threshold: STL requires at least 2 full periods
        if n < 2 * potential_period:
            return {"comment": f"Sequence is too short for detected period T={potential_period} (requires at least {2*potential_period} points)."}

        res = STL(seq, period=potential_period).fit()
        trend_strength = max(0, 1 - np.var(res.resid) / np.var(res.trend + res.resid))
        seasonal_strength = max(0, 1 - np.var(res.resid) / np.var(res.seasonal + res.resid))
        
        return {
            "trend_strength": float(trend_strength), "seasonality_strength": float(seasonal_strength),
            "period_used": int(potential_period), "residual_variance": float(np.var(res.resid))
        }
    except Exception as e:
        return {"error": f"STL decomposition failed: {e}"}

# ===================================================================
# 16. VOLATILITY
# ===================================================================
def volatility_analysis(seq: np.ndarray, window: int = 20) -> Dict[str, Any]:
    """
    Estimates the volatility of a series, including the GARCH model.
    """
    n = len(seq)
    if n < window:
        return {"rolling_std_mean": None, "garch_params": None, "comment": "Sequence too short."}

    # A simple metric: the moving standard deviation of the differences
    diffs = np.diff(seq)
    rolling_std = np.std([diffs[i:i+window] for i in range(len(diffs) - window)])
    
    results = {"rolling_std_of_diffs": float(rolling_std)}
    
    # GARCH analysis
    if ARCH_AVAILABLE and n > 50:
        try:
            # Use returns because GARCH models the volatility of returns.
            returns = 100 * np.diff(np.log(np.abs(seq) + 1e-8))
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            fit = model.fit(disp="off")
            params = fit.params
            results["garch_params"] = {
                "alpha_1": float(params.get('alpha[1]', 0)), # ARCH term
                "beta_1": float(params.get('beta[1]', 0))   # GARCH term
            }
        except Exception as e:
            results["garch_params"] = {"error": f"GARCH fit failed: {e}"}
    else:
        results["garch_params"] = {"comment": "ARCH library not available or sequence too short."}
        
    return results

# ===================================================================
# 17. MOTIFS ANALYSIS
# ===================================================================
def motif_analysis(seq: np.ndarray, window: int = 30) -> Dict[str, Any]:
    """
    Uses Matrix Profile to find repeating patterns (motifs) and anomalies (dissonances).
    """
    n = len(seq)
    m = min(window, n // 4)
    if not STUMPY_AVAILABLE or n < m * 2 or m < 4:
        return {"comment": "Not enough data or 'stumpy' library missing."}
        
    try:
        # --- Stumpy requires an exact float64 type ---
        seq_float64 = seq.astype(np.float64)
        if not np.all(np.isfinite(seq_float64)):
            return {"comment": "Sequence contains non-finite values, skipping."}
        mp = stumpy.stump(seq_float64, m=m)
        
        motif_idx = np.argmin(mp[:, 0])
        neighbor_idx = mp[motif_idx, 1]
        
        discord_idx = np.argmax(mp[:, 0])
        
        return {
            "window_size": m,
            "top_motif_indices": [int(motif_idx), int(neighbor_idx)],
            "top_motif_distance": float(mp[motif_idx, 0]),
            "top_discord_index": int(discord_idx)
        }
    except Exception as e:
        return {"error": f"Matrix Profile analysis failed: {e}"}

# ===================================================================
# 18. SEGMENTATION BY CONDITIONS
# ===================================================================
def state_segmentation_analysis(seq: np.ndarray, n_states: int = 3) -> Dict[str, Any]:
    """
    Uses a hidden Markov model (HMM) to segment the series into hidden states.
    """
    n = len(seq)
    # Reduced multiplier from 20 to 12 for more flexibility
    if not HMMLEARN_AVAILABLE or n < n_states * 12:
        return {"comment": "Not enough data or 'hmmlearn' library missing."}

    try:
        # Use 'diag' instead of 'full' for greater stability on 1D data
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, tol=1e-2)
        reshaped_seq = seq.reshape(-1, 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(reshaped_seq)
        
        # Unified processing of incorrect matrices
        # If the matrix is ​​incorrect, we return a standardized comment rather than throwing an error.
        if not np.all(np.isclose(model.transmat_.sum(axis=1), 1)):
             return {
                "error": "Model fitting resulted in an invalid transition matrix.",
                "comment": "HMM is unsuitable for this sequence, likely due to strong monotonicity or other data characteristics preventing valid state transitions."
            }

        hidden_states = model.predict(reshaped_seq)
        state_means = model.means_.flatten().tolist()
        # Covariance is a vector, not a matrix.
        state_variances = model.covars_.flatten().tolist()
        
        unique, counts = np.unique(hidden_states, return_counts=True)
        state_counts = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))

        return {
            "num_states": n_states,
            "state_means": state_means,
            "state_variances": state_variances,
            "state_counts": state_counts
        }
    except ValueError as e:
        error_str = str(e).lower()
        if "transmat_" in error_str and "sum to 1" in error_str:
            return {
                "error": "Model fitting failed due to data characteristics.",
                "comment": "HMM is unsuitable for this sequence, likely due to strong monotonicity preventing valid state transitions from being learned."
            }
        return {"error": f"HMM segmentation failed with ValueError: {e}"}
    except Exception as e:
        return {"error": f"HMM segmentation failed with an unexpected error: {e}"}

# ===================================================================
# 19. Complexity Analysis
# ===================================================================
def complexity_analysis(seq: np.ndarray, m: int = 2) -> Dict[str, Any]:
    """
    Evaluates advanced complexity metrics: Lempel-Ziv (compression-based) and Approximate Entropy.
    """
    n = len(seq)
    results = {}
    
    # --- Lempel-Ziv Complexity ---
    # Estimates the complexity of a sequence based on its compressibility. The more unique subsequences, the higher the complexity.
    if NOLDS_AVAILABLE and n >= 20:
        try:
            # LZ requires a discrete (symbolic) input.
            # The standard approach is binarization with respect to the median.
            median_val = np.median(seq)
            binary_seq = (seq > median_val).astype(int)
            
            # nolds.lz_complexity returns the number of unique substrings.
            # Let's normalize it to get a value between ~0 and 1.
            lz_raw = nolds.lz_complexity(binary_seq)
            # Theoretical maximum for a binary sequence: n / log2(n)
            # This normalization makes the result more comparable for series of different lengths.
            lz_normalized = lz_raw * np.log2(n) / n if n > 1 else 0
            results["lempel_ziv_complexity_normalized"] = float(lz_normalized)
        except Exception:
            results["lempel_ziv_complexity_normalized"] = None
    else:
        results["lempel_ziv_complexity_normalized"] = None

    # --- Approximate Entropy - ApEn ---
    # Similar to Sample Entropy, but includes self-comparison of templates in the calculation.
    # May be biased for short rows, but useful as an additional metric.
    def _apen(L, m_arg, r):
        N = len(L)

        def _phi(m_phi):
            x = np.array([L[i : i + m_phi] for i in range(N - m_phi + 1)])
            C = np.zeros(len(x))
            for i in range(len(x)):
                # Chebyshev distance between x[i] and all x[j]
                diffs = np.max(np.abs(x[i] - x), axis=1)
                C[i] = np.sum(diffs < r) / len(x)
            
            return np.mean(np.log(C))

        phi_m = _phi(m_arg)
        phi_m_plus_1 = _phi(m_arg + 1)
        
        return phi_m - phi_m_plus_1

    if n >= 20:
        try:
            std_dev = np.std(seq)
            if std_dev > 1e-9:
                r_val = 0.2 * std_dev
                approx_entropy = _apen(seq, m, r_val)
            else:
                approx_entropy = 0.0 # For a constant series, the entropy is 0
            results["approximate_entropy"] = float(approx_entropy)
        except Exception:
            results["approximate_entropy"] = None
    else:
        results["approximate_entropy"] = None
        
    if not results:
        return {"comment": "Sequence too short or required libraries missing for analysis."}
        
    return results

# ===================================================================
# 20. SYMBOLIC ANALYSIS
# ===================================================================
def symbolic_analysis(seq: np.ndarray, n_bins=8, word_size=4) -> Dict[str, Any]:
    """
    Performs SAX transformation and parses the symbolic representation of a sequence.
    """
    n = len(seq)
    if not PYTS_AVAILABLE or n < 30:
        return {"comment": "Not enough data or 'pyts' library missing."}

    try:
        # Use the 'normal' strategy, which assumes a Gaussian distribution
        # after z-normalization. This works significantly better for trended and
        # asymmetric distributions than 'uniform'.
        transformer = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
        sax_symbols = transformer.fit_transform(seq.reshape(1, -1))
        
        # Calculate the entropy of the distribution of symbols
        _, counts = np.unique(sax_symbols, return_counts=True)
        symbol_entropy = stats.entropy(counts / len(sax_symbols[0]), base=2)
        
        return {"sax_symbol_entropy": float(symbol_entropy), "num_symbols": n_bins}
    except Exception as e:
        return {"error": f"Symbolic (SAX) analysis failed: {e}"}

# ===================================================================
# 21. EMPIRICAL MODE DECOMPOSITION
# ===================================================================
def empirical_mode_decomposition(seq: np.ndarray) -> Dict[str, Any]:
    """
    Decomposes a non-stationary series into internal mode functions (IMFs).
    """
    if not PYEMD_AVAILABLE or len(seq) < 50:
        return {"comment": "Not enough data or 'PyEMD' library missing."}
        
    try:
        emd = EMD()
        imfs = emd.emd(seq)
        return {
            "num_imfs": len(imfs),
            "residual_variance_ratio": float(np.var(imfs[-1]) / np.var(seq)) if np.var(seq) > 0 else 0
        }
    except Exception as e:
        return {"error": f"EMD analysis failed: {e}"}

# ===================================================================
# 22. SEGMENT CLUSTERING
# ===================================================================
def clustering_segmentation(seq: np.ndarray, n_clusters=3, window=30) -> Dict[str, Any]:
    """
    Segments a series using sliding window clustering.
    """
    n = len(seq)
    if not TSLEARN_AVAILABLE or n < window * n_clusters:
        return {"comment": "Not enough data or 'tslearn' library missing."}

    try:
        # Creating a dataset from sliding windows
        subsequences = np.array([seq[i:i+window] for i in range(n - window + 1)])
        # tslearn requires 3D input: (n_samples, n_timesteps, n_features)
        dataset = to_time_series_dataset(subsequences)
        
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0, n_jobs=-1)
        
        # Suppress the harmless UndefinedMetricWarning warning
        # It occurs when only one element is in the cluster.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            labels = model.fit_predict(dataset)
        
        # Sort cluster centers by their mean value for consistency
        cluster_centers = model.cluster_centers_
        center_means = np.mean(cluster_centers, axis=1).flatten()
        sorted_indices = np.argsort(center_means)
        
        # Reassigning labels according to sorted centers
        sorted_labels = np.zeros_like(labels)
        for i, original_idx in enumerate(sorted_indices):
            sorted_labels[labels == original_idx] = i

        _, counts = np.unique(sorted_labels, return_counts=True)
        # Make sure we have counters for all clusters, even if they are empty (unlikely)
        cluster_counts = {i: 0 for i in range(n_clusters)}
        for label, count in zip(np.unique(sorted_labels), counts):
            cluster_counts[label] = int(count)

        return {"n_clusters": n_clusters, "segment_counts_per_cluster": cluster_counts}
    except Exception as e:
        return {"error": f"Time series clustering failed: {e}"}

# ===================================================================
# 23. NUMBER-THEORETICAL PROPERTIES
# ===================================================================
import math

def is_prime(n):
    """A simple primality check."""
    if n < 2: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def number_theoretic_properties(seq: np.ndarray) -> Dict[str, Any]:
    """Analyzes a sequence for number-theoretic properties."""
    if len(seq) < 5:
        return {"comment": "Sequence too short."}

    # Convert to integers if possible
    try:
        int_seq = seq.astype(np.int64)
        # Using np.allclose to correctly check float representations of integers
        if not np.allclose(seq, int_seq):
            return {"comment": "Sequence is not purely integer."}
    except (ValueError, TypeError, OverflowError):
        return {"comment": "Sequence is not integer-like or contains values too large for standard integers."}

    results = {}
    
    # Property 1: Are all elements prime numbers?
    prime_count = sum(1 for x in int_seq if is_prime(x))
    results["prime_percentage"] = (prime_count / len(int_seq)) * 100

    # Property 2: Analysis of differences (gaps) between elements
    diffs = np.diff(int_seq)
    if len(diffs) > 1:
        # Are all differences even (except perhaps the first)?
        # This is a strong indication of prime numbers > 2
        even_gap_count = sum(1 for d in diffs[1:] if d % 2 == 0)
        results["even_gap_percentage_after_first"] = (even_gap_count / (len(diffs) - 1)) * 100 if len(diffs) > 1 else 100.0

    return results

# ===================================================================
# 24. PROPERTIES OF DIVISIBILITY
# ===================================================================
def divisibility_properties(seq: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
    """Analyzes the divisibility properties of a sequence."""
    # Check if the values ​​are integers, not just dtype.
    # This allows to work with longdouble arrays containing integers.
    try:
        int_seq = seq.astype(np.int64)
        if not np.allclose(seq, int_seq):
            return {"comment": "Sequence is not purely integer."}
    except (ValueError, TypeError, OverflowError):
        return {"comment": "Sequence is not integer-like or contains values too large."}

    if len(int_seq) < 5:
        return {"comment": "Sequence too short."}

    results = {}
    n = len(int_seq)
    
    # Even/odd check
    even_count = np.sum(int_seq % 2 == 0)
    if even_count / n >= threshold: results["all_divisible_by"] = 2
    elif (n - even_count) / n >= threshold: results["all_odd"] = True

    # Testing for divisibility by other small prime numbers
    if "all_divisible_by" not in results:
        for p in [3, 5, 7]:
            if np.sum(int_seq % p == 0) / n >= threshold:
                results["all_divisible_by"] = p
                break
    
    return results

def analyze_ratios(seq: np.ndarray) -> Dict[str, Any]:
    """Analyzes the behavior of the ratio a(n)/a(n-1)."""
    if seq.dtype != np.longdouble:
        seq = seq.astype(np.longdouble)
    if len(seq) < 10: return {"comment": "Sequence too short for ratio analysis."}
    
    # Begin the analysis with the first non-zero element
    try:
        non_zero_indices = np.where(seq != 0)[0]
        if len(non_zero_indices) < 2:
             return {"comment": "Not enough non-zero elements for ratio analysis."}
        start_idx = non_zero_indices[0]
    except IndexError:
        return {"comment": "Sequence is all zeros."}
        
    # Let's make sure there is enough data for analysis after start_idx
    if len(seq) - (start_idx + 1) < 5:
        return {"comment": "Not enough elements after the first non-zero for ratio analysis."}

    # Create slices for numerators and denominators, skipping zeros
    numerators = seq[start_idx+1:]
    denominators = seq[start_idx:-1]
    
    # Find indices where the denominator is not equal to zero
    valid_indices = np.where(denominators != 0)[0]
    if len(valid_indices) < 5:
        return {"comment": "Too many zeros in sequence prevent stable ratio analysis."}
        
    ratios = (numerators[valid_indices] / denominators[valid_indices]).astype(np.longdouble)
    x_data = np.arange(start_idx + 1, len(seq), dtype=np.longdouble)[valid_indices]
    y_data = ratios

    if np.allclose(ratios, ratios[0]):
        return {"ratio_type": "constant", "value": float(ratios[0])}

    best_fit = {"p_deg": -1, "q_deg": -1, "r_squared": -1.0, "p_coeffs": [], "q_coeffs": []}
    
    for p_deg in range(1, 4):
        for q_deg in range(1, 4):
            if len(x_data) < p_deg + q_deg + 1: continue
            A_p = -np.vstack([x_data**i for i in range(p_deg, -1, -1)]).T
            A_q = np.vstack([y_data * x_data**i for i in range(q_deg - 1, -1, -1)]).T
            A = np.hstack([A_p, A_q]); b = -y_data * x_data**q_deg
            if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)): continue
            try:
                # Use float64 for lstsq, as it is more optimized.
                coeffs, _, _, _ = np.linalg.lstsq(A.astype(np.float64), b.astype(np.float64), rcond=None)
                p_coeffs = coeffs[:p_deg + 1]; q_coeffs_rem = coeffs[p_deg + 1:]
                q_coeffs = np.concatenate(([1.0], q_coeffs_rem))
                y_pred = np.poly1d(p_coeffs)(x_data) / np.poly1d(q_coeffs)(x_data)
                
                # Using R-squared instead of a custom metric
                ss_total = np.sum((y_data - np.mean(y_data))**2)
                if ss_total < 1e-12: # If the data is constant
                    r_squared = 1.0 if np.allclose(y_data, y_pred) else 0.0
                else:
                    ss_residual = np.sum((y_data - y_pred)**2)
                    r_squared = 1 - (ss_residual / ss_total)

                if r_squared > best_fit["r_squared"]:
                    best_fit.update({
                        "p_deg": p_deg, "q_deg": q_deg, "r_squared": r_squared,
                        "p_coeffs": p_coeffs.tolist(), "q_coeffs": q_coeffs.tolist()
                    })
            except (np.linalg.LinAlgError, ValueError): 
                continue
    
    # Check R-squared to determine the quality of the approximation
    if best_fit["r_squared"] > 0.99:
        return {
            "ratio_type": "rational_function",
            "p_coeffs": best_fit["p_coeffs"],
            "q_coeffs": best_fit["q_coeffs"],
            "r_squared": float(best_fit["r_squared"])
        }
    return {"ratio_type": "complex", "values": [float(v) for v in ratios]}

# ===================================================================
# 25. SEARCH FOR DOMINANT MULTIPLICATIVE RECURRENCE
# ===================================================================
def simple_recurrence_analysis(seq: np.ndarray, max_offset: int = 15, num_formulas: int = 3) -> Dict[str, Any]:
    """
    Analyzes a sequence for simple recurrences of the form a(n) ≈ C * a(n-offset), where C is an integer.
    """
    if len(seq) < 10: # Minimum length for meaningful analysis
        return {"comment": "Sequence too short for simple recurrence analysis."}
        
    # Using longdouble to work with large numbers without losing precision
    try:
        seq_ld = seq.astype(np.longdouble)
    except (ValueError, OverflowError):
        return {"comment": "Could not convert sequence to high-precision float."}

    if np.all(seq_ld == 0):
        return {"comment": "Sequence contains only zeros."}

    found_formulas = []
    
    # Iterate over all possible offsets
    for offset in range(1, min(max_offset, len(seq) // 2) + 1):
        
        # Vectorized calculation of coefficients where possible
        denominators = seq_ld[:-offset]
        numerators = seq_ld[offset:]
        
        # Division-by-zero protection mask
        valid_mask = denominators != 0
        if np.sum(valid_mask) < 5: # Require at least 5 points for statistics
            continue
            
        ratios = numerators[valid_mask] / denominators[valid_mask]
        
        # Find the most probable integer factor
        median_ratio = np.median(ratios)
        best_C = int(round(median_ratio))
        
        # Skip trivial cases
        if best_C == 0 or best_C == 1:
            continue
            
        # Vectorized Formula Quality Assessment (MAPE)
        actual = seq_ld[offset:]
        predicted = seq_ld[:-offset] * best_C
        
        # Division-by-zero protection mask in MAPE
        valid_mape_mask = actual != 0
        if np.sum(valid_mape_mask) == 0:
            continue
            
        errors_pct = np.abs(actual[valid_mape_mask] - predicted[valid_mape_mask]) / actual[valid_mape_mask]
        mape = np.mean(errors_pct) * 100
        
        # Only keep formulas that are good enough
        if mape < 50: # Cutoff threshold (in %)
            found_formulas.append({
                'mape': float(mape),
                'offset': int(offset),
                'C': int(best_C),
                'formula_str': f"a(n) ≈ {best_C} * a(n-{offset})"
            })

    # Sort all found formulas by their error
    sorted_formulas = sorted(found_formulas, key=lambda x: x['mape'])
    
    if not sorted_formulas:
        return {"comment": "No simple multiplicative recurrence found."}
        
    return {"best_formulas": sorted_formulas[:num_formulas]}

# ===================================================================
# 26. SEARCH FOR HYBRID MODELS (with multiple data streams)
# ===================================================================
def hybrid_model_analysis(seq: np.ndarray, base_meta_report: Dict[str, Any], max_k: int = 8) -> Dict[str, Any]:
    """
    Searches for hybrid models where the generation rules change cyclically depending on n % k.
    """
    # --- Guarantee that the input is a NumPy array ---
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
        
    # This analysis is very data-intensive, so the threshold is higher.
    if len(seq) < 50:
        return {"comment": "Sequence too short for hybrid model analysis."}

    # Step 1: Obtain the best baseline candidate formulas from the analysis already performed
    base_formulas = base_meta_report.get('simple_recurrence', {}).get('best_formulas', [])
    if len(base_formulas) < 2:
        return {"comment": "Not enough good base formula candidates to build a hybrid model."}

    # Use longdouble for precision
    try:
        # Use an existing array if it is of the appropriate type.
        if seq.dtype == np.longdouble:
            seq_ld = seq
        else:
            seq_ld = seq.astype(np.longdouble)
    except (ValueError, OverflowError):
        return {"comment": "Could not convert sequence to high-precision float."}

    best_hybrid_model = {'mape': float('inf')}
    
    # Step 2: For each base formula, calculate its remainders (where it goes wrong)
    residuals_map = {}
    for f in base_formulas:
        offset, C = f['offset'], f['C']
        predicted = seq_ld[:-offset] * C
        # Residuals as the ratio of actual/predicted, centered around 1.0
        residuals = np.ones_like(seq_ld)
        # Division-by-zero protection
        valid_mask = (predicted != 0)
        residuals[offset:][valid_mask] = seq_ld[offset:][valid_mask] / predicted[valid_mask]
        residuals_map[f['formula_str']] = residuals

    # Step 3: Heuristic search instead of exhaustive search
    from itertools import combinations
    formula_pairs = list(combinations(base_formulas, 2))

    for f1, f2 in formula_pairs:
        # The residuals show where each model is better (closer to 1.0)
        res1 = np.abs(residuals_map[f1['formula_str']] - 1)
        res2 = np.abs(residuals_map[f2['formula_str']] - 1)
        
        # Create a "choice map" - which model is best for which?
        # 0 if f1 is better, 1 if f2 is better
        choice_map = (res2 < res1).astype(int)
        
        # Look for periodicity in this choice map
        for k in range(2, max_k + 1):
            if len(choice_map) < k * 3: continue
            
            # Find the dominant pattern for a given k
            pattern = []
            for i in range(k):
                votes = choice_map[i::k]
                winner = 0 if np.sum(votes == 0) >= np.sum(votes == 1) else 1
                pattern.append(winner)
            
            if all(p == 0 for p in pattern) or all(p == 1 for p in pattern):
                continue

            # Step 4: Evaluate the quality of the hybrid model with the found pattern
            errors_pct = []
            max_offset = max(f1['offset'], f2['offset'])
            for n in range(max_offset, len(seq_ld)):
                active_rule_idx = pattern[n % k]
                active_formula = f1 if active_rule_idx == 0 else f2
                offset, C = active_formula['offset'], active_formula['C']
                
                prediction = seq_ld[n - offset] * C
                actual = seq_ld[n]
                if actual != 0:
                    errors_pct.append(np.abs(actual - prediction) / np.abs(actual))

            if not errors_pct: continue
            mape = np.mean(errors_pct) * 100

            if mape < best_hybrid_model['mape']:
                best_hybrid_model = {
                    'mape': mape,
                    'base_formulas': [f1['formula_str'], f2['formula_str']],
                    'k': k,
                    'pattern': pattern
                }

    if best_hybrid_model['mape'] > 50:
        return {"comment": "No accurate hybrid model found."}

    return best_hybrid_model

def hybrid_model_analysis_v2(seq: np.ndarray, max_k: int = 7, max_offset: int = 12) -> Dict[str, Any]:
    """
    Finds the best hybrid model by dividing the indices into two groups of n % k and finding the optimal simple recurrence for each group independently.
    """
    if len(seq) < max_offset * 2:
        return {"comment": "Sequence too short for this analysis."}

    try:
        seq_obj = seq.astype(object)
    except (ValueError, OverflowError):
        return {"comment": "Could not convert sequence to object type for arbitrary precision."}

    def find_best_formula_for_subset(subset_n):
        best_formula = {'mape': float('inf'), 'offset': None, 'C': None}
        if not subset_n or min(subset_n) < max_offset: return best_formula

        for offset in range(1, max_offset + 1):
            denominators = seq_obj[np.array(subset_n) - offset]
            numerators = seq_obj[subset_n]
            valid_mask = denominators != 0
            if np.sum(valid_mask) < 3: continue
            
            ratios = numerators[valid_mask] / denominators[valid_mask]
            best_C = int(round(np.median(ratios)))
            if best_C in [0, 1]: continue

            actual = seq_obj[subset_n]
            predicted = seq_obj[np.array(subset_n) - offset] * best_C
            valid_mape_mask = actual != 0
            if np.sum(valid_mape_mask) == 0: continue

            errors_pct = np.abs(actual[valid_mape_mask] - predicted[valid_mape_mask]) / np.abs(actual[valid_mape_mask])
            mape = np.mean(errors_pct) * 100
            
            if mape < best_formula['mape']:
                best_formula = {'mape': mape, 'offset': offset, 'C': best_C}
        return best_formula

    best_hybrid_model = {'mape': float('inf')}
    all_n = list(range(max_offset, len(seq_obj)))

    from itertools import combinations
    for k in range(2, max_k + 1):
        phases = list(range(k))
        # Go through all possible divisions into 2 groups (up to half, since the rest is symmetrical)
        for size_of_group1 in range(1, k // 2 + 1):
            for group1_phases in combinations(phases, size_of_group1):
                group1_phases = set(group1_phases)
                group2_phases = set(phases) - group1_phases
                
                group1_n = [n for n in all_n if (n % k) in group1_phases]
                group2_n = [n for n in all_n if (n % k) in group2_phases]
                
                f1 = find_best_formula_for_subset(group1_n)
                f2 = find_best_formula_for_subset(group2_n)
                
                if f1['mape'] == float('inf') or f2['mape'] == float('inf'): continue
                    
                hybrid_mape = (f1['mape'] * len(group1_n) + f2['mape'] * len(group2_n)) / len(all_n)
                
                if hybrid_mape < best_hybrid_model['mape']:
                    best_hybrid_model = {
                        'mape': hybrid_mape, 'k': k,
                        'group1_phases': sorted(list(group1_phases)), 'f1': f1,
                        'group2_phases': sorted(list(group2_phases)), 'f2': f2
                    }

    if best_hybrid_model['mape'] > 20: # The threshold for a good model
        return {"comment": "No accurate hybrid model found."}
        
    return best_hybrid_model

def local_formula_analysis(seq: np.ndarray, max_offset: int = 15, error_threshold: float = 0.1) -> Dict[str, Any]:
    """
    For each step, n searches for the best simple formula a(n) ≈ C * a(n-k)
    and returns the sequence of offsets k found.
    """
    if len(seq) < max_offset + 2:
        return {"comment": "Sequence too short for this analysis."}
    
    try:
        seq_obj = seq.astype(object)
    except (ValueError, OverflowError):
        return {"comment": "Could not convert sequence to object type."}

    sequence_of_k = []
    sequence_of_c = []
    
    for n in range(max_offset, len(seq_obj)):
        best_fit_for_n = {'error': float('inf'), 'k': None, 'C': None}
        
        for k in range(1, max_offset + 1):
            if seq_obj[n - k] == 0: continue
                
            C_float = seq_obj[n] / seq_obj[n - k]
            C_int = int(round(C_float))
            
            if C_int in [0, 1]: continue
                
            error = abs(C_float - C_int)
            
            if error < best_fit_for_n['error']:
                best_fit_for_n = {'error': error, 'k': k, 'C': C_int}
                
        if best_fit_for_n['error'] < error_threshold:
            sequence_of_k.append(best_fit_for_n['k'])
            sequence_of_c.append(best_fit_for_n['C'])
        else:
            sequence_of_k.append(None)
            sequence_of_c.append(None)
    
    # Analyze the sequence k itself for patterns
    k_series = [k for k in sequence_of_k if k is not None]
    if not k_series:
        return {"comment": "No consistent local formulas found."}
    
    k_analysis = {}
    unique_k, counts_k = np.unique(k_series, return_counts=True)
    k_analysis['k_distribution'] = dict(zip(unique_k.astype(int), counts_k.astype(int)))
    
    # Checking whether the sequence k is constant
    if len(unique_k) == 1:
        k_analysis['k_pattern'] = f"Constant k = {unique_k[0]}"
    
    return {
        "k_sequence": sequence_of_k,
        "C_sequence": sequence_of_c,
        "k_analysis": k_analysis
    }

# ===================================================================
# 27. Finding the greatest common divisor and finding common roots (perfect power analysis)
# ===================================================================
import math

def _is_perfect_power(n: int, k: int) -> Tuple[bool, Optional[int]]:
    """(Helper) Robustly checks if n is a perfect k-th power."""
    if n < 1 or k < 2:
        return False, None
    
    # Use integer-based root finding to avoid float precision errors with large numbers
    root = int(round(n**(1/k)))
    
    # Check for both sides of the rounding to be sure
    if (root + 1)**k == n:
        return True, root + 1
    if root**k == n:
        return True, root
    if (root - 1)**k == n and root > 1:
        return True, root - 1
        
    return False, None

def _find_robust_gcd(seq: np.ndarray) -> int:
    """(Helper) Finds the GCD of the tail of a sequence to be robust to prefixes."""
    if len(seq) < 2:
        return 1
    
    # Analyze the last 80% of the sequence, but at least 5 elements.
    start_index = max(0, len(seq) - max(5, int(len(seq) * 0.8)))
    subset = seq[start_index:]
    
    if len(subset) == 0:
        return 1
        
    # FIX: Explicit conversion to int for math.gcd.
    result_gcd = int(subset[0])
    for i in range(1, len(subset)):
        result_gcd = math.gcd(result_gcd, int(subset[i]))
        if result_gcd == 1:
            return 1 # Optimization.
            
    return result_gcd

def advanced_number_theory(seq: np.ndarray, threshold: float = 0.9) -> Dict[str, Any]:
    """
    Analyzes for deeper number-theoretic properties like a robust GCD
    and whether the sequence consists of perfect powers (squares, cubes, etc.).
    This analysis is robust to prefixes that may follow a different pattern.
    """
    # This analysis is only valid for integer-like sequences.
    try:
        # Use an object-array to work with potentially large numbers.
        int_seq = seq.astype(object)
        if not np.all(seq == int_seq):
            return {"comment": "Sequence is not purely integer."}
    except (ValueError, TypeError, OverflowError):
        return {"comment": "Sequence is not integer-like."}

    n = len(int_seq)
    if n < 5:
        return {"comment": "Sequence too short."}

    results = {}
    
    # --- 1. Robust GCD Analysis ---
    # FIX: Convert to int before calculating abs to avoid errors.
    non_zero_seq = np.array([abs(int(x)) for x in int_seq if int(x) != 0])
    if len(non_zero_seq) > 0:
        robust_gcd = _find_robust_gcd(non_zero_seq)
        
        if robust_gcd > 1:
            # FIX: Convert to int before the % operation.
            divisible_count = np.sum([int(x) % robust_gcd == 0 for x in int_seq])
            divisible_percentage = divisible_count / n
            
            if divisible_percentage >= threshold:
                results["robust_gcd"] = {
                    "value": int(robust_gcd),
                    "coverage_percentage": float(divisible_percentage * 100)
                }

    # --- 2. Perfect Power Analysis ---
    start_index = max(0, n - max(5, int(n * 0.8)))
    subset = [int(x) for x in int_seq[start_index:] if int(x) > 1] # Powers are typically > 1.
    
    if len(subset) >= 3:
        # Check for common powers (squares, cubes, etc.).
        for k in range(2, 7): # Check up to the 6th power.
            power_count = sum(1 for x in subset if _is_perfect_power(x, k)[0])
            
            if power_count / len(subset) >= threshold:
                # Found a dominant power. Now verify it on the whole sequence.
                all_positive = [int(x) for x in int_seq if int(x) > 1]
                total_power_count = sum(1 for x in all_positive if _is_perfect_power(x, k)[0])
                total_positive_count = len(all_positive)
                
                coverage = total_power_count / total_positive_count if total_positive_count > 0 else 1.0
                
                # Extract the sequence of roots.
                roots_sequence = [_is_perfect_power(int(x), k)[1] if int(x) > 1 else int(x) for x in int_seq]
                
                results["perfect_power_analysis"] = {
                    "power": k,
                    "coverage_percentage": float(coverage * 100),
                    "roots_sequence": roots_sequence
                }
                # Once the lowest dominant power is found (e.g., squares), we stop.
                break

    return results if results else {"comment": "No advanced number-theoretic properties found."}

# ===================================================================
# 28. Benford's Law analysis
# ===================================================================
def _get_first_digit(n: float) -> int:
    """(Helper) Extracts the first significant digit of a number."""
    if n == 0:
        return 0
    # Use logarithms for a fast, robust way to find the first digit
    return int(10**(math.log10(abs(n)) % 1))

def benford_law_analysis(seq: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes the conformity of the sequence's first significant digits
    to Benford's Law using a Chi-squared goodness-of-fit test.

    This test is useful for distinguishing naturally occurring data from
    artificially generated or constrained data.
    """
    # Pre-conditions for a meaningful Benford's Law test
    if len(seq) < 50:
        return {"comment": "Sequence too short for a reliable Benford's Law analysis (requires at least 50 points)."}
    
    # Filter for positive numbers, as the law applies to them
    positive_seq = seq[seq > 0]
    
    if len(positive_seq) < 30:
        return {"comment": "Not enough positive data points for analysis."}
        
    # The data should span at least two orders of magnitude
    min_val, max_val = np.min(positive_seq), np.max(positive_seq)
    orders_of_magnitude = math.log10(max_val) - math.log10(min_val) if min_val > 0 else math.log10(max_val)
    
    if orders_of_magnitude < 2:
        return {"comment": f"Data spans only {orders_of_magnitude:.2f} orders of magnitude. Benford's Law is not applicable."}

    # 1. Get the observed distribution of first digits
    first_digits = [_get_first_digit(n) for n in positive_seq]
    observed_counts = np.bincount(first_digits, minlength=10)[1:] # Counts for digits 1-9
    
    # 2. Get the expected Benford distribution
    total_count = len(first_digits)
    benford_probs = np.array([math.log10(1 + 1/d) for d in range(1, 10)])
    expected_counts = total_count * benford_probs
    
    # 3. Perform the Chi-squared goodness-of-fit test
    try:
        # Avoid test failure if an expected count is zero (unlikely here, but good practice)
        if np.any(expected_counts < 1e-9):
             return {"comment": "Cannot perform Chi-squared test due to zero expected frequencies."}
        
        chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

        # H0 (Null Hypothesis): The observed data follows the Benford distribution.
        # We fail to reject H0 if p_value is high (e.g., > 0.05).
        conforms = p_value >= 0.05
        
        observed_dist_pct = (observed_counts / total_count * 100).tolist()
        expected_dist_pct = (benford_probs * 100).tolist()

        return {
            "conforms_to_benford": bool(conforms),
            "p_value": float(p_value),
            "chi2_statistic": float(chi2_stat),
            "observed_distribution_percent": {d: pct for d, pct in enumerate(observed_dist_pct, 1)},
            "expected_distribution_percent": {d: pct for d, pct in enumerate(expected_dist_pct, 1)}
        }
    except Exception as e:
        return {"error": f"Benford's Law analysis failed: {e}"}

# ===================================================================
# 29. P-recursive (holonomic) recurrent dependency
# ===================================================================
import sympy as sp
from sympy.printing.pretty.pretty import pretty_print
import math

def find_p_recursive_relation(sequence: np.ndarray, max_order=3, max_degree=2):
    """
    Search for a P-recursive (holonomic) recurrent dependency of the form:
        p_k(n)*a(n+k) + p_{k-1}(n)*a(n+k-1) + ... + p_0(n)*a(n) = 0,
    where p_i(n) are polynomials of degree <= max_degree with INTEGER coefficients.

    Args:
        sequence (np.ndarray): The input sequence (a(0), a(1), ...).
        max_order (int): The maximum order of the recurrence (k).
        max_degree (int): The maximum degree of the polynomials p_i(n).

    Returns:
        dict | None: A dictionary with the found polynomials or None if not found.
    """
    if not SYMPY_AVAILABLE:
        return {"comment": "sympy library is not available."}
    
    # CHECK: This analysis is only applicable to integer-like sequences.
    if not np.allclose(sequence, np.round(sequence)):
        return {"comment": "Sequence is not integer-like; P-recursive analysis is not applicable."}

    N = len(sequence)
    n = sp.Symbol('n', integer=True)

    # Convert to int before creating a Rational for compatibility with sympy.
    try:
        seq_rational = [sp.Rational(int(x)) for x in sequence]
    except (TypeError, ValueError):
        return {"comment": "Failed to convert sequence to rational numbers for sympy."}


    # Iterate over increasing complexity (s = order + degree)
    # This ensures that we find the simplest relation first.
    max_complexity = max_order + max_degree
    for s in range(1, max_complexity + 1):
        for k in range(1, min(s, max_order) + 1):
            deg = s - k
            if deg > max_degree:
                continue

            num_coeffs_per_poly = deg + 1
            total_coeffs = (k + 1) * num_coeffs_per_poly

            # At least as many equations as unknowns are required.
            if N - k < total_coeffs:
                continue

            # Create symbolic variables for the polynomial coefficients.
            c_vars = sp.symbols(f'c_0:{total_coeffs}')
            
            polys = []
            for i in range(k + 1):
                poly_coeffs = c_vars[i * num_coeffs_per_poly : (i + 1) * num_coeffs_per_poly]
                polys.append(sum(poly_coeffs[j] * n**j for j in range(num_coeffs_per_poly)))

            # Form a system of linear equations.
            equations = []
            for ni in range(N - k):
                expr = sum(polys[i].subs(n, ni) * seq_rational[ni + i] for i in range(k + 1))
                equations.append(expr)
            
            # Solve the system.
            try:
                # linsolve may return an empty set if there are no solutions.
                solution_set = sp.linsolve(equations, c_vars)
                if not solution_set:
                    continue
            except Exception:
                continue # If the solution fails, skip.

            # Extract the solution. We only work with a single solution.
            # The case with an infinite number of solutions (underdetermined system) is skipped.
            if len(solution_set) == 1:
                sol = list(solution_set)[0]
                
                # Skip the trivial zero solution.
                if all(s == 0 for s in sol):
                    continue
                
                # --- Normalization to the minimal integer solution ---
                denominators = [s.q for s in sol if s != 0]
                lcm = sp.lcm(denominators) if denominators else 1
                int_sol = [s * lcm for s in sol]
                common_divisor = sp.gcd(int_sol) if int_sol else 1
                final_coeffs = [s / common_divisor for s in int_sol]

                # Collect the found polynomials.
                found_polys = []
                for i in range(k + 1):
                    poly_coeffs = final_coeffs[i * num_coeffs_per_poly : (i + 1) * num_coeffs_per_poly]
                    p = sum(poly_coeffs[j] * n**j for j in range(num_coeffs_per_poly))
                    found_polys.append(sp.simplify(p))

                # Final accuracy check.
                is_valid = all(
                    sp.simplify(sum(found_polys[i].subs(n, ni) * seq_rational[ni + i] for i in range(k + 1))) == 0
                    for ni in range(N - k)
                )

                if is_valid:
                    relation = sum(found_polys[i] * sp.Function('a')(n + i) for i in range(k + 1))
                    
                    return {
                        "order": k,
                        "degree": deg,
                        "polynomials_str": [str(p) for p in found_polys],
                        "relation_str": str(sp.Eq(relation, 0))
                    }

    return {"comment": "No P-recursive relations were found within the given limits."}

# ===================================================================
# 30. Nonlinear recurrence analysis
# ===================================================================
try:
    from sklearn.linear_model import Lasso
    from sklearn.exceptions import ConvergenceWarning
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn library not found. Nonlinear recurrence analysis will be unavailable.")
    print("Install it via 'pip install scikit-learn' for this feature.")
    class ConvergenceWarning(Warning): pass


def nonlinear_recurrence_analysis(seq: np.ndarray, max_order: int = 2, max_degree: int = 2, use_log: bool = False) -> Dict[str, Any]:
    """
    Finds a nonlinear recurrence relationship of the form:
        a(n) = F(a(n-1), a(n-2), ..., a(n-k))
    where F is a polynomial of degree up to max_degree.
    Uses Lasso (L1) regression to find the simplest (sparse) formula and avoid overfitting.
    """
    if not SKLEARN_AVAILABLE or not SYMPY_AVAILABLE:
        return {"comment": "scikit-learn or sympy library is not available."}
        
    n = len(seq)
    if n < 20:
        return {"comment": "Sequence too short for nonlinear analysis."}
        
    # Handle log transformation for multiplicative patterns.
    target_seq = seq
    if use_log:
        if np.any(seq <= 1e-9): # Allow zero, but not negative values.
            return {"comment": "Log mode is not applicable to sequences with negative values."}
        target_seq = np.log(seq + 1e-9) # Add epsilon for stability.

    best_model = None
    best_mse_ratio = float('inf')

    # Iterate through increasing order (memory depth).
    for k in range(1, max_order + 1):
        if n - k < 10: continue

        # Generate all monomial exponents.
        exponents = [
            e for e in itertools.product(range(max_degree + 1), repeat=k)
            if 0 < sum(e) <= max_degree
        ]
        
        # --- Overfitting Guard ---
        if n - k < len(exponents) * 2:
            continue

        # Build the feature matrix X and target vector y.
        X = []
        y = target_seq[k:]
        
        for i in range(k, n):
            prev_terms = target_seq[i-k:i][::-1]
            features = [np.prod(prev_terms ** exp) for exp in exponents]
            X.append(features)
        
        X = np.array(X)
        
        model = Lasso(alpha=1e-3, fit_intercept=True, max_iter=10000, tol=1e-4)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, y)
        
        preds = model.predict(X)
        mse = np.mean((y - preds)**2)
        
        target_variance = np.var(y)
        if target_variance < 1e-9: continue
        mse_ratio = mse / target_variance

        if mse_ratio < best_mse_ratio:
            best_mse_ratio = mse_ratio
            best_model = {
                "order": k,
                "degree": max_degree,
                "coefficients": np.concatenate(([model.intercept_], model.coef_)),
                "exponents": exponents,
                "mse_ratio": mse_ratio,
                "log_mode": use_log,
            }
            
    if best_model is None or best_model['mse_ratio'] > 0.01:
        return {"comment": "No simple nonlinear recurrence found with low error."}

    k = best_model['order']
    coeffs = best_model['coefficients']
    exponents = best_model['exponents']
    
    vars_sym = sp.symbols(f'x_1:{k+1}') 
    
    expr = sp.N(coeffs[0], 4) 
    for i, exp in enumerate(exponents):
        coeff = coeffs[i + 1]
        if abs(coeff) > 1e-5:
            term = sp.N(coeff, 4)
            for j in range(k):
                if exp[j] > 0:
                    term *= vars_sym[j]**exp[j]
            expr += term
    
    formula_str = str(expr)
    for i in range(1, k + 1):
        formula_str = formula_str.replace(f'x_{i}', f'a(n-{i})')
    best_model['expression_str'] = formula_str
    
    return best_model

# ===================================================================
# MAIN FUNCTION
# ===================================================================
def analyze_sequence_full(seq: List[float], is_sub_stream: bool = False) -> Dict[str, Any]:
    """
    The main analysis function.
    Creates two versions of the data (longdouble for precision and float64 for library compatibility) and passes the corresponding version to each analysis subfunction.
    """
    if not isinstance(seq, (np.ndarray, list)):
        return {"error": "Input sequence must be a list or numpy array"}
    if len(seq) == 0: 
        return {"error": "Sequence is empty"}

    # --- Create two versions of the sequence ---
    # 1. High precision version for critical calculations.
    try:
        seq_longdouble = np.array(seq, dtype=np.longdouble)
    except Exception as e:
        return {"error": f"Failed to create longdouble array: {e}"}

    # 2. Float64 version for compatibility with all external libraries.
    #    Suppress overflow warnings since `inf` is expected behavior.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seq_float64 = seq_longdouble.astype(np.float64)
    
    # Checking whether the transformation leads to infinity
    has_non_finite = not np.all(np.isfinite(seq_float64))

    meta = {}
    # Add the sequence length to the meta report so that 
    # the interpretation function can use it correctly.
    meta['sequence_length'] = len(seq)
    
    analysis_functions = {
        "statistical_moments": statistical_moments,
        "value_bounds": value_bounds,
        "dependency_model": dependency_model_analysis,
        "local_dynamics": local_dynamics,
        "cyclicity_and_seasonality": cyclicity_and_seasonality,
        "stationarity": stationarity,
        "autocorrelation_memory": autocorrelation_memory,
        "decomposition": decomposition_analysis,
        "volatility": volatility_analysis,
        "structural_breaks": structural_breaks,
        "anomalies": anomalies,
        "entropy_analysis": entropy_analysis,
        "complexity_analysis": complexity_analysis,
        "nonlinear_fractal_analysis": nonlinear_fractal_analysis,
        "spectral_analysis": spectral_analysis,
        "empirical_mode_decomposition": empirical_mode_decomposition,
        "symbolic_analysis": symbolic_analysis,
        "motif_analysis": motif_analysis,
        "state_segmentation": state_segmentation_analysis,
        "clustering_segmentation": clustering_segmentation,
        "number_theoretic_properties": number_theoretic_properties, 
        "ratio_analysis": analyze_ratios,
        "divisibility_properties": divisibility_properties,
        "simple_recurrence": simple_recurrence_analysis,
        "benford_law": benford_law_analysis,
        "p_recursive_relation": find_p_recursive_relation,
        "nonlinear_recurrence": nonlinear_recurrence_analysis,
        "advanced_number_theory": advanced_number_theory
    }

    # List of all functions that require float64 due to dependencies on external libraries
    float64_required_funcs = {
        "dependency_model",
        "cyclicity_and_seasonality",
        "stationarity",
        "autocorrelation_memory",
        "decomposition", "motif_analysis",
        "state_segmentation",
        "clustering_segmentation",
        "volatility",
        "structural_breaks",
        "spectral_analysis",
        "empirical_mode_decomposition",
        "symbolic_analysis",
        "nonlinear_fractal_analysis",
        "entropy_analysis",
        "complexity_analysis",
        "benford_law",
        "nonlinear_recurrence"
    }
    
    # Intelligent data version selection and feature skipping during overflow
    for name, func in analysis_functions.items():
        try:
            is_float64_func = name in float64_required_funcs
            
            # If a function requires float64 and we have infinities, we skip it.
            if is_float64_func and has_non_finite:
                meta[name] = {"comment": "Skipped due to non-finite values from numeric overflow."}
                continue

            # Select the appropriate data type for the function
            input_seq = seq_float64 if is_float64_func else seq_longdouble
            
            meta[name] = func(input_seq)
        except Exception as e:
            error_type = type(e).__name__
            meta[name] = {"error": f"Analysis step '{name}' failed with {error_type}: {str(e)}"}
            
    # Flow classification and analysis use float64 for generality
    if not has_non_finite:
        meta['pattern_class'] = classify_pattern(meta, seq_float64)
    else:
        meta['pattern_class'] = "Indeterminate (Numeric Overflow)"

    # This analysis uses the results of other modules, so it is called separately.
    try:
        meta['hybrid_model'] = hybrid_model_analysis(seq_longdouble, meta)
        meta['smart_hybrid_model'] = hybrid_model_analysis_v2(seq_longdouble)
        meta['local_formula'] = local_formula_analysis(seq_longdouble)
    except Exception as e:
        meta['hybrid_model'] = {"error": f"Hybrid model analysis failed: {e}"}
        meta['smart_hybrid_model'] = {"error": f"Smart hybrid model analysis failed: {e}"}
        meta['local_formula'] = {"error": f"Local formula analysis failed: {e}"}

    if not is_sub_stream:
        try:
            # Stream analysis can use any version, longdouble is more reliable
            stream_report = stream_structure_analysis(seq_longdouble, meta)
            meta['stream_structure'] = stream_report
            if stream_report.get('is_interleaved'):
                 meta['pattern_class'] = f"Interleaved ({stream_report.get('num_streams', '?')} streams)"
        except Exception as e:
            meta['stream_structure'] = {"error": f"Failed during stream analysis: {e}"}

    return meta

# ===================================================================
# Output function
# ===================================================================
def pretty_print_report(title: str, data: Dict[str, Any]):
    ENABLE_COLORS = False
    class Colors:
        def __init__(self, enabled: bool):
            if enabled: self.HEADER, self.BLUE, self.GREEN, self.YELLOW, self.FAIL, self.ENDC, self.BOLD = '\033[95m', '\033[94m', '\033[92m', '\033[93m', '\033[91m', '\033[0m', '\033[1m'
            else: self.HEADER, self.BLUE, self.GREEN, self.YELLOW, self.FAIL, self.ENDC, self.BOLD = ('',) * 7
    colors = Colors(ENABLE_COLORS)
    def format_value(value):
        if isinstance(value, str): return f"{colors.GREEN}'{' '.join(value.split())}'{colors.ENDC}"
        if isinstance(value, bool): return f"{colors.GREEN if value else colors.FAIL}{value}{colors.ENDC}"
        if isinstance(value, (int, float)): return f"{colors.YELLOW}{value:.4f}{colors.ENDC}" if isinstance(value, float) else f"{colors.YELLOW}{value}{colors.ENDC}"
        if value is None: return f"{colors.FAIL}None{colors.ENDC}"
        return str(value)
    def _print_recursive(data, indent_level=0):
        indent = '    ' * indent_level
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'sub_stream_analysis' and isinstance(value, list) and value:
                    print(f"{indent}{colors.BLUE}{key}{colors.ENDC}:")
                    for j, item in enumerate(value): print(f"{indent}    {colors.HEADER}--- Analysis for Stream #{j+1} ---{colors.ENDC}"); _print_recursive(item, indent_level + 2); print()
                    continue
                print(f"{indent}{colors.BLUE}{key}{colors.ENDC}: ", end="")
                if isinstance(value, dict) and value: print(); _print_recursive(value, indent_level + 1)
                elif isinstance(value, list) and value and isinstance(value[0], dict): print(); _print_recursive(value, indent_level + 1)
                else:
                    if isinstance(value, list): print(format_value(f"[{', '.join(map(str, value[:5]))}, ..., {value[-1]}] ({len(value)} items)" if len(value) > 10 else f"[{', '.join(map(str, value))}]"))
                    else: print(format_value(value))
        elif isinstance(data, list):
            for item in data: _print_recursive(item, indent_level)
    print(f"\n{colors.BOLD}{colors.HEADER}{'=' * 60}\n{title.center(60)}\n{'=' * 60}{colors.ENDC}"); _print_recursive(data)

import numpy as np
from typing import Dict, Any, List

def interpret_analysis_results(meta: Dict[str, Any], sequence_is_integer_like: bool) -> Dict[str, Any]:
    """
    Interprets the raw report from analyze_sequence_full, combining metrics
    to obtain high-level conclusions about the nature of the sequence.
    Version 2.6: Improved scoring logic for number-theoretic properties.

    Args:
        meta: Dictionary with results from analyze_sequence_full.
        sequence_is_integer_like: A flag indicating whether the sequence is integer-like.

    Returns:
        A dictionary with structured conclusions.
    """
    
    def get_val(path: str, default: Any = None) -> Any:
        keys = path.split('.')
        val = meta
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
            if val is None:
                return default
        return val

    conclusions = []
    
    # ===================================================================
    # HYPOTHESIS ANALYSIS BLOCKS
    # ===================================================================

    # --- 1.0 Hybrid Model (highest priority) ---
    hybrid_model = get_val('hybrid_model', {})
    if hybrid_model and 'mape' in hybrid_model and hybrid_model['mape'] < 10: # 10% error threshold
        score_hybrid = 10 # A very high score to override everything else
        details_hybrid = []
        mape = hybrid_model['mape']
        k = hybrid_model.get('k')
        pattern = hybrid_model.get('pattern')
        formulas = hybrid_model.get('base_formulas')
        
        if k and pattern is not None and formulas:
            rule_str = []
            for i in range(k):
                formula_idx = pattern[i]
                rule_str.append(f"  - if n % {k} == {i}: use formula '{formulas[formula_idx]}'")
            
            details_hybrid.append(f"A high-precision hybrid model was found (MAPE: {mape:.2f}%).")
            details_hybrid.append(f"The rules change cyclically with a period of k={k}:")
            details_hybrid.extend(rule_str)

        if details_hybrid:
            conclusions.append({"property": "Hybrid Cyclic Model", "score": score_hybrid, "details": details_hybrid})

    # --- 1.1 Interleaved Streams ---
    stream_info = get_val('stream_structure', {})
    if stream_info.get('is_interleaved'):
        sub_analyses = stream_info.get('sub_stream_analysis', [])
        if sub_analyses:
            sub_interpretations = [interpret_analysis_results(sub_meta, sequence_is_integer_like) for sub_meta in sub_analyses]
            return {
                "primary_conclusion": {
                    "property": "Interleaved Streams",
                    "details": [ f"Detected {stream_info.get('num_streams')} interleaved streams.", f"Reason: {stream_info.get('detection_reason')}."],
                    "confidence": "High"
                },
                "sub_stream_interpretations": sub_interpretations,
                "commentary": "The sequence consists of several subsequences. The analysis of each is presented below."
            }
            
    # --- 1.2 Linear Recurrence ---
    score_recurrence = 0
    details_recurrence = []
    if 'value_dependent' in get_val('dependency_model.dependency_type', ''):
        model = get_val('dependency_model.best_model', {})
        mse_ratio = model.get('performance', {}).get('mse_ratio', 1.0)
        order = model.get('order', 0)
        
        if order > 0 and mse_ratio < 0.01:
            score_recurrence += 2
            
            # General case: approximate recurrence
            coeffs = model.get('coeffs', [])
            if coeffs:
                ar_coeffs = coeffs[:-1]
                const = coeffs[-1]
                terms = [f"{c:+.3f}·a(n-{i+1})" for i, c in enumerate(ar_coeffs)]
                eq_str = " ".join(terms).lstrip("+ ")
                if abs(const) > 1e-5: eq_str += f" {const:+.3f}"
                details_recurrence.append(f"Detected an AR({order}) model with high precision (MSE ratio: {mse_ratio:.4f}). Equation: a(n) ≈ {eq_str}.")
                
            # Specific case: exact recurrence (BOOST SCORE and refine details)
            if get_val('dependency_model.dependency_type') == 'value_dependent (exact)':
                score_recurrence += 6 # Huge score boost
                bma = get_val('dependency_model.berlekamp_massey_analysis', {})
                if bma.get('formula_str'):
                    # Overwrite the approximate formula with the clean, exact one.
                    details_recurrence = [f"An exact integer linear recurrence was found: {bma['formula_str']}."]
            
            # Add analysis of roots for both cases
            try:
                # Use the clean integer coeffs if available, otherwise the float coeffs
                clean_coeffs = get_val('dependency_model.berlekamp_massey_analysis.coeffs')
                if clean_coeffs:
                    # BM coeffs are for C(x), need to flip sign for recurrence a(n) = ...
                    roots = np.roots([1] + [-c for c in clean_coeffs])
                else:
                    roots = np.roots([1] + [-c for c in model.get('coeffs', [])[:-1]])
                
                abs_roots = np.abs(roots)
                details_recurrence.append(f"Roots of the characteristic polynomial: {[f'{r:.2f}' for r in roots]}.")
                if np.any(abs_roots > 1.01): details_recurrence.append("Behavior: Unstable growth/decay (there are roots > 1 in magnitude).")
                elif np.any(np.isclose(abs_roots, 1.0)): details_recurrence.append("Behavior: Polynomial growth or oscillations (there are roots ≈ 1 in magnitude).")
                else: details_recurrence.append("Behavior: Damped (all roots < 1 in magnitude).")
            except Exception: pass
    
    if score_recurrence > 0:
        prop_name = "Exact Linear Recurrence" if get_val('dependency_model.dependency_type') == 'value_dependent (exact)' else "Linear Recurrence"
        conclusions.append({"property": prop_name, "score": score_recurrence, "details": details_recurrence})

    # --- 1.3 Exponential Growth/Decay ---
    score_expo = 0
    details_expo = []
    idx_model_path = 'dependency_model.best_model' if get_val('dependency_model.dependency_type') == 'index_dependent' else 'dependency_model.model_candidates.index_dependent'
    best_idx_model = get_val(idx_model_path, {})
        
    if best_idx_model.get('type') == 'exponential' and best_idx_model.get('performance', {}).get('mse_ratio', 1.0) < 0.05:
        score_expo += 2
        params = best_idx_model.get('params', {})
        details_expo.append(f"A high-precision approximation by an exponential function (y ≈ {params.get('a',0):.2f} * exp({params.get('b',0):.2f}*n)) was found with low error (MSE ratio: {best_idx_model.get('performance', {}).get('mse_ratio', 1.0):.4f}).")
    
    if get_val('local_dynamics.operator_type') == 'multiplicative' and get_val('local_dynamics.pattern_type') == 'monotonic':
        score_expo += 1; details_expo.append("Local dynamics are monotonic and have a multiplicative character.")
    
    sample_entropy = get_val('entropy_analysis.sample_entropy')
    if sample_entropy is not None and sample_entropy < 0.1:
        score_expo += 1; details_expo.append(f"Very low sample entropy ({sample_entropy:.3f}) indicates high predictability.")
    
    num_anomalies = get_val('anomalies.num_anomalies', 0)
    seq_len = get_val('sequence_length', 1) 
    
    if seq_len > 0 and num_anomalies > seq_len / 4 and get_val('local_dynamics.pattern_type') == 'monotonic':
         score_expo += 1; details_expo.append("The detection of anomalies at the end of the series is an artifact confirming accelerating growth.")

    if score_expo > 0:
        conclusions.append({"property": "Exponential Growth/Decay", "score": score_expo, "details": details_expo})
        
    # 1.3 Monotonic (Polynomial) Trend
    score_mono = 0
    details_mono = []
    if get_val('local_dynamics.pattern_type') == 'monotonic':
        score_mono += 1; details_mono.append("Local dynamics are monotonic (the series is consistently increasing or decreasing).")
        if get_val(idx_model_path + '.type') in ['linear', 'quadratic', 'cubic']:
             score_mono += 2; details_mono.append(f"A good approximation by a polynomial function was found (type: {get_val(idx_model_path + '.type')}).")
        if get_val('stationarity.conclusion') and 'Non-stationary' in get_val('stationarity.conclusion'):
             score_mono += 1; details_mono.append("Stationarity tests confirm the presence of a trend.")
        
        rolling_std = get_val('volatility.rolling_std_of_diffs')
        diff_var = get_val('local_dynamics.diff_variance')
        if rolling_std is not None and diff_var is not None and (rolling_std**2) < diff_var:
             score_mono += 1; details_mono.append("Low volatility of differences indicates a smooth trend.")
    
    if score_mono > 0:
        conclusions.append({"property": "Monotonic Trend", "score": score_mono, "details": details_mono})

    # ===================================================================
    # BLOCK 2: OSCILLATORY PROPERTIES
    # ===================================================================
    score_osc = 0
    details_osc = []
    cyc_strength = get_val('cyclicity_and_seasonality.strength', 0)
    period = get_val('cyclicity_and_seasonality.dominant_period', 0)
    if cyc_strength > 0.4 and period > 1: score_osc += 2; details_osc.append(f"Autocorrelation detected (strength: {cyc_strength:.3f}) with period T={period}.")
    if get_val('spectral_analysis.power_spectral_density.dominant_frequency', 0) > 0.05: # >0.05 to filter out trends
        score_osc += 1; details_osc.append("Spectral analysis shows a power peak at a non-zero frequency.")
    if get_val('decomposition.seasonality_strength', 0) > 0.5: score_osc += 1; details_osc.append("STL decomposition shows a strong seasonal component.")
    if get_val('local_dynamics.pattern_type') == 'oscillating': score_osc += 1; details_osc.append("Local dynamics are oscillatory in nature.")
        
    if score_osc > 0:
        prop_name = "Periodic Structure" if cyc_strength > 0.6 else "Oscillatory Process"
        conclusions.append({"property": prop_name, "score": score_osc, "details": details_osc})

    # ===================================================================
    # BLOCK 3: STOCHASTICITY AND COMPLEXITY
    # ===================================================================
    score_stochastic = 0
    details_stochastic = []
    nolds_available = get_val('nonlinear_fractal_analysis.comment') is None
    
    if get_val('dependency_model.dependency_type') == 'stochastic_or_complex': score_stochastic += 1; details_stochastic.append("No simple deterministic model was found.")
    if get_val('stationarity.conclusion') == 'Stationary': score_stochastic += 1; details_stochastic.append("The series is stationary, which is characteristic of random processes.")

    lyap_exp = get_val('nonlinear_fractal_analysis.lyapunov_exponent')
    if nolds_available:
        if lyap_exp is not None and lyap_exp > 0.05:
            score_stochastic += 2; details_stochastic.append(f"A positive Lyapunov exponent ({lyap_exp:.3f}) indicates chaotic dynamics.")
        else:
            hurst = get_val('nonlinear_fractal_analysis.hurst_exponent')
            corr_dim = get_val('nonlinear_fractal_analysis.correlation_dimension')
            if hurst is not None and not (0.4 < hurst < 0.6): details_stochastic.append(f"The Hurst exponent ({hurst:.2f}) is different from 0.5, which indicates the presence of memory.")
            if corr_dim is not None and corr_dim > 0: details_stochastic.append(f"A low fractal dimension ({corr_dim:.2f}) may indicate a hidden structure.")
    else:
        details_stochastic.append("Comment: The 'nolds' library is not available, analysis of chaos and fractal properties is limited.")
    
    sample_entropy = get_val('entropy_analysis.sample_entropy')
    if sample_entropy is not None and sample_entropy > 0.5: score_stochastic += 1; details_stochastic.append("High sample entropy indicates low predictability.")
        
    if score_stochastic > 0:
        prop_name = "Chaotic Dynamics" if lyap_exp and lyap_exp > 0.05 else "Stochastic/Complex Process"
        conclusions.append({"property": prop_name, "score": score_stochastic, "details": details_stochastic})
        
    # ===================================================================
    # BLOCK 4: SPECIFIC PROPERTIES (FIXED)
    # ===================================================================
    # 4.1 Number-Theoretic Properties
    if sequence_is_integer_like:
        score_num_theory = 0
        details_num_theory = []
        prime_perc = get_val('number_theoretic_properties.prime_percentage', 0)

        # FIX: Significantly increase the score for having almost 100% prime numbers.
        if prime_perc > 95.0:
            score_num_theory += 8 # Huge boost to make this property the primary conclusion
            details_num_theory.append(f"The sequence consists almost entirely ({prime_perc:.1f}%) of prime numbers.")
        elif prime_perc > 70.0:
            score_num_theory += 3
            details_num_theory.append(f"A significant portion of elements ({prime_perc:.1f}%) are prime numbers.")

        if get_val('divisibility_properties.all_divisible_by') is not None:
            score_num_theory += 1
            details_num_theory.append(f"Almost all elements are divisible by {get_val('divisibility_properties.all_divisible_by')}.")
        elif get_val('divisibility_properties.all_odd'):
            score_num_theory += 1
            details_num_theory.append("Almost all elements are odd.")

        if get_val('local_dynamics.structure_of_changes') == 'stochastic_or_complex' and prime_perc > 10:
            score_num_theory += 1
            details_num_theory.append("The irregular structure of changes is characteristic of prime number sequences.")
        
        if score_num_theory > 0:
            conclusions.append({"property": "Number-Theoretic Properties", "score": score_num_theory, "details": details_num_theory})
            
    # 4.2 Structural Breaks / Segmentation
    score_breaks = 0
    details_breaks = []
    if get_val('structural_breaks.break_points'): score_breaks += 2; details_breaks.append(f"Structural break points were detected: {get_val('structural_breaks.break_points')}.")
    if get_val('state_segmentation.state_means'): score_breaks += 1; details_breaks.append("HMM segmentation has identified several distinct states (regimes).")
    if get_val('clustering_segmentation.segment_counts_per_cluster'): score_breaks += 1; details_breaks.append("Clustering of time windows has identified several typical patterns.")
        
    if score_breaks > 0:
        conclusions.append({"property": "Presence of Structural Breaks / Regimes", "score": score_breaks, "details": details_breaks})

    # ===================================================================
    # FINAL PROCESSING (FIXED)
    # ===================================================================
    if not conclusions:
        return {
            "primary_conclusion": {"property": "Undefined/Complex Pattern", "details": ["Failed to uniquely classify the sequence."], "confidence": "Low"},
            "commentary": "The sequence may be too short, or its structure does not conform to standard models."
        }

    conclusions.sort(key=lambda x: x.get('score', 0), reverse=True)
    for c in conclusions: c['confidence'] = "High" if c['score'] >= 3 else ("Medium" if c['score'] == 2 else "Low")
    
    primary = conclusions[0]
    secondary = [c for c in conclusions[1:] if c['score'] > 0]
    commentary = ""

    # --- Contextual commentary and conflict resolution ---
    props = {c['property'] for c in conclusions}
    has_recurrence = any("Linear Recurrence" in p for p in props)
    has_exponential = "Exponential Growth/Decay" in props
    
    # FIX: Added a comment for number-theoretic cases
    if primary.get('property') == "Number-Theoretic Properties":
        commentary += (
            "COMMENTARY: A strong number-theoretic pattern was detected. "
            "The presence of a secondary 'Monotonic Trend' conclusion is expected, as sequences "
            "like the prime numbers have a well-defined asymptotic density (Prime Number Theorem), "
            "which curve-fitting models capture as a trend."
        )

    if "Hybrid Cyclic Model" in props and primary['property'] == "Hybrid Cyclic Model":
         commentary += (
            "\nCOMMENTARY: The detection of a precise hybrid model is the strongest result. "
            "This explains why other, simpler models (AR, exponential) might have shown "
            "good, but not perfect, results—they were merely approximating this complex cyclical behavior."
        )

    if has_exponential and primary.get('property', '').startswith("Exponential"):
        commentary += (
            "\nCOMMENTARY: All key metrics point to a dominant exponential trend. "
            "The failures of segmentation methods (HMM, Clustering) are expected, "
            "as the sequence contains only a single continuous growth regime that "
            "'overwhelms' more subtle structural features."
        )
        
    if has_recurrence and has_exponential:
        commentary += (
            "\nCOMMENTARY: Signs of both linear recurrence and exponential growth have been detected. "
            "This is typical for recurrences whose characteristic equation has a root with a magnitude greater than 1. "
            "The recurrence relation is a more precise and fundamental description of the sequence's law of generation."
        )

    return {
        "primary_conclusion": primary,
        "secondary_conclusions": secondary,
        "commentary": commentary.strip()
    }

if __name__ == "__main__":
   
    sequence = [
        0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9, 24, 8, 25, 43, 62, 42, 
        63, 41, 18, 42, 17, 43, 16, 44, 15, 45, 14, 46, 79, 113, 78, 114, 77, 39, 
        78, 38, 79, 37, 80, 36, 81, 35, 82, 34, 83, 33, 84, 32, 85, 31, 86, 30, 87, 
        29, 88, 28, 89, 27, 90, 26, 91
    ] # A005132 Recaman's sequence): a(0) = 0; for n > 0, a(n) = a(n-1) - n if nonnegative and not already in the sequence, otherwise a(n) = a(n-1) + n.
    sequence = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79] # A000040 The prime numbers.
    #sequence = [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000, 9261, 10648, 12167, 13824, 15625, 17576, 19683] # A000578 The cubes: a(n) = n^3.
    #sequence = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676] # A000290 The squares: a(n) = n^2.
    #sequence = [0, 1, 5, 19, 65, 211, 665, 2059, 6305, 19171, 58025, 175099, 527345, 1586131, 4766585, 14316139, 42981185, 129009091, 387158345, 1161737179, 3485735825, 10458256051] # A001047 a(n) = 3^n - 2^n.
    #sequence = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700] # A000108 Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!).
    #sequence = [1, 6, 90, 1680, 34650, 756756, 17153136, 399072960, 9465511770, 227873431500, 5550996791340, 136526995463040, 3384731762521200, 84478098072866400, 2120572665910728000, 53494979785374631680, 1355345464406015082330] # A006480 De Bruijn's S(3,n): (3n)!/(n!)^3.
    #sequence = [1, 0, 1, 0, 2, 0, 5, 0, 14, 0, 42, 0, 132, 0, 429, 0, 1430, 0, 4862, 0, 16796, 0, 58786, 0, 208012, 0, 742900, 0, 2674440, 0, 9694845, 0, 35357670] # A126120 Catalan numbers (A000108) interpolated with 0's.
    #sequence = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322] # A000032 Lucas numbers beginning at 2: L(n) = L(n-1) + L(n-2), L(0) = 2, L(1) = 1.
    try:
        # Check that the input is an iterable and not a single number
        if not hasattr(sequence, '__iter__'):
            raise TypeError("The input must be a list or array, not a single number.")
        
        # Convert to a NumPy array for further testing
        seq_np = np.array(sequence)
        
        if seq_np.size == 0:
            raise ValueError("The sequence cannot be empty.")

        # Launch the main analysis
        full_report = analyze_sequence_full(sequence)

        # --- Safe integer check ---
        is_integer_like = False
        if seq_np.dtype.kind in 'iu':
            is_integer_like = True
        elif seq_np.dtype.kind == 'f':
            # This check is now safe since seq_np is guaranteed to be an array.
            is_integer_like = np.allclose(seq_np, np.round(seq_np))
        
        # --- Obtaining and deriving an interpretation ---
        interpreted_report = interpret_analysis_results(full_report, is_integer_like)

        # Output of results
        pretty_print_report("Full technical report", full_report)
        pretty_print_report("Interpretation of analysis results", interpreted_report)
    except Exception as e:
        print("\n--- A CRITICAL ERROR OCCURRED ---")
        print(f"Error: {e}")
        print("Check the correctness of the input sequence.")
