# SequenceAnalyzer
A Python toolkit for the automatic analysis and interpretation of numerical sequences.

## Purpose

This project is designed to address the following tasks:

*   **Pattern Identification:** Determines the type of a sequence (e.g., polynomial, exponential, recursive, cyclic, stochastic).
*   **Comprehensive Analysis:** Includes over 25 types of analysis covering statistical, structural, frequency-domain, and nonlinear properties.
*   **Result Interpretation:** Synthesizes calculated metrics into conclusions about the nature of the sequence.
*   **Complex Model Detection:** Identifies interleaved (multi-stream) and hybrid structures within the data.

The tool is suitable for analyzing time series and investigating mathematical sequences.

## Installation

For full functionality, install all dependencies with a single command:

```bash
pip install numpy scipy statsmodels PyWavelets ruptures nolds stumpy hmmlearn scikit-learn arch pyts EMD-signal tslearn
```

---

### 31+ Analysis Types

1.  **Local Dynamics:** Analysis of transitions between adjacent elements (velocity, acceleration, jerk).
2.  **Dependency Model:** Determines if a value depends on its index (`y = f(n)`) or on previous values (autoregression, `y(n) = f(y(n-1), ...)`).
3.  **Value Bounds:** Finds the minimum and maximum values of the sequence.
4.  **Cyclicity & Seasonality:** Searches for repeating patterns using autocorrelation and determines their period and strength.
5.  **Statistical Moments:** Calculates mean, variance, skewness, and kurtosis.
6.  **Stationarity:** Checks if statistical properties are constant over time using ADF and KPSS tests.
7.  **Autocorrelation and Memory:** Measures how much the current value depends on past values and estimates the "memory depth".
8.  **Structural Breaks:** Detects points where the statistical properties of the sequence change abruptly.
9.  **Anomalies:** Finds outliers by value and sharp breaks in shape (local "spikes").
10. **Entropy Analysis:** Measures complexity and predictability using Shannon and sample entropy.
11. **Stream Structure:** Checks for interleaved, independent subsequences (e.g., `a, b, c, a, b, c, ...`).
12. **Nonlinear/Fractal Analysis:** Assesses chaotic behavior (Lyapunov exponent) and long-term memory (Hurst exponent).
13. **Spectral Analysis:** Analyzes frequency components using Fourier and wavelet transforms.
14. **Pattern Classification:** Assigns a final, high-level classification (e.g., "Exponential Growth", "Stochastic Process") to the sequence based on all metrics.
15. **Decomposition:** Splits the sequence into trend, seasonal, and residual components using STL.
16. **Volatility Analysis:** Analyzes the magnitude of changes, including GARCH models for financial-type series.
17. **Motif Analysis:** Finds the most frequently recurring subsequences (patterns) using the Matrix Profile.
18. **State Segmentation:** Divides the sequence into segments corresponding to different regimes using Hidden Markov Models (HMM).
19. **Complexity Analysis:** Measures complexity based on data compressibility (Lempel-Ziv) and approximate entropy.
20. **Symbolic Analysis:** Converts the sequence into a symbolic representation (SAX) for abstract analysis.
21. **Empirical Mode Decomposition:** Decomposes a non-stationary signal into a set of intrinsic mode functions (IMFs).
22. **Clustering Segmentation:** Segments the sequence by clustering its subsequences based on their shape (DTW k-Means).
23. **Number-Theoretic Properties:** Analyzes basic properties of integers (e.g., percentage of primes).
24. **Divisibility Properties:** Checks for common divisors among the sequence elements.
25. **Ratio Analysis:** Studies the behavior of the ratio `a(n)/a(n-1)` to identify constant or functional growth factors.
26. **Simple Recurrence Search:** Searches for simple multiplicative relations of the form `a(n) ≈ C * a(n-k)`.
27. **Hybrid Model Analysis:** Searches for models where the generation rule changes cyclically based on the index `n % k`.
28. **Advanced Number-Theoretic Properties:** Checks for a robust greatest common divisor (GCD) and whether the sequence consists of perfect powers (squares, cubes, etc.), ignoring prefixes.
29. **Benford's Law Analysis:** Checks if the distribution of first significant digits conforms to Benford's Law, which is often used to distinguish natural from artificial data.
30. **P-Recursive (Holonomic) Analysis:** Searches for exact recurrence relations with polynomial coefficients, e.g., `(n+1)*a(n+1) - (2*n+1)*a(n) = 0`.
31. **Nonlinear Recurrence Analysis:** Attempts to find polynomial relationships between a term and its predecessors, e.g., `a(n) = a(n-1)^2 - a(n-2)`.
    
## Usage Example

```python
# Import the main functions
from SequenceAnalyzer import analyze_sequence_full, interpret_analysis_results, pretty_print_report
import numpy as np

# 1. Define the sequence to analyze
# Example: a(n) = n^3 (the cubes)
sequence = [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]

# 2. Run the full analysis
full_report = analyze_sequence_full(sequence)

# 3. Check if the sequence is integer-like
seq_np = np.array(sequence)
is_integer_like = np.allclose(seq_np, np.round(seq_np))

# 4. Run the interpreter to get conclusions
interpreted_report = interpret_analysis_results(full_report, is_integer_like)

# 5. Print the results
pretty_print_report("Full Technical Report", full_report)
pretty_print_report("Interpretation of Analysis Results", interpreted_report)
```

---

The analyzer uses a variety of methods, each with its own data requirements. Generally, the more complex the pattern a method seeks, the more data it needs to produce a reliable result.

Here is a brief guide to the recommended minimum sequence lengths for different types of analysis:

*   **1. Foundational Metrics (5-15+ terms):**
    *   **Functions:** `local_dynamics (5)`, `statistical_moments (2)`, `dependency_model (polynomials, 5-10)`, `anomalies (4)`.
    *   **Why:** This is enough data to calculate basic moments (mean, variance), derivatives (velocity, acceleration), and fit simple curves. These concepts are not meaningful on fewer points.

*   **2. Core Time Series Analysis (20-40+ terms):**
    *   **Functions:** `stationarity (20)`, `cyclicity_and_seasonality (approx. 20-30)`, `entropy_analysis (20)`, `spectral_analysis (20)`, `structural_breaks (20)`, `complexity_analysis (20)`, `stream_structure_analysis (30)`, `state_segmentation (HMM, ~36)`.
    *   **Why:** These methods need sufficient data to establish a "statistical regime." Stationarity tests must distinguish trends from noise, cycle detection needs to see at least one or two full periods, and HMM needs enough samples to learn the characteristics of each hidden state.

*   **3. Advanced & Structural Analysis (50+ terms):**
    *   **Functions:** `nonlinear_fractal_analysis (Hurst/DFA, 50)`, `volatility_analysis (GARCH, 50)`, `empirical_mode_decomposition (50)`, `hybrid_model_analysis (50)`.
    *   **Why:** These methods model more complex properties. GARCH analyzes the volatility of returns, fractal analysis investigates self-similarity and scaling properties, and EMD decomposes the signal into nested oscillatory modes—all of which require a longer data history to be reliable.

*   **4. Highly Demanding (Chaos & Attractor) Analysis (100+ terms):**
    *   **Functions:** `nonlinear_fractal_analysis (Correlation Dimension/Lyapunov Exponent, 100)`.
    *   **Why:** These methods attempt to reconstruct a system's attractor in phase space from a single time series. This is an extremely data-hungry task, and results on shorter sequences would be statistically insignificant.

In general, longer sequences yield more reliable and insightful results, especially for the advanced methods.

---

*   feklindn@gmail.com 

---
