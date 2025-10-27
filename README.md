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

## 26+ Analysis Types

1.  **Local Dynamics:** Analysis of transitions between adjacent elements.
2.  **Dependency Model:** Finds if a value depends on its index (`f(n)`) or on previous values (autoregression).
3.  **Value Bounds:** Finds the minimum and maximum values.
4.  **Cyclicity:** Searches for repeating patterns and determines their period.
5.  **Statistical Moments:** Calculates mean, variance, skewness, and kurtosis.
6.  **Stationarity:** Checks if statistical properties are constant over time.
7.  **Autocorrelation and Memory:** Measures how much the current value depends on past values.
8.  **Structural Breaks:** Detects points where the sequence's behavior changes abruptly.
9.  **Anomalies:** Finds outliers and sharp structural breaks.
10. **Entropy:** Measures the complexity and predictability of the sequence.
11. **Stream Structure:** Checks for interleaved, independent subsequences.
12. **Nonlinear/Fractal Analysis:** Assesses chaotic behavior (Lyapunov exponent) and long-term memory (Hurst exponent).
13. **Spectral Analysis:** Analyzes frequency components using Fourier and wavelet transforms.
14. **Pattern Classification:** Assigns a final classification to the sequence based on all metrics.
15. **Decomposition:** Splits the sequence into trend, seasonal, and residual components.
16. **Volatility:** Analyzes the magnitude of changes (GARCH models).
17. **Motif Analysis:** Finds the most frequently recurring subsequences.
18. **State Segmentation:** Divides the sequence into segments corresponding to different regimes (HMM).
19. **Complexity Analysis:** Measures complexity based on data compressibility (Lempel-Ziv).
20. **Symbolic Analysis:** Converts the sequence into a symbolic representation (SAX) for abstract analysis.
21. **Empirical Mode Decomposition:** Decomposes a signal into a set of oscillatory components.
22. **Clustering Segmentation:** Segments the sequence by clustering its subsequences based on shape.
23. **Number-Theoretic Properties:** Analyzes properties of integers (e.g., percentage of primes).
24. **Divisibility Properties:** Checks for common divisors.
25. **Ratio Analysis:** Studies the behavior of the ratio `a(n)/a(n-1)`.
26. **Simple Recurrence Search:** Looks for relations of the form `a(n) ≈ C * a(n-k)`.
27. **Hybrid Model Analysis:** Searches for models where the generation rule changes cyclically based on the index `n`.

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

*   feklindn@gmail.com 

---
