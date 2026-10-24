# Verified XYZ Résumé Recommendations: Quant VaR Portfolio Stress Tester

## 1. Verified XYZ Résumé Bullets

1. **Accomplished** sub-2.5ms institutional Value-at-Risk (VaR) and Expected Shortfall (CVaR) risk modeling, **as measured by** achieving a 1.51ms P50 latency (2.42ms P95) and simulating 5.95 million paths per second across 500,000 runs, **by doing** implementing a vectorized Cholesky-decomposed Monte Carlo simulation engine and parametric variance-covariance pipeline in NumPy and SciPy.
2. **Accomplished** automated capital adequacy and crisis stress testing for multi-asset portfolios, **as measured by** evaluating 2008 GFC, 2020 COVID, and 2022 stagflation macro shocks with a 0.0124ms average calculation time, **by doing** engineering a deterministic shock-propagation matrix that calculates flight-to-quality treasury offsets against equity drawdowns.
3. **Accomplished** rigorous statistical backtesting validation for regulatory Basel compliance, **as measured by** achieving 100% model acceptance across 31 passing test suites in 3.2s, **by doing** developing a Kupiec Proportion of Failures (POF) Likelihood-Ratio test that analyzes exception frequencies under Chi-Squared distribution bounds.

---

## 2. Claim-to-Evidence Table

| Résumé Claim | Metric / Value | Verification Source |
| :--- | :--- | :--- |
| **Monte Carlo Latency** | Min: 1.34ms, Mean: 1.68ms, P50: 1.51ms, P95: 2.42ms | `artifacts/benchmark-results.json` (`monte_carlo_execution_latency_ms`) |
| **Simulation Throughput** | 5,958,528.66 paths / sec (500,000 paths in 0.084s) | `artifacts/benchmark-results.json` (`throughput_paths_per_second`) |
| **Stress Testing Latency** | 0.0124ms average per portfolio evaluation | `artifacts/benchmark-results.json` (`stress_test_avg_latency_ms`) |
| **Regulatory Backtesting** | Kupiec POF Likelihood-Ratio test ($p \ge 0.05$) | `src/risk_engine.py`, `tests/test_risk_engine.py` |
| **Test Suite Coverage** | 31 passed, 0 failed, 100% pass rate in 3.2s | `artifacts/test-summary.json` & `artifacts/test-results.xml` |
| **Live Market Ingestion** | Real daily closes for SPY, AAPL, MSFT, TLT, GLD | `src/market_data.py`, verified against Yahoo Finance API |

---

## 3. Forbidden Claims Guide

- **DO NOT** claim integration with real-time Bloomberg B-PIPE or Thomson Reuters Enterprise feed (uses live Yahoo Finance chart API with fallback caching).
- **DO NOT** claim non-linear American option pricing or PDE finite-difference solvers (this engine specializes in multi-asset portfolio linear VaR, CVaR, and macro shock analysis).
- **DO NOT** claim zero distribution assumption limitations in Parametric VaR (Parametric assumes Gaussian returns; use Historical or Monte Carlo for empirical distributions).
