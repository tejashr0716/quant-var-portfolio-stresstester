# Limitations & Boundary Conditions: Quant VaR Portfolio Stress Tester

## 1. Distributional Assumptions in Parametric VaR
- Parametric VaR assumes multi-asset multivariate normal return distributions. While accurate for small-to-moderate moves, real market tails exhibit leptokurtosis (fat tails) and skewness.
- To counter this, the engine provides Historical Simulation and Monte Carlo VaR alongside Expected Shortfall (CVaR).

## 2. Linear Correlation in Cholesky Factorization
- Monte Carlo paths use standard Cholesky decomposition ($L L^T = \Sigma$), assuming linear Pearson correlation.
- In severe market crashes, asset correlations often spike towards 1.0 (correlation breakdown). The engine addresses this through deterministic crisis stress tests (2008 Lehman, 2020 COVID, 2022 Stagflation).

## 3. Rate Limits & Free Ticker Data
- Historical prices are retrieved from Yahoo Finance's public chart API. For multi-thousand position portfolios, enterprise data providers (Bloomberg, Refinitiv, FactSet) should be plugged in via the modular `MarketDataClient`.

## 4. Constant Volatility Assumption
- The 1-day horizon calculations treat current 1-year sample covariance as stationary over the forward holding period. For multi-week horizons, GARCH(1,1) volatility forecasting can be incorporated.
