import time
import math
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
from src.models import (
    Position,
    PortfolioRequest,
    RiskMetric,
    VaRReport,
    StressScenarioResult,
    StressTestResponse,
    BacktestResponse
)


class RiskEngine:
    """Quantitative risk engine computing Parametric, Historical, Monte Carlo VaR, Stress Tests, and Backtesting."""

    def __init__(self):
        pass

    def calculate_var_suite(
        self,
        returns_matrix: np.ndarray,
        tickers: List[str],
        positions: List[Position],
        portfolio_value: float = 1_000_000.0,
        confidence_level: float = 0.95,
        simulations: int = 10000
    ) -> VaRReport:
        """
        Calculates Parametric, Historical, and Monte Carlo VaR/CVaR.
        """
        start_t = time.perf_counter()

        # Map positions to weights vector aligned with tickers list
        ticker_map = {t: i for i, t in enumerate(tickers)}
        raw_weights = np.zeros(len(tickers))
        for p in positions:
            if p.ticker in ticker_map:
                raw_weights[ticker_map[p.ticker]] = p.weight

        # Normalize weights so sum is 1.0
        w = raw_weights / np.sum(raw_weights)

        # Asset returns statistics
        cov_matrix = np.atleast_2d(np.cov(returns_matrix, rowvar=False))
        mu = np.atleast_1d(np.mean(returns_matrix, axis=0))

        # Ensure covariance matrix symmetry
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

        # Portfolio daily volatility & annualized volatility
        port_variance = float(w.T @ cov_matrix @ w)
        port_sigma = math.sqrt(max(0.0, port_variance))
        annualized_vol = port_sigma * math.sqrt(252)

        # -------------------------------------------------------------
        # 1. Parametric (Variance-Covariance) VaR & CVaR
        # -------------------------------------------------------------
        z_score = float(stats.norm.ppf(confidence_level))
        param_var_pct = z_score * port_sigma
        param_var_amt = param_var_pct * portfolio_value

        # Parametric Expected Shortfall (CVaR)
        norm_pdf = float(stats.norm.pdf(z_score))
        param_cvar_pct = (norm_pdf / (1.0 - confidence_level)) * port_sigma
        param_cvar_amt = param_cvar_pct * portfolio_value

        parametric_metric = RiskMetric(
            confidence_level=confidence_level,
            var_amount=round(param_var_amt, 2),
            var_pct=round(param_var_pct, 6),
            cvar_amount=round(param_cvar_amt, 2),
            cvar_pct=round(param_cvar_pct, 6)
        )

        # -------------------------------------------------------------
        # 2. Historical Simulation VaR & CVaR
        # -------------------------------------------------------------
        # Daily historical portfolio return series: R_p = returns_matrix @ w
        hist_port_returns = returns_matrix @ w
        cutoff_pct = (1.0 - confidence_level) * 100.0
        hist_var_pct = float(-np.percentile(hist_port_returns, cutoff_pct))
        hist_var_amt = hist_var_pct * portfolio_value

        tail_losses = hist_port_returns[hist_port_returns <= -hist_var_pct]
        if len(tail_losses) > 0:
            hist_cvar_pct = float(-np.mean(tail_losses))
        else:
            hist_cvar_pct = hist_var_pct
        hist_cvar_amt = hist_cvar_pct * portfolio_value

        historical_metric = RiskMetric(
            confidence_level=confidence_level,
            var_amount=round(hist_var_amt, 2),
            var_pct=round(hist_var_pct, 6),
            cvar_amount=round(hist_cvar_amt, 2),
            cvar_pct=round(hist_cvar_pct, 6)
        )

        # -------------------------------------------------------------
        # 3. Monte Carlo Simulation VaR & CVaR (Cholesky Decomposition)
        # -------------------------------------------------------------
        # Regularize cov_matrix if near-singular
        min_eig = np.min(np.linalg.eigvals(cov_matrix))
        if min_eig <= 0:
            cov_matrix += np.eye(len(tickers)) * (abs(min_eig) + 1e-6)

        chol_l = np.linalg.cholesky(cov_matrix)
        
        # Draw uncorrelated standard normal random paths
        z_random = np.random.standard_normal((len(tickers), simulations))
        
        # Correlated simulated asset returns: R_sim = mu + L * Z
        sim_asset_returns = mu[:, np.newaxis] + (chol_l @ z_random)
        
        # Simulated portfolio returns: R_p,sim = w^T * R_sim
        sim_port_returns = w @ sim_asset_returns
        
        mc_var_pct = float(-np.percentile(sim_port_returns, cutoff_pct))
        mc_var_amt = mc_var_pct * portfolio_value

        mc_tail_losses = sim_port_returns[sim_port_returns <= -mc_var_pct]
        if len(mc_tail_losses) > 0:
            mc_cvar_pct = float(-np.mean(mc_tail_losses))
        else:
            mc_cvar_pct = mc_var_pct
        mc_cvar_amt = mc_cvar_pct * portfolio_value

        monte_carlo_metric = RiskMetric(
            confidence_level=confidence_level,
            var_amount=round(mc_var_amt, 2),
            var_pct=round(mc_var_pct, 6),
            cvar_amount=round(mc_cvar_amt, 2),
            cvar_pct=round(mc_cvar_pct, 6)
        )

        calc_ms = round((time.perf_counter() - start_t) * 1000.0, 3)

        return VaRReport(
            portfolio_value=portfolio_value,
            positions=positions,
            annualized_volatility=round(annualized_vol, 6),
            parametric=parametric_metric,
            historical=historical_metric,
            monte_carlo=monte_carlo_metric,
            calculation_time_ms=calc_ms
        )

    def run_stress_tests(
        self,
        positions: List[Position],
        portfolio_value: float = 1_000_000.0
    ) -> StressTestResponse:
        """Execute historical macro stress scenarios on portfolio."""
        scenarios_def = [
            {
                "name": "2008 Lehman Global Financial Crisis",
                "desc": "Severe liquidity freeze, global equity collapse, flight to quality US Treasuries and Gold.",
                "shocks": {
                    "SPY": -0.38, "AAPL": -0.32, "MSFT": -0.30, "TLT": +0.14, "GLD": +0.06,
                    "DEFAULT_EQUITY": -0.35, "DEFAULT_BOND": +0.10, "DEFAULT_COMMODITY": +0.05
                }
            },
            {
                "name": "2020 COVID-19 Liquidity Shock",
                "desc": "Abrupt global shutdown, rapid equity drawdown, treasury rally.",
                "shocks": {
                    "SPY": -0.28, "AAPL": -0.19, "MSFT": -0.18, "TLT": +0.09, "GLD": -0.04,
                    "DEFAULT_EQUITY": -0.25, "DEFAULT_BOND": +0.06, "DEFAULT_COMMODITY": -0.10
                }
            },
            {
                "name": "2022 Stagflation & Rate Hike Regime",
                "desc": "Simultaneous equity and bond drawdown driven by 500bps Fed rate hikes.",
                "shocks": {
                    "SPY": -0.19, "AAPL": -0.26, "MSFT": -0.28, "TLT": -0.22, "GLD": -0.01,
                    "DEFAULT_EQUITY": -0.20, "DEFAULT_BOND": -0.20, "DEFAULT_COMMODITY": +0.15
                }
            }
        ]

        total_w = sum(p.weight for p in positions)
        results: List[StressScenarioResult] = []

        for s in scenarios_def:
            scenario_shocks: Dict[str, float] = {}
            weighted_ret = 0.0

            for p in positions:
                w_norm = p.weight / total_w
                shock = s["shocks"].get(p.ticker)
                if shock is None:
                    shock = s["shocks"]["DEFAULT_EQUITY"]
                scenario_shocks[p.ticker] = shock
                weighted_ret += w_norm * shock

            pnl = weighted_ret * portfolio_value
            results.append(StressScenarioResult(
                scenario_name=s["name"],
                description=s["desc"],
                asset_shocks=scenario_shocks,
                portfolio_pnl=round(pnl, 2),
                portfolio_return_pct=round(weighted_ret, 6)
            ))

        worst = min(results, key=lambda r: r.portfolio_pnl)

        return StressTestResponse(
            portfolio_value=portfolio_value,
            scenarios=results,
            worst_scenario=worst.scenario_name,
            max_loss_amount=round(abs(worst.portfolio_pnl), 2),
            max_loss_pct=round(abs(worst.portfolio_return_pct), 6)
        )

    def run_kupiec_backtest(
        self,
        returns_matrix: np.ndarray,
        tickers: List[str],
        positions: List[Position],
        confidence_level: float = 0.95,
        portfolio_value: float = 1_000_000.0
    ) -> BacktestResponse:
        """
        Executes Kupiec Proportion of Failures (POF) Likelihood-Ratio backtest.
        """
        ticker_map = {t: i for i, t in enumerate(tickers)}
        raw_weights = np.zeros(len(tickers))
        for p in positions:
            if p.ticker in ticker_map:
                raw_weights[ticker_map[p.ticker]] = p.weight

        w = raw_weights / np.sum(raw_weights)
        hist_returns = returns_matrix @ w
        n = len(hist_returns)

        # Reference Parametric VaR threshold at confidence level
        cov_matrix = np.atleast_2d(np.cov(returns_matrix, rowvar=False))
        port_variance = float(w.T @ cov_matrix @ w)
        port_sigma = math.sqrt(max(1e-8, port_variance))
        z_score = float(stats.norm.ppf(confidence_level))
        var_threshold = z_score * port_sigma

        # Count exceptions: days where loss exceeded VaR threshold
        exceptions = int(np.sum(hist_returns <= -var_threshold))
        expected_exceptions = n * (1.0 - confidence_level)
        exception_rate = exceptions / n if n > 0 else 0.0

        p = 1.0 - confidence_level
        p_hat = exception_rate

        # Kupiec Likelihood Ratio:
        # LR = -2 * ln( (1 - p)^(N - x) * p^x / ( (1 - p_hat)^(N - x) * p_hat^x ) )
        if exceptions == 0 or exceptions == n or p_hat <= 0:
            lr_stat = 0.0
            p_val = 1.0
        else:
            num = ((1.0 - p) ** (n - exceptions)) * (p ** exceptions)
            den = ((1.0 - p_hat) ** (n - exceptions)) * (p_hat ** exceptions)
            if den > 0 and num > 0:
                lr_stat = -2.0 * math.log(num / den)
            else:
                lr_stat = 0.0
            p_val = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

        # Model accepted if p_value >= 0.05 (fail to reject null)
        accepted = p_val >= 0.05
        verdict = "ACCEPT_MODEL" if accepted else "REJECT_MODEL"

        return BacktestResponse(
            total_days=n,
            confidence_level=confidence_level,
            exceptions_count=exceptions,
            expected_exceptions=round(expected_exceptions, 2),
            exception_rate=round(exception_rate, 4),
            kupiec_lr_statistic=round(lr_stat, 4),
            p_value=round(p_val, 4),
            model_accepted=accepted,
            verdict=verdict
        )


risk_engine = RiskEngine()
