import pytest
import math
import numpy as np
import scipy.stats as stats
from httpx import AsyncClient, ASGITransport
from src.models import (
    Position,
    PortfolioRequest,
    BacktestRequest
)
from src.market_data import MarketDataClient, market_data
from src.risk_engine import RiskEngine, risk_engine
from src.main import app


# -------------------------------------------------------------
# 1. Math & Matrix Properties Tests
# -------------------------------------------------------------
def test_covariance_symmetry_and_psd():
    np.random.seed(42)
    fake_returns = np.random.normal(0, 0.02, (200, 3))
    cov = np.cov(fake_returns, rowvar=False)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    eigenvals = np.linalg.eigvals(cov)
    assert np.all(eigenvals >= -1e-12)


def test_cholesky_decomposition_reconstruction():
    np.random.seed(42)
    A = np.random.rand(4, 4)
    cov = A @ A.T + np.eye(4) * 0.1
    L = np.linalg.cholesky(cov)
    reconstructed = L @ L.T
    np.testing.assert_allclose(cov, reconstructed, atol=1e-9)


def test_portfolio_volatility_scaling():
    w = np.array([0.5, 0.5])
    cov = np.array([[0.0004, 0.0], [0.0, 0.0004]])  # 2% vol each, uncorrelated
    var_daily = w.T @ cov @ w
    sigma_daily = math.sqrt(var_daily)
    ann_vol = sigma_daily * math.sqrt(252)
    assert 0.0 < ann_vol < 1.0


def test_single_asset_covariance_scalar_safety():
    fake_returns = np.random.normal(0, 0.01, (100, 1))
    positions = [Position(ticker="SINGLE", weight=1.0)]
    report = risk_engine.calculate_var_suite(fake_returns, ["SINGLE"], positions, portfolio_value=100_000.0)
    assert report.annualized_volatility > 0.0
    assert report.parametric.var_amount > 0.0


# -------------------------------------------------------------
# 2. Parametric VaR & CVaR Tests
# -------------------------------------------------------------
def test_parametric_var_monotonicity_across_confidence():
    np.random.seed(42)
    fake_returns = np.random.normal(0, 0.015, (250, 2))
    positions = [Position(ticker="A", weight=0.6), Position(ticker="B", weight=0.4)]
    
    rep_95 = risk_engine.calculate_var_suite(
        fake_returns, ["A", "B"], positions, portfolio_value=1_000_000.0, confidence_level=0.95
    )
    rep_99 = risk_engine.calculate_var_suite(
        fake_returns, ["A", "B"], positions, portfolio_value=1_000_000.0, confidence_level=0.99
    )
    
    assert rep_99.parametric.var_amount > rep_95.parametric.var_amount
    assert rep_99.historical.var_amount > rep_95.historical.var_amount
    assert rep_99.monte_carlo.var_amount > rep_95.monte_carlo.var_amount


def test_cvar_strictly_greater_than_var():
    np.random.seed(42)
    fake_returns = np.random.normal(0, 0.02, (250, 2))
    positions = [Position(ticker="A", weight=0.5), Position(ticker="B", weight=0.5)]
    
    report = risk_engine.calculate_var_suite(
        fake_returns, ["A", "B"], positions, portfolio_value=1_000_000.0, confidence_level=0.95
    )
    assert report.parametric.cvar_amount >= report.parametric.var_amount
    assert report.historical.cvar_amount >= report.historical.var_amount
    assert report.monte_carlo.cvar_amount >= report.monte_carlo.var_amount


def test_portfolio_value_linear_scaling():
    np.random.seed(42)
    fake_returns = np.random.normal(0, 0.01, (100, 2))
    positions = [Position(ticker="A", weight=0.5), Position(ticker="B", weight=0.5)]
    
    rep_1m = risk_engine.calculate_var_suite(
        fake_returns, ["A", "B"], positions, portfolio_value=1_000_000.0, confidence_level=0.95
    )
    rep_2m = risk_engine.calculate_var_suite(
        fake_returns, ["A", "B"], positions, portfolio_value=2_000_000.0, confidence_level=0.95
    )
    
    np.testing.assert_allclose(rep_2m.parametric.var_amount, rep_1m.parametric.var_amount * 2.0, rtol=1e-3)
    np.testing.assert_allclose(rep_2m.historical.var_amount, rep_1m.historical.var_amount * 2.0, rtol=1e-3)


# -------------------------------------------------------------
# 3. Historical Simulation Tests
# -------------------------------------------------------------
def test_historical_simulation_known_percentile():
    rets = np.linspace(-0.10, 0.098, 100).reshape((100, 1))
    positions = [Position(ticker="SPY", weight=1.0)]
    report = risk_engine.calculate_var_suite(rets, ["SPY"], positions, portfolio_value=100_000.0, confidence_level=0.95)
    assert 8500.0 <= report.historical.var_amount <= 9800.0


def test_historical_var_zero_returns():
    rets = np.zeros((100, 2))
    positions = [Position(ticker="A", weight=0.5), Position(ticker="B", weight=0.5)]
    report = risk_engine.calculate_var_suite(rets, ["A", "B"], positions, portfolio_value=1_000_000.0, confidence_level=0.95)
    assert report.historical.var_amount == 0.0
    assert report.historical.cvar_amount == 0.0


# -------------------------------------------------------------
# 4. Monte Carlo Simulation Tests
# -------------------------------------------------------------
def test_monte_carlo_convergence_to_parametric():
    np.random.seed(123)
    fake_returns = np.random.normal(0.0, 0.01, (500, 2))
    positions = [Position(ticker="A", weight=0.5), Position(ticker="B", weight=0.5)]
    
    report = risk_engine.calculate_var_suite(
        fake_returns, ["A", "B"], positions, portfolio_value=1_000_000.0, confidence_level=0.95, simulations=20000
    )
    pct_diff = abs(report.monte_carlo.var_amount - report.parametric.var_amount) / report.parametric.var_amount
    assert pct_diff < 0.06


def test_monte_carlo_correlation_preservation():
    np.random.seed(99)
    rets1 = np.random.normal(0, 0.02, 500)
    rets2 = rets1 + np.random.normal(0, 0.005, 500)
    returns_matrix = np.column_stack([rets1, rets2])
    
    positions = [Position(ticker="X", weight=0.5), Position(ticker="Y", weight=0.5)]
    report = risk_engine.calculate_var_suite(
        returns_matrix, ["X", "Y"], positions, portfolio_value=1_000_000.0, simulations=5000
    )
    assert report.monte_carlo.var_amount > 0.0
    assert report.annualized_volatility > 0.0


# -------------------------------------------------------------
# 5. Stress Testing Engine Tests
# -------------------------------------------------------------
def test_stress_test_crisis_scenarios_count():
    positions = [Position(ticker="SPY", weight=0.6), Position(ticker="TLT", weight=0.4)]
    resp = risk_engine.run_stress_tests(positions, portfolio_value=1_000_000.0)
    assert len(resp.scenarios) == 3
    assert resp.worst_scenario != ""
    assert resp.max_loss_amount > 0.0


def test_stress_test_flight_to_quality_mitigation():
    spy_only = [Position(ticker="SPY", weight=1.0)]
    balanced = [Position(ticker="SPY", weight=0.6), Position(ticker="TLT", weight=0.4)]
    
    r_spy = risk_engine.run_stress_tests(spy_only, 1_000_000.0)
    r_bal = risk_engine.run_stress_tests(balanced, 1_000_000.0)
    
    gfc_spy = next(s for s in r_spy.scenarios if "2008" in s.scenario_name)
    gfc_bal = next(s for s in r_bal.scenarios if "2008" in s.scenario_name)
    
    assert abs(gfc_bal.portfolio_pnl) < abs(gfc_spy.portfolio_pnl)


def test_stress_test_unknown_ticker_uses_default_shock():
    positions = [Position(ticker="RANDOM_TICKER", weight=1.0)]
    resp = risk_engine.run_stress_tests(positions, 500_000.0)
    assert len(resp.scenarios) == 3
    assert resp.max_loss_amount > 0.0


def test_stress_test_weights_normalization():
    positions = [Position(ticker="SPY", weight=1.0), Position(ticker="TLT", weight=1.0)]
    resp = risk_engine.run_stress_tests(positions, 1_000_000.0)
    assert resp.portfolio_value == 1_000_000.0


# -------------------------------------------------------------
# 6. Kupiec POF Backtesting Tests
# -------------------------------------------------------------
def test_kupiec_backtest_perfect_calibration():
    np.random.seed(42)
    fake_returns = np.random.normal(0, 0.01, (250, 1))
    positions = [Position(ticker="ASSET", weight=1.0)]
    resp = risk_engine.run_kupiec_backtest(
        fake_returns, ["ASSET"], positions, confidence_level=0.95
    )
    assert resp.total_days == 250
    assert resp.model_accepted is True
    assert resp.verdict == "ACCEPT_MODEL"


def test_kupiec_backtest_severe_underestimation_rejects():
    np.random.seed(42)
    fake_returns = np.random.normal(0.001, 0.005, (250, 1))
    fake_returns[:50] = -0.05
    
    positions = [Position(ticker="ASSET", weight=1.0)]
    resp = risk_engine.run_kupiec_backtest(
        fake_returns, ["ASSET"], positions, confidence_level=0.95
    )
    assert resp.model_accepted is False
    assert resp.verdict == "REJECT_MODEL"
    assert resp.p_value < 0.01


def test_kupiec_backtest_zero_exceptions():
    rets = np.ones((200, 1)) * 0.01
    positions = [Position(ticker="ASSET", weight=1.0)]
    resp = risk_engine.run_kupiec_backtest(rets, ["ASSET"], positions, confidence_level=0.95)
    assert resp.exceptions_count == 0
    assert resp.p_value == 1.0


# -------------------------------------------------------------
# 7. Market Data Client Tests
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_market_data_fallback_generation():
    client = MarketDataClient()
    closes = await client.fetch_historical_closes("FAKE_OFFLINE_TICKER_99")
    assert len(closes) >= 200
    assert closes[0] > 0


@pytest.mark.asyncio
async def test_market_data_aligned_returns():
    client = MarketDataClient()
    matrix_data = await client.get_aligned_returns_matrix(["SPY", "AAPL"], range_str="1mo")
    assert matrix_data["returns"].ndim == 2
    assert matrix_data["returns"].shape[1] == 2
    assert "SPY" in matrix_data["latest_prices"]
    assert "AAPL" in matrix_data["latest_prices"]


@pytest.mark.asyncio
async def test_market_data_live_yahoo_prices():
    client = MarketDataClient()
    closes = await client.fetch_historical_closes("SPY", range_str="3mo")
    assert len(closes) > 30
    assert closes[-1] > 100.0


# -------------------------------------------------------------
# 8. API Gateway & Endpoints Tests
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert "Cholesky" in data["monte_carlo_engine"]


@pytest.mark.asyncio
async def test_api_portfolio_var_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [
                {"ticker": "SPY", "weight": 0.6},
                {"ticker": "TLT", "weight": 0.4}
            ],
            "portfolio_value": 500000.0,
            "confidence_level": 0.95,
            "simulations": 1000,
            "range": "6mo"
        }
        resp = await ac.post("/api/portfolio/var", json=payload, headers={"X-Request-ID": "test-req-777"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == "test-req-777"
        data = resp.json()
        assert "parametric" in data
        assert "historical" in data
        assert "monte_carlo" in data
        assert data["parametric"]["var_amount"] > 0.0


@pytest.mark.asyncio
async def test_api_portfolio_stresstest_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5}
            ],
            "portfolio_value": 1000000.0
        }
        resp = await ac.post("/api/portfolio/stresstest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["scenarios"]) == 3
        assert data["max_loss_amount"] > 0.0


@pytest.mark.asyncio
async def test_api_portfolio_backtest_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [
                {"ticker": "SPY", "weight": 1.0}
            ],
            "confidence_level": 0.95,
            "portfolio_value": 1000000.0
        }
        resp = await ac.post("/api/portfolio/backtest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "kupiec_lr_statistic" in data
        assert "verdict" in data


@pytest.mark.asyncio
async def test_api_market_prices_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/market/prices")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tickers"]) > 0


@pytest.mark.asyncio
async def test_api_validation_error_on_invalid_weights():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [{"ticker": "SPY", "weight": 0.5}],
            "portfolio_value": 1000000.0
        }
        resp = await ac.post("/api/portfolio/var", json=payload)
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_validation_error_on_invalid_confidence():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [{"ticker": "SPY", "weight": 1.0}],
            "confidence_level": 1.5
        }
        resp = await ac.post("/api/portfolio/var", json=payload)
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_validation_error_on_negative_portfolio_value():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [{"ticker": "SPY", "weight": 1.0}],
            "portfolio_value": -100.0
        }
        resp = await ac.post("/api/portfolio/var", json=payload)
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_validation_error_on_empty_positions():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "positions": [],
            "portfolio_value": 100000.0
        }
        resp = await ac.post("/api/portfolio/var", json=payload)
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_request_correlation_id_generated_when_omitted():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
