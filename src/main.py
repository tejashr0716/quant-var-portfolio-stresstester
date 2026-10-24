import uuid
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from src.config import settings
from src.models import (
    PortfolioRequest,
    VaRReport,
    StressTestResponse,
    BacktestRequest,
    BacktestResponse,
    HealthResponse
)
from src.market_data import market_data
from src.risk_engine import risk_engine

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("quant_var")


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        corr_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = corr_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = corr_id
        return response


app = FastAPI(
    title="Quant VaR Portfolio Stress Tester",
    description="Institutional-grade quantitative risk analytics engine providing Parametric, Historical, Monte Carlo VaR (Cholesky), Stress Testing, and Kupiec Backtesting.",
    version="1.0.0"
)

app.add_middleware(RequestCorrelationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="HEALTHY",
        service="Quant VaR Portfolio Stress Tester",
        supported_assets=settings.tickers_list,
        monte_carlo_engine="NumPy-Vectorized-Cholesky"
    )


@app.post("/api/portfolio/var", response_model=VaRReport)
async def compute_portfolio_var(payload: PortfolioRequest):
    try:
        tickers = [p.ticker for p in payload.positions]
        data = await market_data.get_aligned_returns_matrix(tickers, range_str=payload.range)
        
        report = risk_engine.calculate_var_suite(
            returns_matrix=data["returns"],
            tickers=data["tickers"],
            positions=payload.positions,
            portfolio_value=payload.portfolio_value,
            confidence_level=payload.confidence_level,
            simulations=payload.simulations
        )
        return report
    except Exception as e:
        logger.error("Error computing portfolio VaR: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VaR computation failed: {str(e)}"
        )


@app.post("/api/portfolio/stresstest", response_model=StressTestResponse)
async def run_stress_test(payload: PortfolioRequest):
    try:
        response = risk_engine.run_stress_tests(
            positions=payload.positions,
            portfolio_value=payload.portfolio_value
        )
        return response
    except Exception as e:
        logger.error("Error executing stress tests: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress testing failed: {str(e)}"
        )


@app.post("/api/portfolio/backtest", response_model=BacktestResponse)
async def run_backtest(payload: BacktestRequest):
    try:
        tickers = [p.ticker for p in payload.positions]
        data = await market_data.get_aligned_returns_matrix(tickers, range_str="1y")
        
        response = risk_engine.run_kupiec_backtest(
            returns_matrix=data["returns"],
            tickers=data["tickers"],
            positions=payload.positions,
            confidence_level=payload.confidence_level,
            portfolio_value=payload.portfolio_value
        )
        return response
    except Exception as e:
        logger.error("Error running backtest: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtesting failed: {str(e)}"
        )


@app.get("/api/market/prices")
async def get_market_prices():
    tickers = settings.tickers_list
    data = await market_data.get_aligned_returns_matrix(tickers, range_str="1mo")
    return {
        "tickers": data["tickers"],
        "latest_prices": data["latest_prices"],
        "trading_days_analyzed": len(data["returns"])
    }
