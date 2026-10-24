from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class Position(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=25)
    weight: float = Field(..., gt=0.0, le=1.0)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()


class PortfolioRequest(BaseModel):
    positions: List[Position] = Field(..., min_length=1)
    portfolio_value: float = Field(default=1_000_000.0, gt=0.0)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    simulations: int = Field(default=10000, ge=100, le=100000)
    range: str = Field(default="1y", description="Time horizon e.g. 1y, 2y")

    @field_validator("positions")
    @classmethod
    def validate_weights(cls, v: List[Position]) -> List[Position]:
        total_weight = sum(p.weight for p in v)
        if not (0.95 <= total_weight <= 1.05):
            raise ValueError(f"Total weights must sum to ~1.0 (got {total_weight:.3f})")
        return v


class RiskMetric(BaseModel):
    confidence_level: float
    var_amount: float
    var_pct: float
    cvar_amount: float
    cvar_pct: float


class VaRReport(BaseModel):
    portfolio_value: float
    positions: List[Position]
    annualized_volatility: float
    parametric: RiskMetric
    historical: RiskMetric
    monte_carlo: RiskMetric
    calculation_time_ms: float


class StressScenarioResult(BaseModel):
    scenario_name: str
    description: str
    asset_shocks: Dict[str, float]
    portfolio_pnl: float
    portfolio_return_pct: float


class StressTestResponse(BaseModel):
    portfolio_value: float
    scenarios: List[StressScenarioResult]
    worst_scenario: str
    max_loss_amount: float
    max_loss_pct: float


class BacktestRequest(BaseModel):
    positions: List[Position] = Field(..., min_length=1)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    portfolio_value: float = Field(default=1_000_000.0, gt=0.0)


class BacktestResponse(BaseModel):
    total_days: int
    confidence_level: float
    exceptions_count: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr_statistic: float
    p_value: float
    model_accepted: bool
    verdict: str


class HealthResponse(BaseModel):
    status: str
    service: str
    supported_assets: List[str]
    monte_carlo_engine: str
