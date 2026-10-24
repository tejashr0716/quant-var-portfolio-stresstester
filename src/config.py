from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DEFAULT_TICKERS: str = "SPY,AAPL,MSFT,TLT,GLD"
    DEFAULT_PORTFOLIO_VALUE: float = 1_000_000.0
    MONTE_CARLO_SIMULATIONS: int = 10000
    CONFIDENCE_LEVEL: float = 0.95
    YAHOO_CHART_BASE_URL: str = "https://query1.finance.yahoo.com/v8/finance/chart"
    LOG_LEVEL: str = "INFO"
    REQUEST_TIMEOUT_SECONDS: float = 15.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def tickers_list(self) -> List[str]:
        return [t.strip().upper() for t in self.DEFAULT_TICKERS.split(",") if t.strip()]


settings = Settings()
