# Setup & Operations Guide: Quant VaR Portfolio Stress Tester

## Prerequisites
- Python 3.12+ (tested on Python 3.14)
- Internet access to fetch historical market data from Yahoo Finance (`https://query1.finance.yahoo.com`)

## Installation
```bash
git clone https://github.com/tejashr0716/quant-var-portfolio-stresstester.git
cd quant-var-portfolio-stresstester

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration (.env)
```bash
cp .env.example .env
```

Available configuration flags:
- `DEFAULT_TICKERS`: Comma-separated default asset symbols (`SPY,AAPL,MSFT,TLT,GLD`)
- `DEFAULT_PORTFOLIO_VALUE`: Default notional capital (`1000000.0`)
- `MONTE_CARLO_SIMULATIONS`: Default number of paths (`10000`)
- `CONFIDENCE_LEVEL`: Default VaR confidence level (`0.95`)
- `YAHOO_CHART_BASE_URL`: Live price feed endpoint

## Running the API Gateway
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running Automated Tests
```bash
pytest -v --junitxml=artifacts/test-results.xml
```

## Running Empirical Benchmark
```bash
python benchmarks/benchmark_var_simulation.py
```
Outputs empirical simulation performance metrics to `artifacts/benchmark-results.json`.
