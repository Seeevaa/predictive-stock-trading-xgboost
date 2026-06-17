Markdown

# Trading Engine using XGBoost

An end-to-end, production-ready Machine Learning pipeline and backtesting engine built with Python. This project utilizes historical stock data and foreign exchange rates via the Yahoo Finance API to engineer technical quant features and leverage an Optimized XGBoost Classifier for predictive directional trading signals.

Designed with core software engineering principles, the project strictly isolates data ingestion, mathematical transformation, walk-forward validation, and backtesting evaluation into modular components.

---

## 📈 Features

- **Data Engineering (`data_pipeline.py`)**: Robust market data ingestion utilizing asynchronous API polling with strict inner-join synchronization across asset classes. Handles mechanical data purification by capturing and cleansing structural anomalies (Infinities and NaNs).
- **Predictive Modeling (`model.py`)**: Implements strict `TimeSeriesSplit` (Walk-Forward Validation) to simulate realistic market conditions and completely eradicate look-ahead bias or data leakage.
- **Backtesting Simulation (`backtest.py`)**: Compiles and parses strategy performance vs. a benchmark Buy & Hold baseline. Computes equity curve compounding and dynamically maps tactical exit execution points.
- **Orchestration Interface (`main.py`)**: Central control gateway that executes the full end-to-end analytical matrix, maps model confidence metric probability distribution, and updates predictive inferences for the upcoming trading session.

---

## 🛠️ Project Structure

```text
├── utils/
│   ├── __init__.py
│   ├── data_pipeline.py    # Market data ingestion and feature engineering engine
│   ├── model.py            # Walk-Forward TimeSeries validation & training matrix
│   └── backtest.py         # Quantitative backtester & visualization suite
├── main.py                 # Pipeline orchestrator and live session inference gateway
├── requirements.txt        # Production dependencies mapping
└── .gitignore              # Automated exclusions pattern for binary cache and raw charts

🔄 Ticker & Asset Customization

This quantitative framework is completely agnostic to specific stock tickers or fiat currencies. While the baseline configuration is set to analyze PT Bank Central Asia Tbk (BBCA.JK), you can dynamically shift the underlying analysis to any global asset or Indonesian equity listed on Yahoo Finance without modifying the core pipeline modules.

To target a different asset, simply update the entry parameters inside main.py:
Python

# Open main.py and locate the Data Ingestion section:

# Example 1: To analyze Telkom Indonesia (TLKM)
df_raw = fetch_market_data(ticker="TLKM.JK", currency_pair="IDR=X")

# Example 2: To analyze Bank Mandiri (BMRI)
df_raw = fetch_market_data(ticker="BMRI.JK", currency_pair="IDR=X")

# Example 3: To analyze global tech assets like Nvidia (NVDA)
df_raw = fetch_market_data(ticker="NVDA", currency_pair="IDR=X")

⚙️ Setup & Execution Instructions
1. Clone the Project Environment
Bash

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

2. Install Dependencies

Ensure you are running Python 3.8 or higher, then execute:
Bash

pip install -r requirements.txt

3. Run the Production Pipeline

To trigger the end-to-end ingestion, forward training folds, strategic backtest simulation, and export your session parameters, run:
Bash

python main.py

Upon successful run, the orchestrator will print a comprehensive Analytical Dashboard Terminal mapping directional signals and model confidence matrix, while exporting the structural charts as strategy_performance.png.
