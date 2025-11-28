# Stock Volatility Forecasting Dashboard

A Streamlit-based web application for analyzing and forecasting stock price volatility using econometric time series models. This dashboard compares three distinct approaches: GARCH for conditional volatility modeling, ARIMA for returns forecasting, and OLS as a baseline benchmark.

## Overview

This project demonstrates end-to-end data science capabilities in financial analytics, from data acquisition through model evaluation. It processes 5-year historical stock data and applies sophisticated time series techniques to forecast volatility patterns, providing insights valuable for risk management and trading strategies.

## Features

- **Real-time Data Integration**: Automated data pipeline using yfinance API with 5-year rolling windows
- **Multi-Model Comparison**: Three complementary approaches to volatility analysis
  - **GARCH(1,1)**: Captures volatility clustering and time-varying conditional heteroskedasticity
  - **ARIMA**: Forecasts returns with autoregressive and moving average components
  - **OLS Regression**: Naive baseline for volatility estimation
- **Interactive Visualizations**: 
  - Historical vs forecasted volatility comparison
  - Conditional volatility evolution
  - Residual diagnostics for model validation
- **Proper Time Series Handling**: Chronological 75/25 train/test splits preserving temporal order
- **Session State Management**: Persistent data across user interactions

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── utils/
│   └── data_loader.py      # Data fetching and preprocessing
├── models/
│   ├── garch_model.py      # GARCH volatility modeling
│   ├── arima.py            # ARIMA returns forecasting
│   └── ols_model.py        # OLS baseline regression
├── data_testing.ipynb      # Exploratory data analysis
└── README.md
```

## Installation

### Requirements

```bash
pip install streamlit yfinance pandas numpy arch matplotlib seaborn
```

### Dependencies

- `streamlit`: Web application framework
- `yfinance`: Financial data acquisition
- `pandas`: Data manipulation and time series handling
- `numpy`: Numerical computations
- `arch`: GARCH model implementation
- `matplotlib` / `seaborn`: Visualization

## Usage

1. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

2. **Enter a stock ticker** (e.g., AAPL, MSFT, TSLA)

3. **Fetch historical data**: Application retrieves 5 years of daily price data

4. **Select a model**: Choose from GARCH, ARIMA, or OLS analysis

5. **View results**: Examine forecasts, diagnostics, and model statistics

## Technical Approach

### Data Pipeline

The `data_loader.py` module handles data acquisition and preprocessing:
- Fetches 5-year historical data via yfinance
- Calculates log returns from closing prices
- Performs chronological train/test split (75/25)
- Returns structured dictionary with aligned prices and returns

### Model Implementations

**GARCH(1,1)**: Models volatility as a function of past squared returns and past volatility, capturing the volatility clustering phenomenon observed in financial markets.

**ARIMA**: Forecasts returns series using autoregressive and moving average components, with model order selected via information criteria.

**OLS**: Simple linear regression baseline for comparison, highlighting the sophistication needed for volatility forecasting.

### Key Considerations

- **No Look-Ahead Bias**: Strict chronological splitting ensures models only use past information
- **Returns vs Prices**: Models operate on returns (stationary) rather than prices (non-stationary)
- **Volatility Clustering**: GARCH specifically addresses the autocorrelation in squared returns

## Future Enhancements

- [ ] Add EGARCH/TGARCH for asymmetric volatility effects
- [ ] Implement rolling window backtesting
- [ ] Include VaR and CVaR risk metrics
- [ ] Add model comparison metrics (RMSE, MAE, QLIKE)
- [ ] Support for multiple tickers and portfolio analysis
- [ ] Export functionality for forecasts and diagnostics

## Use Cases

- **Risk Management**: Forecast volatility for VaR calculations and position sizing
- **Options Trading**: Implied vs realized volatility comparison
- **Portfolio Optimization**: Time-varying covariance matrix estimation
- **Research**: Compare econometric approaches to volatility forecasting

## Technical Skills Demonstrated

- Time series econometrics (GARCH, ARIMA)
- Financial data processing and feature engineering
- Web application development with Streamlit
- Data visualization for financial analytics
- Statistical model evaluation and diagnostics
- API integration and data pipeline design

## Notes

- Data quality depends on yfinance availability
- Models assume continuous trading (no gaps for weekends/holidays handled automatically)
- GARCH requires sufficient data for convergence (5 years provides ~1,250 observations)

## License

This project is open source and available for educational and portfolio purposes.

## Contact

Built as a portfolio project demonstrating data science capabilities for quantitative finance roles.
