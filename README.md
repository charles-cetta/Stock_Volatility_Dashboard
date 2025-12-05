# Stock Volatility Forecasting Dashboard

A Streamlit-based web application for analyzing and forecasting stock price volatility using three distinct econometric approaches. This dashboard implements GJR-GARCH for advanced conditional volatility modeling, ARIMA for returns forecasting with price conversion, and OLS regression with comprehensive diagnostic testing.

## Overview

This project demonstrates sophisticated financial econometrics in a production-ready web interface. It processes 5-year historical stock data and applies state-of-the-art time series techniques to model volatility patterns and forecast returns. The implementation emphasizes methodological rigor with rolling one-step-ahead forecasts, proper chronological data splits, and extensive model validation.

**Key Differentiators:**
- Advanced GARCH specification (GJR with t-distribution) captures asymmetric volatility and fat tails
- Rolling forecast methodology with automatic model refitting for realistic performance evaluation
- Comprehensive diagnostic testing across all models with statistical validation
- Production-grade error handling with automatic fallback logic

## Features

### Core Capabilities

- **Real-time Data Integration**: Automated data pipeline using yfinance API with 5-year rolling windows
- **Multi-Model Framework**: Three complementary approaches to volatility and returns analysis
- **Interactive Visualizations**: 
  - Historical vs forecasted volatility with train/test split visualization
  - Price and returns forecasts with anchored and compounded views
  - Residual diagnostics and distribution analysis
  - Comprehensive statistical plots
- **Proper Time Series Handling**: 
  - Chronological 75/25 train/test splits (no look-ahead bias)
  - Rolling one-step-ahead forecasts for realistic evaluation
  - Log returns for stationarity and better modeling properties
- **Session State Management**: Persistent data across user interactions for smooth workflow

### Model-Specific Features

**GARCH:**
- Automatic model selection with validation (GJR-GARCH → standard GARCH → fallback logic)
- Student's t-distribution for capturing fat tails
- Leverage effect modeling (asymmetric volatility response)
- Persistence metrics and parameter significance testing

**ARIMA:**
- Automatic order selection via ACF/PACF analysis with stationarity testing
- Rolling forecast with periodic refitting (optimized for performance)
- Dual price forecast views (anchored vs compounded)
- Direction accuracy and baseline comparisons

**OLS:**
- Automatic formula selection from candidate models
- External regressors (S&P 500, VIX) with lagged variables
- 20-day rolling volatility as predictor
- Comprehensive diagnostic test suite (9+ statistical tests)

## Project Structure

```
.
├── app.py                  # Main Streamlit application with UI logic
├── data_loader.py          # Data fetching, preprocessing, train/test split
├── models/
│   ├── garch_model.py      # GJR-GARCH volatility modeling with validation
│   ├── arima.py            # ARIMA returns forecasting with price conversion
│   └── ols_model.py        # OLS regression with diagnostic testing
├── requirements.txt        # Python dependencies
├── data_testing.ipynb      # Exploratory data analysis notebook
└── README.md
```

## Installation

### Requirements

```bash
pip install streamlit yfinance pandas numpy arch statsmodels matplotlib seaborn scikit-learn
```

### Core Dependencies

- `streamlit`: Web application framework
- `yfinance`: Financial data API (5-year historical data)
- `pandas`: Data manipulation and time series operations
- `numpy`: Numerical computations and array operations
- `arch`: GARCH family models implementation
- `statsmodels`: ARIMA and OLS with diagnostic tests
- `matplotlib` / `seaborn`: Statistical visualizations
- `scikit-learn`: Evaluation metrics (MAE, RMSE)

## Usage

### Quick Start

1. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

2. **Enter a stock ticker** (e.g., AAPL, MSFT, TSLA, GOOGL)

3. **Fetch historical data**: Retrieves 5 years of daily closing prices

4. **Explore data tabs**: 
   - Raw data inspection
   - Price series visualization
   - Returns distribution
   - Train/test split analysis with statistical comparisons

5. **Select a model**: Choose from three approaches with distinct button clicks

6. **Analyze results**: Examine forecasts, parameter estimates, diagnostics, and validation metrics

### Model Selection Guide

| Model | Use Case | Strengths | Output |
|-------|----------|-----------|--------|
| **GARCH** | Volatility forecasting, risk metrics | Captures clustering, asymmetry, fat tails | Conditional variance forecasts |
| **ARIMA** | Price/returns prediction, directional signals | Mean reversion, trend capture | One-step-ahead price forecasts |
| **OLS** | Baseline comparison, market sensitivity | Interpretability, external factors | Linear return predictions |

## Technical Implementation

### 1. Data Pipeline (`data_loader.py`)

**Workflow:**
```python
fetch_process_stock_data(ticker)
├── Fetch 5-year historical data via yfinance
├── Calculate percentage returns: pct_change()
├── Create chronological train/test split (75/25)
└── Return structured dictionary with aligned data
```

**Key Design Decisions:**
- Returns used instead of prices (stationary for modeling)
- 75/25 split provides ~940 training, ~310 testing observations
- Aligned indices between prices and returns for conversion
- Company name extraction for user-friendly display

### 2. GARCH Model (`garch_model.py`)

**Implementation: GJR-GARCH(1,1,1) with Student's t-distribution**

**Mathematical Specification:**
```
r_t = μ + ε_t,  ε_t = σ_t * z_t,  z_t ~ t(ν)

σ²_t = ω + α*ε²_{t-1} + γ*I_{t-1}*ε²_{t-1} + β*σ²_{t-1}

where:
- ω (omega): Baseline variance floor
- α (alpha): Symmetric shock impact (ARCH effect)
- γ (gamma): Leverage effect (negative returns increase volatility more)
- β (beta): Volatility persistence (GARCH effect)
- ν (nu): Degrees of freedom (tail thickness)
- I_{t-1} = 1 if ε_{t-1} < 0, else 0
```

**Key Features:**
- **Automatic Model Selection**: Tries GJR-GARCH-t → GARCH-t → GARCH-normal with validation
- **Numerical Scaling**: Returns scaled by 100 for stability, forecasts properly rescaled
- **Validation Checks**:
  - Stationarity constraint (persistence < 1)
  - Parameter positivity (α, β > 0)
  - Log-likelihood reasonableness
  - High persistence warnings (β > 0.99)
- **Rolling Forecasts**: One-step-ahead conditional variance predictions
- **Evaluation Metrics**: MAE, RMSE, QLIKE loss function

**Model Interpretation:**
- **Persistence (α + γ/2 + β)**: How long volatility shocks last (typical: 0.95-0.99)
- **Leverage Effect (γ)**: Asymmetry in volatility response (typically positive)
- **Tail Index (ν)**: Lower values = fatter tails (ν < 10 indicates heavy tails)

### 3. ARIMA Model (`arima.py`)

**Implementation: Automatic ARIMA with Rolling Forecasts**

**Mathematical Specification:**
```
ARIMA(p,d,q): Φ(L)(1-L)^d r_t = Θ(L)ε_t

where:
- p: Autoregressive order (lags of returns)
- d: Differencing order (for stationarity)
- q: Moving average order (lags of errors)
```

**Key Features:**
- **Automatic Order Selection**:
  - Augmented Dickey-Fuller test for stationarity → determines d
  - ACF analysis for MA order → determines q
  - PACF analysis for AR order → determines p
  - Bounded to reasonable limits (max p=10, q=10)
  - Default (1,0,1) if both p=q=0
  
- **Optimized Rolling Forecast**:
  - Refit every 21 days (~monthly) instead of every step
  - Reduces ~300 fits to ~15 fits (20x speedup)
  - Maintains forecast accuracy while improving performance
  
- **Returns to Prices Conversion**:
  ```python
  P_t = P_{t-1} * (1 + r_forecast)
  ```
  - **Anchored view**: Each forecast starts from actual previous price
  - **Compounded view**: Forecasts chain from each other (shows drift)
  
- **Evaluation Metrics**:
  - **Returns level**: MAE, RMSE in return units
  - **Price level**: MAE, RMSE, MAPE in dollar terms
  - **Direction accuracy**: % correct sign predictions (>50% = skill)
  - **Naive baseline**: Random walk (predict no change)

**Model Interpretation:**
- Direction accuracy > 50% indicates predictive power
- MAPE < 5% generally considered good for financial forecasting
- Beating random walk difficult in efficient markets (EMH)
- Smooth compounded curve reveals ARIMA captures drift, not noise

### 4. OLS Model (`ols_model.py`)

**Implementation: Multivariate OLS with Automatic Formula Selection**

**Candidate Models (6 specifications tested):**
```python
1. Asset_Returns ~ SP500_Returns + VIX_Returns + Lagged_Asset_Returns + Prices_Volatility_20days
2. Asset_Returns ~ SP500_Returns + VIX_Returns + Lagged_Asset_Returns
3. Asset_Returns ~ SP500_Returns + VIX_Returns
4. Asset_Returns ~ SP500_Returns + Prices_Volatility_20days
5. Asset_Returns ~ Lagged_Asset_Returns + Prices_Volatility_20days
6. Asset_Returns ~ Lagged_Asset_Returns
```

**Model Selection:**
- All 6 formulas fitted and ranked by RMSE
- Median RMSE used as quality threshold
- Best model selected from those meeting threshold
- Fallback to lowest RMSE if none meet threshold

**External Data Integration:**
- S&P 500 (^GSPC): Market benchmark returns
- VIX (^VIX): Volatility index as fear gauge
- Lagged variables to avoid contemporaneous endogeneity
- 20-day rolling volatility (historical standard deviation)

**Comprehensive Diagnostic Suite (9 tests):**

1. **Multicollinearity**: Coefficient magnitude check (|coef| > 10)
2. **Heteroscedasticity**: Breusch-Pagan test (p < 0.05 indicates violation)
3. **Autocorrelation**: Durbin-Watson statistic (ideal: 1.5-2.5)
4. **Normality**: Jarque-Bera test for residuals (p < 0.05 = non-normal)
5. **Significance**: Individual coefficient p-values (p < 0.05)
6. **Outliers**: Studentized residuals (|residual| > 3)
7. **Parameter Stability**: CUSUM test for structural breaks
8. **Specification**: Ramsey's RESET test for functional form
9. **VIF**: Variance Inflation Factors for multicollinearity (VIF > 10 = concern)

**Test Statistics Displayed:**
- Durbin-Watson statistic
- Breusch-Pagan p-value
- Jarque-Bera p-value
- CUSUM p-value
- RESET p-value
- AIC (model complexity penalty)
- VIF values per predictor

**Visualizations:**
- Residuals over time (temporal patterns)
- Residual distribution with normal overlay
- Residuals vs fitted values (homoscedasticity check)

**Model Interpretation:**
- Positive S&P 500 coefficient = stock moves with market (beta > 0)
- Negative VIX coefficient = inverse relationship with volatility
- Significant lagged returns = momentum or mean reversion
- High R² in financial data is rare (markets are noisy)

## Methodological Rigor

### Time Series Best Practices

**Chronological Splitting:**
- Training: First 75% of data (~3.75 years)
- Testing: Last 25% (~1.25 years)
- Preserves temporal ordering (no future information leakage)
- Mimics real forecasting scenario

**Rolling Forecasts:**
- Refit model at each step (or periodically for ARIMA)
- Use all available past data for each prediction
- More realistic than multi-step forecasts
- Properly accounts for model uncertainty

**Returns vs Prices:**
- Financial returns are typically stationary
- Log returns handle compounding correctly
- Prices are integrated processes (unit root)
- Models operate on returns, converted to prices for interpretation

### Statistical Validation

**GARCH:**
- Stationarity check (persistence < 1)
- Parameter constraints (positivity)
- Log-likelihood comparison across specifications
- Persistence interpretation (shock decay rate)

**ARIMA:**
- Stationarity testing (ADF test)
- ACF/PACF for order identification
- Information criteria (implicitly via ACF/PACF)
- Baseline comparison (naive random walk)

**OLS:**
- 9 distinct diagnostic tests covering:
  - Classical assumptions (homoscedasticity, no autocorrelation)
  - Distributional properties (normality)
  - Model specification (RESET test)
  - Multicollinearity (VIF)
  - Stability (CUSUM)

## Evaluation Metrics

### GARCH Performance

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| MAE | Mean Absolute Error in volatility | Average miss in percentage points |
| RMSE | Root Mean Squared Error in variance | Penalizes large errors |
| QLIKE | Quasi-likelihood loss | Standard for variance forecasts (lower better) |

**Typical Good Values:**
- MAE < 2% (daily volatility units)
- QLIKE < 1.0
- Persistence: 0.95-0.99 (realistic for financial data)

### ARIMA Performance

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Return MAE/RMSE | Forecast error in return units | How well returns are predicted |
| Price MAE/RMSE | Forecast error in dollars | Practical trading interpretation |
| MAPE | Mean Absolute % Error | Scale-independent metric |
| Direction Accuracy | % correct sign predictions | >50% = predictive skill |

**Benchmarks:**
- Direction accuracy > 50% = better than random
- MAPE < 5% = good forecast
- Beat naive random walk = model adds value

### OLS Performance

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| R² | Variance explained | Low expected for financial returns |
| Adjusted R² | Penalized for predictors | Better for model comparison |
| AIC | Information criterion | Lower = better fit-complexity tradeoff |
| F-statistic | Overall significance | p < 0.05 = model is meaningful |

**Expected Values:**
- R² = 0.10-0.30 (typical for financial returns)
- Significant F-statistic but low R² = markets are noisy
- Individual coefficient significance varies by specification

## Dashboard Features

### Data Exploration Tabs

1. **Raw Data**: Last 10 rows of historical OHLCV data
2. **Prices**: Closing price line chart with data table
3. **Returns**: Daily returns visualization and statistics
4. **Train/Test Split**: 
   - Visual split point with distinct colors
   - Date ranges for each period
   - Distribution comparison (histograms + boxplots)
   - Summary statistics table
   - Educational expander on chronological splitting

### Model Outputs

**GARCH:**
- Model type and specification (GJR vs standard, t vs normal)
- Parameter estimates table with significance stars
- Interpretation metrics (persistence, leverage effect, tail index)
- Forecast evaluation (MAE, RMSE, QLIKE)
- Volatility forecast vs 20-day historical rolling
- Split point visualization

**ARIMA:**
- Automatic order (p,d,q)
- Return-level metrics (MAE, RMSE)
- Price-level metrics (MAE, RMSE, MAPE)
- Direction accuracy vs random baseline
- Dual price forecast views (anchored vs compounded)
- Returns forecast with error bars
- Model interpretation expander

**OLS:**
- Full regression summary (statsmodels output)
- Coefficient table with standard errors and p-values
- Test statistics summary (9 diagnostic tests)
- Residuals over time
- Residual distribution with normality check
- Residuals vs fitted values scatter

### User Experience

- **Session State**: Data persists across model selections
- **Error Handling**: Graceful failures with traceback display
- **Warnings System**: Model-specific warnings for validation issues
- **Progress Indicators**: Spinners during computation
- **Responsive Metrics**: Dynamic column layouts based on model parameters
- **Educational Content**: Expanders explaining methodology

## Use Cases

### Risk Management
- **VaR Calculation**: GARCH volatility → Value-at-Risk estimates
- **Position Sizing**: Scale exposure based on forecasted volatility
- **Stress Testing**: Tail risk assessment via t-distribution ν parameter
- **Dynamic Hedging**: Adjust hedges as volatility forecasts change

### Trading Applications
- **Volatility Trading**: Trade vol spreads when GARCH forecasts deviate from implied
- **Directional Signals**: ARIMA direction accuracy > 50% = trading edge
- **Mean Reversion**: ARIMA identifies oversold/overbought conditions
- **Market Timing**: Increase exposure in low volatility regimes (GARCH)

### Portfolio Management
- **Covariance Forecasting**: GARCH for time-varying portfolio variance
- **Factor Analysis**: OLS coefficients show market sensitivity (beta)
- **Correlation Dynamics**: Track how stocks move with S&P 500
- **Rebalancing Signals**: Adjust weights based on volatility forecasts

### Research & Education
- **Model Comparison**: Empirical performance across specifications
- **Methodology Learning**: Production-grade implementation examples
- **Diagnostic Testing**: Comprehensive validation procedures
- **Financial Econometrics**: Applied time series in real-world context

## Future Enhancements

### Model Extensions
- [ ] EGARCH for modeling leverage effect in log-space
- [ ] TARCH/FIGARCH for long memory in volatility
- [ ] Multivariate GARCH (DCC-GARCH) for portfolio applications
- [ ] VAR/VECM for multi-asset analysis
- [ ] Machine learning baseline (LSTM, XGBoost)

### Feature Additions
- [ ] Multiple ticker comparison view
- [ ] Portfolio-level risk metrics
- [ ] Options implied volatility overlay
- [ ] Downloadable forecast exports (CSV, Excel)
- [ ] Customizable train/test split ratios
- [ ] Parameter sensitivity analysis
- [ ] Monte Carlo simulation module

### Infrastructure
- [ ] Automated backtesting framework
- [ ] Model performance tracking over time
- [ ] Caching layer for faster repeated queries
- [ ] Docker containerization
- [ ] REST API for programmatic access
- [ ] Real-time data updates
- [ ] User authentication and saved configurations

## Technical Skills Demonstrated

### Financial Econometrics
- GARCH family models (GJR specification, t-distribution)
- ARIMA modeling with automatic order selection
- Rolling forecast methodology
- Statistical diagnostic testing

### Data Science
- Time series preprocessing and feature engineering
- Train/test splitting with temporal awareness
- Model validation and selection strategies
- Evaluation metric selection for financial context

### Software Engineering
- Modular architecture (separation of concerns)
- Error handling and fallback logic
- Session state management
- API integration (yfinance)

### Web Development
- Streamlit application design
- Interactive visualization with matplotlib
- Responsive UI with dynamic layouts
- User experience optimization

### Statistics
- Hypothesis testing (9 diagnostic tests for OLS)
- Distribution fitting (t-distribution for GARCH)
- Stationarity testing (ADF, ACF/PACF)
- Model specification testing (RESET, CUSUM)

## Performance Considerations

**Data Fetching**: ~2-3 seconds for 5-year history (yfinance API)

**GARCH Training**: ~1-2 seconds for 1200 observations
- GJR-GARCH slightly slower than standard GARCH
- t-distribution adds minimal overhead
- Validation checks negligible

**ARIMA Training**: ~3-5 seconds with optimized rolling forecast
- Original implementation: ~20-30 seconds (refit every step)
- Optimized: Refit every 21 days
- 15-20x speedup with minimal accuracy loss

**OLS Training**: ~2-3 seconds including diagnostics
- 6 candidate models fitted in parallel
- External data fetch adds ~1 second
- Diagnostic tests vectorized for speed

**Total Workflow**: 5-10 seconds from ticker entry to first model results

## Known Limitations

### Data Quality
- Depends on yfinance uptime and data accuracy
- Corporate actions (splits, dividends) handled by yfinance
- Delisted stocks may have incomplete data
- Weekend/holiday gaps handled automatically via datetime index

### Model Assumptions
- **GARCH**: Assumes no structural breaks in volatility process
- **ARIMA**: Assumes linear relationships and constant parameters
- **OLS**: Assumes linear relationships and classical assumptions (often violated)

### Statistical Considerations
- In-sample metrics optimistic vs out-of-sample performance
- 25% test set may be insufficient for robust evaluation
- Single backtest (no k-fold due to temporal dependency)
- Parameter uncertainty not quantified (point estimates only)

### Practical Constraints
- Daily frequency only (no intraday)
- Single asset analysis (no portfolio)
- No transaction costs in evaluation
- No slippage or market impact modeling

## Academic Context

**Coursework**: FIN41660 Financial Econometrics, University College Dublin

**Learning Objectives Demonstrated:**
- Time series modeling with real financial data
- Model specification, estimation, and validation
- Statistical inference and hypothesis testing
- Software implementation of theoretical concepts
- Communication of technical results to non-technical audiences

**Differentiation from Typical Student Work:**
- Production-grade code with error handling
- Sophisticated model specifications (GJR-GARCH, automatic ARIMA)
- Comprehensive validation (9 diagnostic tests)
- User-friendly interface (Streamlit vs Jupyter)
- Realistic methodology (rolling forecasts, chronological splits)

## License

Open source project for educational and portfolio purposes.

## Contact

**Project Type**: Financial Econometrics Portfolio Project

**Target Audience**: 
- Recruiters evaluating quantitative finance candidates
- Academics assessing technical econometrics skills
- Data science hiring managers reviewing applied ML/stats

**Key Takeaway**: This project demonstrates end-to-end capability in financial data science, from mathematical theory through production-ready software implementation, with proper statistical rigor and practical business context.

---

**Last Updated**: December 2024  
**Author**: Charlie (MSc Financial Data Science, UCD)  
**Tech Stack**: Python, Streamlit, GARCH, ARIMA, OLS, yfinance