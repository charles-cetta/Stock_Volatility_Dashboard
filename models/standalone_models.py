import numpy as np
import pandas as pd
from utils.data_loader import fetch_process_stock_data
from garch_model import train_garch_model, get_forecast_volatility
from arima import run_arima_price_forecast
from ols_model import fit_ols_model


#Loading Example Ticker to Show Models Working in Standalone Python File
#We will using Microsoft (MSFT) to conduct our standalone tests

print("Standalone Model Tests - Volatility Dashboard Project")

data = fetch_process_stock_data('MSFT')
print(f"Training samples: {len(data['train_returns'])}")
print(f"Testing samples: {len(data['test_returns'])}")

# OLS Standalone Test
print("OLS Regression Test")

prices_series = pd.Series(index=data['prices'].index, data=data['prices'][data['ticker']])

ols_result = fit_ols_model(prices_series)
ols_model = ols_result['model']

print(f"Model Type: {ols_result['model_type']}")
print(f"Formula: {ols_result['formula']}")
print(f"R-squared: {ols_model.rsquared:.4f}")
print(f"AIC: {ols_model.aic:.2f}")
print("\nCoefficients:")
for var in ols_model.params.index:
    print(f"  {var}: {ols_model.params[var]:.6f} (p={ols_model.pvalues[var]:.4f})")


# ARIMA Standalone Test
print("ARIMA Price Forecast")

arima_results = run_arima_price_forecast(
    train_returns=data['train_returns'],
    test_returns=data['test_returns'],
    train_prices=data['train_prices'],
    ticker=data['ticker']
)

print(f"Order (p,d,q): {arima_results['order']}")
print(f"Return MAE: {arima_results['return_mae']:.6f}")
print(f"Return RMSE: {arima_results['return_rmse']:.6f}")
print(f"Price MAE: ${arima_results['price_mae']:.2f}")
print(f"Price RMSE: ${arima_results['price_rmse']:.2f}")
print(f"Direction Accuracy: {arima_results['direction_accuracy']:.1f}%\n")

# GARCH Standalone Test
print("GARCH Volatility Forecast\n")

returns_series = data['returns'][data['ticker']]

# Train model with automatic validation and fallback
garch_result = train_garch_model(returns_series, data['split_index'])
garch_model = garch_result['model']

# Get properly scaled forecasts
forecast_variance, forecast_volatility = get_forecast_volatility(
    garch_result, data['split_index']
)

print(f"Model Type: {garch_result['model_type']}")
print(f"Log-Likelihood: {garch_model.loglikelihood:.2f}")
print(f"AIC: {garch_model.aic:.2f}")
print("\nParameters:")

for param in garch_model.params.index:

    print(f"  {param}: {garch_model.params[param]:.6f}")
print(f"\nForecast Volatility (first 5): {forecast_volatility[:5]}")

