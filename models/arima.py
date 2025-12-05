import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf, pacf, adfuller


def automatic_arima_order(series, max_ar=10, max_ma=10):
    """
    Determine ARIMA order with sensible constraints using ACF/PACF analysis.

    Parameters:
    -----------
    series : pd.Series
        Time series data (returns)
    max_ar : int
        Maximum AR order to consider
    max_ma : int
        Maximum MA order to consider

    Returns:
    --------
    tuple : (p, d, q) order for ARIMA model
    """
    # Ensure 1D series
    if hasattr(series, 'squeeze'):
        series = series.squeeze()

    # Determine d (differencing order) - returns are typically already stationary
    d = 0
    test_series = series.dropna()
    result = adfuller(test_series)

    if result[1] > 0.05:  # Not stationary
        d = 1
        test_series = series.diff().dropna()
        result = adfuller(test_series)
        if result[1] > 0.05:  # Still not stationary
            d = 2
            test_series = series.diff().diff().dropna()

    # Calculate ACF and PACF
    nlags = min(20, len(test_series) // 4)
    acf_vals = acf(test_series, nlags=nlags)
    pacf_vals = pacf(test_series, nlags=nlags)

    threshold = 1.96 / np.sqrt(len(test_series))

    # Find AR order from PACF (first lag where it becomes insignificant)
    p = 0
    for i in range(1, len(pacf_vals)):
        if abs(pacf_vals[i]) > threshold:
            p = i
        else:
            break

    # Find MA order from ACF
    q = 0
    for i in range(1, len(acf_vals)):
        if abs(acf_vals[i]) > threshold:
            q = i
        else:
            break

    # Constrain to reasonable bounds
    p = min(p, max_ar)
    q = min(q, max_ma)

    # Default to (1,d,1) if both are 0
    if p == 0 and q == 0:
        p, q = 1, 1


    return (p, d, q)


def fit_arima_rolling_forecast(train_returns, test_returns, order, refit_every=21):
    """
    Generate rolling one-step-ahead ARIMA forecasts for returns.

    OPTIMIZED: Instead of refitting every step, we refit only every
    `refit_every` days and reuse the model parameters between refits.

    This reduces ~300 full fits to ~15 full fits.

    Parameters:
    -----------
    train_returns : pd.Series
        Training returns data
    test_returns : pd.Series
        Test returns data (for evaluation)
    order : tuple
        ARIMA order (p, d, q)
    refit_every : int
        Fully refit model every N observations (default 21 ≈ monthly)

    Returns:
    --------
    dict with forecast metrics and series
    """
    forecasts = []
    history = list(train_returns.values)

    # Initial fit on training data
    model = ARIMA(history, order=order)
    model_fit = model.fit()

    for i in range(len(test_returns)):
        # Forecast one step ahead using current model
        forecast = model_fit.forecast(steps=1)[0]
        forecasts.append(forecast)

        # Add actual observation to history
        history.append(test_returns.iloc[i])

        # Only refit periodically (not every step)
        if (i + 1) % refit_every == 0:
            model = ARIMA(history, order=order)
            model_fit = model.fit()

    # Create forecast series with proper index
    forecast_returns = pd.Series(forecasts, index=test_returns.index)

    # Calculate return-level metrics
    rmse = np.sqrt(mean_squared_error(test_returns, forecast_returns))
    mae = mean_absolute_error(test_returns, forecast_returns)

    return {
        'rmse': rmse,
        'mae': mae,
        'forecast_returns': forecast_returns,
        'actual_returns': test_returns,
        'model_fit': model_fit
    }


def fit_arima_rolling_forecast_full(train_returns, test_returns, order):
    """
    Original version - refits every step.
    More accurate but much slower. Use for validation/comparison.
    """
    history = list(train_returns.values)
    forecasts = []

    for i in range(len(test_returns)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        forecasts.append(forecast)
        history.append(test_returns.iloc[i])

    # Create forecast series with proper index
    forecast_returns = pd.Series(forecasts, index=test_returns.index)

    # Calculate return-level metrics
    rmse = np.sqrt(mean_squared_error(test_returns, forecast_returns))
    mae = mean_absolute_error(test_returns, forecast_returns)

    return {
        'rmse': rmse,
        'mae': mae,
        'forecast_returns': forecast_returns,
        'actual_returns': test_returns,
        'model_fit': model_fit
    }


def convert_returns_to_prices(forecast_returns, actual_returns, last_train_price):
    """
    Convert return forecasts back to price levels.

    Since: r_t = (P_t - P_{t-1}) / P_{t-1}
    Then:  P_t = P_{t-1} * (1 + r_t)

    Parameters:
    -----------
    forecast_returns : pd.Series
        Forecasted returns
    actual_returns : pd.Series
        Actual returns (for computing actual prices)
    last_train_price : float
        Last price in the training set (starting point)

    Returns:
    --------
    dict with forecast and actual price series plus metrics
    """
    # Build forecast prices: P_t = P_{t-1} * (1 + r_forecast)
    forecast_prices = []
    current_price = last_train_price

    for r in forecast_returns:
        current_price = current_price * (1 + r)
        forecast_prices.append(current_price)

    forecast_prices = pd.Series(forecast_prices, index=forecast_returns.index)

    # Build actual prices from actual returns
    actual_prices = []
    current_price = last_train_price

    for r in actual_returns:
        current_price = current_price * (1 + r)
        actual_prices.append(current_price)

    actual_prices = pd.Series(actual_prices, index=actual_returns.index)

    # Calculate price-level metrics
    price_mae = mean_absolute_error(actual_prices, forecast_prices)
    price_rmse = np.sqrt(mean_squared_error(actual_prices, forecast_prices))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_prices - forecast_prices) / actual_prices)) * 100

    # Direction accuracy: did we predict the direction of price movement correctly?
    forecast_direction = np.sign(forecast_returns)
    actual_direction = np.sign(actual_returns)
    direction_accuracy = np.mean(forecast_direction == actual_direction) * 100

    return {
        'forecast_prices': forecast_prices,
        'actual_prices': actual_prices,
        'price_mae': price_mae,
        'price_rmse': price_rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }


def run_arima_price_forecast(train_returns, test_returns, train_prices, ticker, order=None):
    """
    Complete ARIMA pipeline: forecast returns, convert to prices.

    Parameters:
    -----------
    train_returns : pd.Series or pd.DataFrame
        Training returns
    test_returns : pd.Series or pd.DataFrame
        Test returns
    train_prices : pd.Series or pd.DataFrame
        Training prices (need last value for conversion)
    ticker : str
        Stock ticker symbol
    order : tuple, optional
        ARIMA order. If None, automatically determined.

    Returns:
    --------
    dict with all results
    """
    # Extract series if DataFrame
    if hasattr(train_returns, 'columns'):
        train_ret_series = train_returns[ticker]
        test_ret_series = test_returns[ticker]
    else:
        train_ret_series = train_returns
        test_ret_series = test_returns

    if hasattr(train_prices, 'columns'):
        last_train_price = train_prices[ticker].iloc[-1]
    else:
        last_train_price = train_prices.iloc[-1]

    # Determine order if not provided
    if order is None:
        order = automatic_arima_order(train_ret_series)

    # Fit ARIMA and get return forecasts
    arima_results = fit_arima_rolling_forecast(
        train_ret_series,
        test_ret_series,
        order
    )

    # Convert to prices
    price_results = convert_returns_to_prices(
        arima_results['forecast_returns'],
        arima_results['actual_returns'],
        last_train_price
    )

    # Naive baseline: random walk (predict price stays same)
    naive_prices = []
    prev_price = last_train_price
    for r in test_ret_series:
        naive_prices.append(prev_price)  # Predict price stays same
        prev_price = prev_price * (1 + r)  # Update with actual
    naive_prices = pd.Series(naive_prices, index=test_ret_series.index)

    naive_mae = mean_absolute_error(price_results['actual_prices'], naive_prices)
    naive_rmse = np.sqrt(mean_squared_error(price_results['actual_prices'], naive_prices))

    return {
        'order': order,
        'return_mae': arima_results['mae'],
        'return_rmse': arima_results['rmse'],
        'forecast_returns': arima_results['forecast_returns'],
        'actual_returns': arima_results['actual_returns'],
        'forecast_prices': price_results['forecast_prices'],
        'actual_prices': price_results['actual_prices'],
        'price_mae': price_results['price_mae'],
        'price_rmse': price_results['price_rmse'],
        'mape': price_results['mape'],
        'direction_accuracy': price_results['direction_accuracy'],
        'naive_mae': naive_mae,
        'naive_rmse': naive_rmse,
        'model_fit': arima_results['model_fit'],
        'last_train_price': last_train_price
    }