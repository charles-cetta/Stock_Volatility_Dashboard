"""
GARCH Model Evaluation Script
Run this separately from the Streamlit app to analyze model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
from scipy import stats


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_data(ticker, years=5):
    """Fetch and process stock data"""
    stock = yf.Ticker(ticker)
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(years=years)

    hist = stock.history(start=start_date, end=end_date)
    prices = hist['Close']
    returns = prices.pct_change().dropna()

    split_index = int(0.75 * len(returns))

    return {
        'ticker': ticker,
        'prices': prices,
        'returns': returns,
        'split_index': split_index,
        'train_returns': returns[:split_index],
        'test_returns': returns[split_index:]
    }


# =============================================================================
# MODEL FITTING AND FORECASTING
# =============================================================================

def fit_garch(returns, split_index):
    """Fit GJR-GARCH(1,1,1) model with t-distribution using only training data for parameters"""
    model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='t')
    fitted = model.fit(disp='off', last_obs=split_index)
    return fitted


def get_rolling_forecasts(fitted_model, split_index):
    """Generate rolling one-step-ahead forecasts"""
    forecasts = fitted_model.forecast(start=split_index, horizon=1)
    forecast_variance = forecasts.variance.dropna().values.flatten()
    forecast_volatility = np.sqrt(forecast_variance)
    return forecast_variance, forecast_volatility


# =============================================================================
# REALIZED VOLATILITY MEASURES
# =============================================================================

def compute_realized_measures(test_returns):
    """Compute different proxies for realized volatility"""
    squared_returns = test_returns ** 2
    absolute_returns = np.abs(test_returns)
    rolling_vol = test_returns.rolling(window=20).std()

    return {
        'squared_returns': squared_returns,
        'absolute_returns': absolute_returns,
        'rolling_vol_20d': rolling_vol
    }


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def mean_squared_error(predicted, actual):
    """MSE between predicted and realized variance"""
    return np.mean((predicted - actual) ** 2)


def root_mean_squared_error(predicted, actual):
    """RMSE between predicted and realized"""
    return np.sqrt(np.mean((predicted - actual) ** 2))


def mean_absolute_error(predicted, actual):
    """MAE between predicted and realized volatility"""
    return np.mean(np.abs(predicted - actual))


def mean_absolute_percentage_error(predicted, actual):
    """MAPE between predicted and realized - expressed as percentage"""
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def mincer_zarnowitz_regression(predicted_variance, realized_variance):
    """
    Mincer-Zarnowitz regression: regress realized on predicted
    Good forecasts: intercept ≈ 0, slope ≈ 1, high R²

    Model: realized = alpha + beta * predicted + error
    """
    # Remove any NaN values
    mask = ~(np.isnan(predicted_variance) | np.isnan(realized_variance))
    pred = predicted_variance[mask]
    real = realized_variance[mask]

    # Add constant for intercept
    X = np.column_stack([np.ones(len(pred)), pred])

    # OLS estimation
    beta_hat = np.linalg.lstsq(X, real, rcond=None)[0]
    alpha, beta = beta_hat[0], beta_hat[1]

    # Fitted values and R-squared
    fitted = alpha + beta * pred
    ss_res = np.sum((real - fitted) ** 2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Standard errors and t-stats
    n = len(pred)
    mse = ss_res / (n - 2)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_alpha, se_beta = np.sqrt(np.diag(var_beta))

    t_alpha = alpha / se_alpha  # Test H0: alpha = 0
    t_beta = (beta - 1) / se_beta  # Test H0: beta = 1

    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'se_alpha': se_alpha,
        'se_beta': se_beta,
        't_stat_alpha_eq_0': t_alpha,
        't_stat_beta_eq_1': t_beta,
        'p_value_alpha': 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2)),
        'p_value_beta': 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))
    }


def qlike_loss(predicted_variance, realized_variance):
    """
    QLIKE loss function - commonly used for variance forecasts
    Lower is better
    """
    mask = (predicted_variance > 0) & (realized_variance > 0)
    pred = predicted_variance[mask]
    real = realized_variance[mask]

    return np.mean(np.log(pred) + real / pred)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(ticker='NVDA'):
    """Run full GARCH evaluation"""

    print(f"{'=' * 60}")
    print(f"GJR-GARCH Model Evaluation for {ticker}")
    print(f"{'=' * 60}\n")

    # Fetch data
    print("Fetching data...")
    data = fetch_data(ticker)
    print(f"Total observations: {len(data['returns'])}")
    print(f"Training: {len(data['train_returns'])} | Testing: {len(data['test_returns'])}")
    print(f"Split date: {data['returns'].index[data['split_index']].strftime('%Y-%m-%d')}\n")

    # Fit model
    print("Fitting GJR-GARCH(1,1,1) model with t-distribution...")
    model = fit_garch(data['returns'], data['split_index'])
    print("\nModel Parameters:")
    print(f"  omega:    {model.params['omega']:.6e}")
    print(f"  alpha[1]: {model.params['alpha[1]']:.4f}")
    print(f"  gamma[1]: {model.params['gamma[1]']:.4f}  (asymmetric term)")
    print(f"  beta[1]:  {model.params['beta[1]']:.4f}")
    print(f"  nu:       {model.params['nu']:.2f}  (t-distribution degrees of freedom)")
    print(
        f"  alpha + gamma/2 + beta = {model.params['alpha[1]'] + model.params['gamma[1]'] / 2 + model.params['beta[1]']:.4f}\n")

    # Get forecasts
    forecast_var, forecast_vol = get_rolling_forecasts(model, data['split_index'])

    # Get realized measures
    realized = compute_realized_measures(data['test_returns'])

    # Align lengths (test_returns and forecasts should match)
    n_forecasts = len(forecast_var)
    squared_returns = realized['squared_returns'].values[:n_forecasts]
    absolute_returns = realized['absolute_returns'].values[:n_forecasts]

    # ==========================================================================
    # EVALUATION METRICS
    # ==========================================================================

    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60 + "\n")

    # MSE, RMSE, MAE, MAPE
    mse_var = mean_squared_error(forecast_var, squared_returns)
    rmse_var = root_mean_squared_error(forecast_var, squared_returns)
    mae_vol = mean_absolute_error(forecast_vol, absolute_returns)
    mape_var = mean_absolute_percentage_error(forecast_var, squared_returns)

    print("Loss Functions:")
    print(f"  MSE (variance):    {mse_var:.6e}")
    print(f"  RMSE (variance):   {rmse_var:.6e}")
    print(f"  MAE (volatility):  {mae_vol:.4f}")
    print(f"  MAPE (variance):   {mape_var:.2f}%")
    print(f"  QLIKE:             {qlike_loss(forecast_var, squared_returns):.4f}\n")

    # Mincer-Zarnowitz
    print("Mincer-Zarnowitz Regression (realized = α + β × predicted):")
    mz = mincer_zarnowitz_regression(forecast_var, squared_returns)
    print(f"  α (intercept): {mz['alpha']:.6e}  (SE: {mz['se_alpha']:.6e})")
    print(f"  β (slope):     {mz['beta']:.4f}  (SE: {mz['se_beta']:.4f})")
    print(f"  R²:            {mz['r_squared']:.4f}")
    print(f"\n  Test H0: α = 0:  t = {mz['t_stat_alpha_eq_0']:.2f}, p = {mz['p_value_alpha']:.4f}")
    print(f"  Test H0: β = 1:  t = {mz['t_stat_beta_eq_1']:.2f}, p = {mz['p_value_beta']:.4f}")

    if mz['p_value_alpha'] > 0.05 and mz['p_value_beta'] > 0.05:
        print("\n  ✓ Cannot reject efficient forecast hypothesis (α=0, β=1)")
    else:
        print("\n  ✗ Forecast shows bias (reject α=0 and/or β=1)")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Forecast vs Realized Volatility
    ax1 = axes[0, 0]
    test_dates = data['test_returns'].index[:n_forecasts]
    ax1.plot(test_dates, np.sqrt(squared_returns), alpha=0.5, label='|Returns| (realized proxy)')
    ax1.plot(test_dates, forecast_vol, color='red', linewidth=1.5, label='GARCH Forecast')
    ax1.set_title('GARCH Volatility Forecast vs Realized')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volatility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot with MZ regression line
    ax2 = axes[0, 1]
    ax2.scatter(forecast_var, squared_returns, alpha=0.3, s=10)
    x_line = np.linspace(forecast_var.min(), forecast_var.max(), 100)
    ax2.plot(x_line, mz['alpha'] + mz['beta'] * x_line, 'r-', label=f'MZ: α={mz["alpha"]:.2e}, β={mz["beta"]:.2f}')
    ax2.plot(x_line, x_line, 'k--', alpha=0.5, label='Perfect forecast (45° line)')
    ax2.set_title('Mincer-Zarnowitz Regression')
    ax2.set_xlabel('Predicted Variance')
    ax2.set_ylabel('Realized Variance (r²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Forecast errors over time
    ax3 = axes[1, 0]
    errors = squared_returns - forecast_var
    ax3.plot(test_dates, errors, alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.fill_between(test_dates, errors, 0, alpha=0.3)
    ax3.set_title('Forecast Errors Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Error (Realized - Predicted)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distribution of standardized residuals
    ax4 = axes[1, 1]
    standardized = (data['test_returns'].values[:n_forecasts]) / forecast_vol
    ax4.hist(standardized, bins=50, density=True, alpha=0.7, edgecolor='black')
    x_norm = np.linspace(-4, 4, 100)
    ax4.plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
    ax4.set_title('Standardized Residuals Distribution')
    ax4.set_xlabel('Standardized Return')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('garch_evaluation_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("Plots saved to garch_evaluation_plots.png")
    print("=" * 60)

    return {
        'model': model,
        'forecasts': {'variance': forecast_var, 'volatility': forecast_vol},
        'realized': {'squared_returns': squared_returns},
        'metrics': {'mse': mse_var, 'rmse': rmse_var, 'mae': mae_vol, 'mape': mape_var, 'mz': mz}
    }


if __name__ == "__main__":
    results = run_evaluation('NVDA')