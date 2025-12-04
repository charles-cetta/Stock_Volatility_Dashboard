from arch import arch_model
import numpy as np


def train_garch_model(returns, split_index):
    """
    Train GARCH model with automatic model selection and validation.

    Attempts GJR-GARCH with t-distribution first, falls back to simpler
    specifications if the model fit is poor.

    Parameters:
    -----------
    returns : pd.Series
        Full returns series
    split_index : int
        Index where training data ends

    Returns:
    --------
    dict with:
        - 'model': fitted model object
        - 'model_type': string describing which model was used
        - 'warnings': list of any warnings about the fit
    """
    warnings = []

    # Scale returns by 100 for numerical stability
    scaled_returns = returns * 100

    # Try GJR-GARCH with t-distribution first
    try:
        model = arch_model(scaled_returns, vol='Garch', p=1, o=1, q=1, dist='t')
        fitted = model.fit(disp='off', last_obs=split_index)

        # Validate the fit
        is_valid, validation_warnings = validate_garch_fit(fitted)
        warnings.extend(validation_warnings)

        if is_valid:
            return {
                'model': fitted,
                'model_type': 'GJR-GARCH(1,1,1) with t-distribution',
                'warnings': warnings,
                'scaled': True
            }
        else:
            warnings.append("GJR-GARCH-t fit was poor, trying standard GARCH...")

    except Exception as e:
        warnings.append(f"GJR-GARCH-t failed: {str(e)}")

    # Fallback 1: Standard GARCH with t-distribution
    try:
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t')
        fitted = model.fit(disp='off', last_obs=split_index)

        is_valid, validation_warnings = validate_garch_fit(fitted)
        warnings.extend(validation_warnings)

        if is_valid:
            return {
                'model': fitted,
                'model_type': 'GARCH(1,1) with t-distribution',
                'warnings': warnings,
                'scaled': True
            }
        else:
            warnings.append("GARCH-t fit was poor, trying GARCH with normal distribution...")

    except Exception as e:
        warnings.append(f"GARCH-t failed: {str(e)}")

    # Fallback 2: Standard GARCH with normal distribution
    try:
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
        fitted = model.fit(disp='off', last_obs=split_index)

        is_valid, validation_warnings = validate_garch_fit(fitted)
        warnings.extend(validation_warnings)

        return {
            'model': fitted,
            'model_type': 'GARCH(1,1) with normal distribution',
            'warnings': warnings,
            'scaled': True
        }

    except Exception as e:
        warnings.append(f"All GARCH models failed: {str(e)}")
        raise ValueError(f"Could not fit any GARCH model: {warnings}")


def validate_garch_fit(fitted_model):
    """
    Check if a GARCH model fit is reasonable.

    Returns:
    --------
    (is_valid, warnings) tuple
    """
    warnings = []
    is_valid = True

    # Check 1: Check for positive log-likelihood
    if fitted_model.loglikelihood < -10000:
        warnings.append(f"Very low log-likelihood ({fitted_model.loglikelihood:.0f}) suggests poor fit")
        is_valid = False

    # Check for Stationarity
    params = fitted_model.params
    if 'gamma[1]' in params:
        persistence = params.get('alpha[1]', 0) + params.get('gamma[1]', 0) / 2 + params.get('beta[1]', 0)
    else:
        persistence = params.get('alpha[1]', 0) + params.get('beta[1]', 0)

    if persistence >= 1:
        warnings.append(f"Persistence ({persistence:.4f}) >= 1 indicates non-stationary volatility")
        is_valid = False
    elif persistence > 0.995:
        warnings.append(f"Very high persistence ({persistence:.4f}) - shocks are nearly permanent")

    # Check 3 for negative parameters
    if params.get('alpha[1]', 0) < 0 or params.get('beta[1]', 0) < 0:
        warnings.append("Negative ARCH/GARCH parameters detected")
        is_valid = False

    # Check beta and alpha
    if params.get('beta[1]', 0) > 0.99 and params.get('alpha[1]', 0) < 0.005:
        warnings.append("Model suggests volatility doesn't respond to new shocks")

    return is_valid, warnings


def predict_garch(fitted_model, split_index):
    """
    Generate rolling one-step-ahead forecasts.

    Parameters:
    -----------
    fitted_model : ARCHModelResult
        Fitted GARCH model
    split_index : int
        Index where forecasting begins

    Returns:
    --------
    forecast object from arch library
    """
    forecast = fitted_model.forecast(start=split_index, horizon=1)
    return forecast


def get_forecast_volatility(garch_result, split_index):
    """
    Extract forecast volatility from model result, handling scaling.

    Parameters:
    -----------
    garch_result : dict
        Result from train_garch_model
    split_index : int
        Index where forecasting begins

    Returns:
    --------
    tuple: (forecast_variance, forecast_volatility) as numpy arrays
    """
    predictions = predict_garch(garch_result['model'], split_index)

    # Extract variance forecasts
    forecast_variance = predictions.variance.dropna().values.flatten()

    # If returns were scaled by 100, variance is scaled by 100^2 = 10000
    if garch_result.get('scaled', False):
        forecast_variance = forecast_variance / 10000  # Rescale back

    forecast_volatility = np.sqrt(forecast_variance)

    return forecast_variance, forecast_volatility