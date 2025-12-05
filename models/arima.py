import arch
from arch import arch_model
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def train_arima_model(returns, split_index):
    """
    Train an ARIMA model on the returns data.

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
    # check for stationarity
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(returns)
    if adf_result[1] > 0.05:

        # differencing
        returns = returns.diff().dropna()
        # check again
        adf_result = adfuller(returns)
        if adf_result[1] > 0.05:
            #data is still non-stationary, differencing again
            returns = returns.diff().dropna()
            # check again

            adf_result = adfuller(returns)
            if adf_result[1] > 0.05:
                warnings.append("Data is non-stationary even after differencing twice. ARIMA model may not be appropriate.")
                # Dont proceed with ARIMA fitting
                return {
                    'model': None,
                    'model_type': 'ARIMA',
                    'warnings': warnings
                }
            else:
                # Data is stationary after second differencing
                d = 2
        else:
            # Data is stationary after first differencing
            d = 1
    else:
        # Data is stationary
        d = 0


    # fit ARIMA model based on PACF and ACF plots or use auto_arima for better selection

    # Fit ARIMA model (p,d,q) = (1,0,1) as a common starting point
    try:
        model = ARIMA(returns, order=(1, d, 1))
        fitted = model.fit()

        return {
            'model': fitted,
            'model_type': 'ARIMA(1,0,1)',
            'warnings': warnings
        }

    except Exception as e:
        warnings.append(f"ARIMA model fitting failed: {str(e)}")
        return {
            'model': None,
            'model_type': 'ARIMA(1,0,1)',
            'warnings': warnings
        }