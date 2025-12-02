from arch import arch_model

def train_garch_model(train_returns):
    """
    Train GARCH model on returns data
    """
    # GARCH works on returns
    model = arch_model(train_returns, vol='Garch', p=1, q=1)
    fitted_model = model.fit(disp='off')
    return fitted_model

def predict_garch(fitted_model, test_returns):
    """
    Generate predictions
    """
    forecast = fitted_model.forecast(horizon=len(test_returns))
    return forecast
