
import pandas as pd
import statsmodels.formula.api as smf
import yfinance as yf
import numpy as np

def fit_ols_model(prices: pd.Series) -> dict:
    """
    Fits an Ordinary Least Squares (OLS) regression model for a given price series and S&P 500 historical data.
    The regression uses lagged prices and lagged S&P 500 prices as explanatory variables.
    This function processes the input data, merges S&P 500 data, creates lagged variables, and fits the OLS model
    using observations up to the specified split index.

    :param prices: A pandas Series containing price data as the dependent variable for the regression.
    :param split_index: An integer indicating the index at which to split the data for model training.
    :return: A dictionary containing the fitted OLS model object, model type, formula used,
        and a list of warnings encountered during the fitting process.
    """
    warnings = []

    try:
        # Fetch S&P 500 data for OLS fitting
        SP_500 = yf.Ticker("^GSPC")
        sp500_data = SP_500.history(period="5y")['Close']
        VIX = yf.Ticker("^VIX")
        vix_data = VIX.history(period="5y")['Close']
        vix_df = vix_data.rename('VIX_Prices').to_frame()
        prices = prices.rename('Prices')

        prices = prices.to_frame()

        prices.index = pd.to_datetime(prices.index).date
        sp500_data.index = pd.to_datetime(sp500_data.index).date
        vix_df.index = pd.to_datetime(vix_df.index).date

        merge_data = pd.DataFrame({'SP500_Prices': sp500_data})

        merge_data = merge_data.merge(vix_df, how='left', left_index=True, right_index=True)

        prices = prices.merge(merge_data,  how='left',left_index=True, right_index=True)

        prices['Lagged_Prices'] = prices['Prices'].shift(1)
        prices['SP500_Lagged'] = prices['SP500_Prices'].shift(1)
        prices['VIX_Lagged'] = prices['VIX_Prices'].shift(1)
        prices['Returns'] = np.log(prices['Prices']).diff()
        prices['Prices_Volatility_20days'] = prices['Returns'].rolling(window=20, min_periods=20).std(ddof=0)
        prices = prices.dropna()

        prices['Asset_Returns'] = np.log(prices['Prices']).diff()

        prices['SP500_Returns'] = np.log(prices['SP500_Prices']).diff()

        prices['VIX_Returns'] = np.log(prices['VIX_Prices']).diff()

        prices['Lagged_Asset_Returns'] = prices['Asset_Returns'].shift(1)

        # 20-Tage-Rolling-Volatilität auf Basis der Asset-Returns
        prices['Prices_Volatility_20days'] = (
            prices['Asset_Returns'].rolling(window=20, min_periods=20).std(ddof=0)
        )

        # Define the regression formulas
        formular_List = [
            'Asset_Returns ~ SP500_Returns + VIX_Returns + Lagged_Asset_Returns + Prices_Volatility_20days',
            'Asset_Returns ~ SP500_Returns + VIX_Returns + Lagged_Asset_Returns',
            'Asset_Returns ~ SP500_Returns + VIX_Returns',
            'Asset_Returns ~ SP500_Returns + Prices_Volatility_20days',
            'Asset_Returns ~ Lagged_Asset_Returns + Prices_Volatility_20days',
            'Asset_Returns ~ Lagged_Asset_Returns'
        ]

        models_list = []
        for formular in formular_List:
            try:
                m = smf.ols(formula=formular, data=prices).fit()
                rmse = float(np.sqrt(np.mean(m.resid ** 2)))
                models_list.append((rmse, formular, m))
            except Exception as e:
                warnings.append(f"OLS fitting with formula '{formular}' failed: {str(e)}")
        # Wähle das beste Model basierend auf RMSE (kleiner ist besser).
        if models_list:
            rmses = [t[0] for t in models_list]
            threshold = float(np.median(rmses))
            saved_models = [t for t in models_list if t[0] <= threshold]
            if saved_models:
                best_rmse, best_formula, best_model = min(saved_models, key=lambda x: x[0])
            else:
                best_rmse, best_formula, best_model = min(models_list, key=lambda x: x[0])
        else:
            raise Exception("No OLS models could be fitted.")
        formular = best_formula

        # Fit the OLS model
        model = smf.ols(formula=formular, data=prices).fit()

        # Test for multicollinearity
        if any(abs(model.params[1:]) > 10):
            warnings.append("High multicollinearity detected among explanatory variables.")
        # Check for heteroscedasticity using Breusch-Pagan test
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        if bp_test[1] < 0.05:
            warnings.append("Heteroscedasticity detected in the residuals of the OLS model.")
        # Test for autocorrelation using Durbin-Watson test
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(model.resid)
        if dw_stat < 1.5 or dw_stat > 2.5:
            warnings.append("Autocorrelation detected in the residuals of the OLS model.")
        # Test for normality of residuals using Jarque-Bera test
        from statsmodels.stats.stattools import jarque_bera
        jb_test = jarque_bera(model.resid)
        if jb_test[1] < 0.05:
            warnings.append("Residuals of the OLS model are not normally distributed.")
        # Test for significance of coefficients
        if any(model.pvalues[1:] > 0.05):
            warnings.append("Some coefficients in the OLS model are not statistically significant.")
        # Test for outliers using standardized residuals
        standardized_residuals = model.get_influence().resid_studentized_internal
        if any(abs(standardized_residuals) > 3):
            warnings.append("Outliers detected in the residuals of the OLS model.")
        # Test for parameter stability using CUSUM test
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        cusum_test = breaks_cusumolsresid(model.resid)
        if cusum_test[1] < 0.05:
            warnings.append("Parameter instability detected in the OLS model using CUSUM test.")
        # Test for model specification using Ramsey's RESET test
        from statsmodels.stats.diagnostic import linear_reset
        reset_test = linear_reset(model, power=2, use_f=True)
        if reset_test.pvalue < 0.05:
            warnings.append("Model specification error detected in the OLS model using Ramsey's RESET test.")
        # Test for parsimony using Akaike Information Criterion (AIC)
        if model.aic > 1000:
            warnings.append("OLS model may lack parsimony based on high AIC value.")
        # Compute Variance Inflation Factor (VIF) for multicollinearity
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["feature"] = model.model.exog_names
        vif_data["VIF"] = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
        if any(vif_data['VIF'][1:] > 10):
            warnings.append("High Variance Inflation Factor (VIF) detected, indicating multicollinearity among explanatory variables.")
        # Return the fitted model and any warnings

        # Important Test Statistics
        test_df = pd.DataFrame(model.model.exog)

        # Add test statistics to display in the dashboard
        model.test_statistics = {
            'Durbin_Watson': dw_stat,
            'Breusch_Pagan_pvalue': bp_test[1],
            'Jarque_Bera_pvalue': jb_test[1],
            'CUSUM_pvalue': cusum_test[1],
            'RESET_pvalue': reset_test.pvalue,
            'AIC': model.aic,
            'VIF': vif_data.set_index('feature')['VIF'].to_dict()
        }

        return {
            'model': model,
            'model_type': 'Ordinary Least Squares (OLS)',
            'formula': formular,
            'warnings': warnings,
            'test_statistics': model.test_statistics
        }

    except Exception as e:
        warnings.append(f"OLS fitting failed: {str(e)}")
        return {
            'model': None,
            'model_type': 'Ordinary Least Squares (OLS)',
            'warnings': warnings
        }