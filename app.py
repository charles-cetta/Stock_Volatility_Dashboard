import streamlit as st
from utils.data_loader import fetch_process_stock_data
from models.garch_model import train_garch_model, get_forecast_volatility
from models.ols_model import fit_ols_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Fixed: was 'matplotlib as plt'

st.title('Stock Volatility Dashboard')

# Initialize Session state
if 'stock_data_fetched' not in st.session_state:
    st.session_state.stock_data_fetched = False

ticker = st.text_input('Enter stock symbol').upper()

if st.button("Fetch Data"):
    with st.spinner(f'Fetching data for {ticker}'):
        try:
            result = fetch_process_stock_data(ticker)

            st.session_state.stock_data = result
            st.session_state.stock_data_fetched = True

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.session_state.stock_data_fetched = False

        else:
            st.warning("Please enter a stock ticker symbol")

# Display data if fetched
if st.session_state.get('stock_data_fetched', False):
    data = st.session_state.stock_data

    st.write(f"### Data for {data['ticker']}")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Prices", "Returns", "Train/Test Split"])

    with tab1:
        st.write("**Historical Data (last 10 rows)**")
        st.dataframe(data['raw_data'].tail(10))

    with tab2:
        st.write("**Closing Prices**")
        st.line_chart(data['prices'])
        st.dataframe(data['prices'].tail(10))

    with tab3:
        st.write("**Returns**")
        st.line_chart(data['returns'])
        st.dataframe(data['returns'].tail(10))

    with tab4:
        st.write("**Train/Test Split Info**")

        # Summary Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(data['train_returns']))
        with col2:
            st.metric("Testing Samples", len(data['test_returns']))  # Fixed: was missing value argument
        with col3:
            split_ratio = len(data['train_returns']) / (len(data['train_returns']) + len(data['test_returns']))
            st.metric("Split Ratio", f"{split_ratio:.1%} / {1 - split_ratio:.1%}")

        # Date Ranges
        st.write("**Date Ranges**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(
                f"Training: {data['train_returns'].index[0].strftime('%Y-%m-%d')} to {data['train_returns'].index[-1].strftime('%Y-%m-%d')}")
        with col2:
            st.write(
                f"Testing: {data['test_returns'].index[0].strftime('%Y-%m-%d')} to {data['test_returns'].index[-1].strftime('%Y-%m-%d')}")

        st.write("**Price Series with Train/Test Split**")

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot training period
        ax.plot(data['train_prices'].index, data['train_prices'][data['ticker']],
                color='blue', label='Training Period', linewidth=1.5)

        # Plot testing period
        ax.plot(data['test_prices'].index, data['test_prices'][data['ticker']],
                color='orange', label='Testing Period', linewidth=1.5)

        # Add vertical line at split point
        split_date = data['test_prices'].index[0]
        ax.axvline(x=split_date, color='red', linestyle='--', linewidth=2, label='Split Point')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{data["ticker"]} - Chronological Train/Test Split')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Returns distribution comparison
        st.write("**Returns Distribution: Train vs Test**")

        train_rets = data['train_returns'][data['ticker']]
        test_rets = data['test_returns'][data['ticker']]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histograms
        axes[0].hist(train_rets, bins=50, alpha=0.7, color='blue', label='Train', density=True)
        axes[0].hist(test_rets, bins=50, alpha=0.7, color='orange', label='Test', density=True)
        axes[0].set_xlabel('Daily Returns')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Returns Distribution')
        axes[0].legend()

        # Box Plots
        axes[1].boxplot([train_rets, test_rets], labels=['Train', 'Test'])
        axes[1].set_ylabel('Daily Returns')
        axes[1].set_title('Returns Spread Comparison')

        plt.tight_layout()
        st.pyplot(fig)

        # Summary statistics table
        st.write("**Summary Statistics Comparison**")

        stats_df = pd.DataFrame({
            'Metric': ['Mean Return', 'Std Dev (Volatility)', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Training': [
                f"{train_rets.mean():.4%}",
                f"{train_rets.std():.4%}",
                f"{train_rets.min():.4%}",
                f"{train_rets.max():.4%}",
                f"{train_rets.skew():.3f}",
                f"{train_rets.kurtosis():.3f}"
            ],
            'Testing': [
                f"{test_rets.mean():.4%}",
                f"{test_rets.std():.4%}",
                f"{test_rets.min():.4%}",
                f"{test_rets.max():.4%}",
                f"{test_rets.skew():.3f}",
                f"{test_rets.kurtosis():.3f}"
            ]
        })

        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        # Expander explaining chronological split
        with st.expander("Why Chronological Split?"):
            st.write("""
            **Time series data requires chronological splitting.**

            Random splits would leak future information into training.
            This mimics real forecasting. In practice, you only have past data to predict the future

            Our 75/25 split uses approximately 3.75 years for training and 1.25 years for testing.
            """)

    # Model Selection
    st.write("---")
    st.write("### Select Model")

    if st.button("GARCH Volatility Forecast", use_container_width=True):
        st.session_state.selected_model = 'GARCH'
        with st.spinner("Training GARCH Model"):
            try:
                returns_series = data['returns'][data['ticker']]

                # Train model with automatic validation and fallback
                garch_result = train_garch_model(returns_series, data['split_index'])
                garch_model = garch_result['model']

                # Get properly scaled forecasts
                forecast_variance, forecast_volatility = get_forecast_volatility(
                    garch_result, data['split_index']
                )

                st.success(f"{garch_result['model_type']} trained successfully!")

                # Display any warnings
                if garch_result['warnings']:
                    with st.expander("️Model Warnings", expanded=False):
                        for warning in garch_result['warnings']:
                            st.warning(warning)

                st.write("### Model Information")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Model Type", garch_result['model_type'].split(' with')[0])
                col2.metric("Log-Likelihood", f"{garch_model.loglikelihood:.2f}")
                col3.metric("AIC", f"{garch_model.aic:.2f}")
                col4.metric("Observations", f"{garch_model.nobs}")

                st.write("### Parameter Estimates")

                params = garch_model.params
                std_err = garch_model.std_err
                pvalues = garch_model.pvalues

                # Build parameter table dynamically based on model type
                param_rows = []

                # Omega (always present)
                param_rows.append({
                    'Parameter': 'ω (omega)',
                    'Description': 'Constant variance term',
                    'Estimate': f"{params['omega']:.6e}",
                    'Std Error': f"{std_err['omega']:.2e}",
                    'P-Value': f"{pvalues['omega']:.4f}" if pvalues['omega'] >= 0.0001 else "<0.0001",
                    'Significant': "✓" if pvalues['omega'] < 0.05 else ""
                })

                # Alpha (always present)
                param_rows.append({
                    'Parameter': 'α (alpha)',
                    'Description': 'Shock impact (ARCH term)',
                    'Estimate': f"{params['alpha[1]']:.4f}",
                    'Std Error': f"{std_err['alpha[1]']:.4f}",
                    'P-Value': f"{pvalues['alpha[1]']:.4f}",
                    'Significant': "✓" if pvalues['alpha[1]'] < 0.05 else ""
                })

                # Gamma (only for GJR-GARCH)
                if 'gamma[1]' in params:
                    param_rows.append({
                        'Parameter': 'γ (gamma)',
                        'Description': 'Asymmetric/leverage effect',
                        'Estimate': f"{params['gamma[1]']:.4f}",
                        'Std Error': f"{std_err['gamma[1]']:.4f}",
                        'P-Value': f"{pvalues['gamma[1]']:.4f}" if pvalues['gamma[1]'] >= 0.0001 else "<0.0001",
                        'Significant': "✓" if pvalues['gamma[1]'] < 0.05 else ""
                    })

                # Beta (always present)
                param_rows.append({
                    'Parameter': 'β (beta)',
                    'Description': 'Volatility persistence (GARCH term)',
                    'Estimate': f"{params['beta[1]']:.4f}",
                    'Std Error': f"{std_err['beta[1]']:.4f}",
                    'P-Value': f"{pvalues['beta[1]']:.4f}" if pvalues['beta[1]'] >= 0.0001 else "<0.0001",
                    'Significant': "✓" if pvalues['beta[1]'] < 0.05 else ""
                })

                # Nu (only for t-distribution)
                if 'nu' in params:
                    param_rows.append({
                        'Parameter': 'ν (nu)',
                        'Description': 'Degrees of freedom (tail thickness)',
                        'Estimate': f"{params['nu']:.2f}",
                        'Std Error': f"{std_err['nu']:.2f}",
                        'P-Value': f"{pvalues['nu']:.4f}" if pvalues['nu'] >= 0.0001 else "<0.0001",
                        'Significant': "✓" if pvalues['nu'] < 0.05 else ""
                    })

                param_df = pd.DataFrame(param_rows)
                st.dataframe(param_df, use_container_width=True, hide_index=True)

                st.write("### Model Interpretation")

                # Calculate persistence
                if 'gamma[1]' in params:
                    persistence = params['alpha[1]'] + params['gamma[1]'] / 2 + params['beta[1]']
                else:
                    persistence = params['alpha[1]'] + params['beta[1]']

                # Dynamic columns based on model type
                if 'gamma[1]' in params and 'nu' in params:
                    col1, col2, col3 = st.columns(3)
                elif 'gamma[1]' in params or 'nu' in params:
                    col1, col2 = st.columns(2)
                else:
                    col1 = st.columns(1)[0]

                with col1:
                    st.metric("Persistence", f"{persistence:.4f}")
                    if persistence > 0.99:
                        st.caption("Very high: shocks decay very slowly")
                    elif persistence > 0.95:
                        st.caption("High persistence: volatility shocks are long-lasting")
                    else:
                        st.caption("Moderate persistence: shocks decay reasonably fast")

                if 'gamma[1]' in params:
                    with col2:
                        st.metric("Leverage Effect (γ)", f"{params['gamma[1]']:.4f}")
                        if pvalues['gamma[1]'] < 0.05:
                            st.caption("✓ Significant: negative returns increase volatility more")
                        else:
                            st.caption("Not significant: symmetric volatility response")

                if 'nu' in params:
                    target_col = col3 if 'gamma[1]' in params else col2
                    with target_col:
                        st.metric("Tail Index (ν)", f"{params['nu']:.2f}")
                        if params['nu'] < 6:
                            st.caption("Very heavy tails: frequent extreme moves")
                        elif params['nu'] < 10:
                            st.caption("Heavy tails: more extremes than normal")
                        elif params['nu'] < 30:
                            st.caption("Moderate tails")
                        else:
                            st.caption("Near-normal distribution")

                st.write("### Forecast Evaluation")

                # Calculate metrics
                test_returns = data['test_returns'][data['ticker']].values[:len(forecast_variance)]
                squared_returns = test_returns ** 2
                absolute_returns = np.abs(test_returns)

                mae = np.mean(np.abs(forecast_volatility - absolute_returns))
                mse = np.mean((forecast_variance - squared_returns) ** 2)
                rmse = np.sqrt(mse)

                # QLIKE
                mask = (forecast_variance > 0) & (squared_returns > 0)
                qlike = np.mean(np.log(forecast_variance[mask]) + squared_returns[mask] / forecast_variance[mask])

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "MAE (Volatility)",
                    f"{mae:.4f}",
                    help="Mean Absolute Error - average forecast miss in volatility units"
                )
                col2.metric(
                    "RMSE (Variance)",
                    f"{rmse:.2e}",
                    help="Root Mean Squared Error in variance units"
                )
                col3.metric(
                    "QLIKE",
                    f"{qlike:.4f}",
                    help="Quasi-likelihood loss - standard for variance forecasts (lower is better)"
                )

                st.write("### Volatility Forecast")

                fig, ax = plt.subplots(figsize=(12, 6))

                historical_vol = data['returns'][data['ticker']].rolling(window=20).std()
                ax.plot(historical_vol.index, historical_vol, label='Historical Volatility (20-day)', alpha=0.7)

                test_dates = data['test_returns'].index[:len(forecast_volatility)]
                ax.plot(test_dates, forecast_volatility, label='GARCH Forecast', color='red', linewidth=2)

                split_date = data['returns'].index[data['split_index']]
                ax.axvline(x=split_date, color='green', linestyle='--', linewidth=2, label='Train/Test Split')

                ax.set_xlabel('Date')
                ax.set_ylabel('Volatility')
                ax.set_title(f'{garch_result["model_type"]} Forecast for {data["ticker"]}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error training GARCH model: {e}")
                import traceback

                st.text(traceback.format_exc())


    # OLS Model Button
    if st.button("OLS Regression Analyse", use_container_width=True):
        st.session_state.selected_model = 'OLS Regression'
        with st.spinner("Fitting OLS Model"):
            try:

                prices_series = pd.Series(index = data['prices'].index, data = data['prices'][data['ticker']])

                ols_result = fit_ols_model(prices_series)
                ols_model = ols_result['model']

                st.success(f"OLS model fitted successfully!")

                # Display any warnings
                if ols_result['warnings']:
                    with st.expander("️Model Warnings", expanded=False):
                        for warning in ols_result['warnings']:
                            st.warning(warning)

                st.write("### Model Summary")
                summary_str = ols_model.summary().as_text()
                st.text(summary_str, text_alignment='center',width="stretch")

                st.write("### Model Coefficients")
                coef_df = pd.DataFrame({
                    'Coefficient': ols_model.params,
                    'Std Error': ols_model.bse,
                    'P-Value': ols_model.pvalues
                })
                st.dataframe(coef_df, use_container_width=True, hide_index=False)

                st.write("### Plot Residuals")
                residuals = ols_model.resid
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(residuals.index, residuals, label='Residuals', color='purple')
                ax.axhline(0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('Date')
                ax.set_ylabel('Residuals')
                ax.set_title('OLS Model Residuals Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                # Plot histogram of residuals and normality
                st.write("### Residuals Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(residuals, bins=50, color='teal', alpha=0.7)
                # normality line
                mu = residuals.mean()
                sigma = residuals.std()
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ax.plot(x, p * len(residuals) * (xmax - xmin) / 50, 'r--', linewidth=2, label='Normal Distribution')
                ax.legend()
                ax.set_xlabel('Residuals')
                ax.set_ylabel('Frequency')
                ax.set_title('Histogram of OLS Model Residuals')
                st.pyplot(fig)
                # Scatter plot of residuals vs fitted values
                st.write("### Residuals vs Fitted Values")
                fitted_values = ols_model.fittedvalues
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(fitted_values, residuals, alpha=0.6, color='orange')
                ax.axhline(0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('Fitted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residuals vs Fitted Values')
                st.pyplot(fig)


            except Exception as e:
                st.error(f"Error fitting OLS model: {e}")
                import traceback

                st.text(traceback.format_exc())



    if 'selected_model' in st.session_state:
        st.write(f"**Currently viewing: {st.session_state.selected_model}**")