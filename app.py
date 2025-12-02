import streamlit as st
from utils.data_loader import fetch_process_stock_data
from models.garch_model import train_garch_model, predict_garch
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

        # Box plot comparison
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
            **Time series data requires chronological splitting, not random splitting.**

            - **Avoids look-ahead bias**: Random splits would leak future information into training
            - **Mimics real forecasting**: In practice, you only have past data to predict the future
            - **Preserves temporal dependencies**: Volatility clustering and other patterns stay intact

            Our 75/25 split uses approximately 3.75 years for training and 1.25 years for testing.
            """)

    # Model Selection
    st.write("---")
    st.write("### Select Model")

    if st.button("GARCH Volatility Forecast", use_container_width=True):
        st.session_state.selected_model = 'GARCH'
        with st.spinner("Training GARCH Model"):
            try:
                train_returns_series = data['train_returns'][data['ticker']]
                garch_model = train_garch_model(train_returns_series)
                garch_predictions = predict_garch(garch_model, data['test_returns'])

                forecast_variance = garch_predictions.variance.values[-1, :]
                forecast_volatility = np.sqrt(forecast_variance)

                st.success("GARCH model trained successfully!")

                # Model Summary
                st.write("### Model Summary")
                st.text(str(garch_model.summary()))

                # Volatility Forecast Plot
                st.write("### Volatility Forecast")

                fig, ax = plt.subplots(figsize=(12, 6))

                historical_vol = data['train_returns'][data['ticker']].rolling(window=20).std()
                ax.plot(historical_vol.index, historical_vol, label='Historical Volatility (20-day)', alpha=0.7)

                test_dates = data['test_returns'].index[:len(forecast_volatility)]
                ax.plot(test_dates, forecast_volatility, label='GARCH Forecast', color='red', linewidth=2)

                ax.set_xlabel('Date')
                ax.set_ylabel('Volatility')
                ax.set_title(f'GARCH Volatility Forecast for {data["ticker"]}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error training GARCH model: {e}")
                import traceback

                st.text(traceback.format_exc())

    if 'selected_model' in st.session_state:
        st.write(f"**Currently viewing: {st.session_state.selected_model}**")