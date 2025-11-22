import streamlit as st
from utils.data_loader import fetch_process_stock_data
from models.garch_model import train_garch_model, predict_garch
import numpy as np
import matplotlib as plt


st.title('Stock Volatility Dashboard')

#Initialize Session state
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
        st.write(f"- Training samples: {len(data['train_returns'])}")
        st.write(f"- Testing samples: {len(data['test_returns'])}")
        st.write(
            f"- Split ratio: {len(data['train_returns']) / (len(data['train_returns']) + len(data['test_returns'])):.2%}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Train Returns (last 5)**")
            st.dataframe(data['train_returns'].tail())
        with col2:
            st.write("**Test Returns (first 5)**")
            st.dataframe(data['test_returns'].head())


    #Model Selection
    st.write("---")
    st.write("### Select Model")

    if st.button("GARCH Volatility Forecast", use_container_width=True):
        st.session_state.selected_model = 'GARCH'
        with st.spinner("Training GARCH Model"):
            try:

                train_returns_series = data['train_returns'][data['ticker']]
                garch_model = train_garch_model(data['train_returns'][data['ticker']])
                garch_predictions = predict_garch(garch_model, data['test_returns'])

                forecast_variance = garch_predictions.variance.values[-1, :()]
                forecast_volatility = np.sqrt(forecast_variance)

                st.success("GARCH model trained successfully!")

            except Exception as e:
                st.error(f"Error training GARCH model: {e}")

                st.write("### Model Summary")
                st.text(garch_model.summary())

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

    if 'selected_model' in st.session_state:
        st.write(f"**Currently viewing: {st.session_state.selected_model}**")