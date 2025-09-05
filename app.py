import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objs as go

# Set page config
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Price Prediction App")
st.markdown("""
This app uses an LSTM neural network to predict future stock prices.
Select a stock symbol and the number of days to predict.
""")

# Sidebar for user input
st.sidebar.header("User Input")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
period = st.sidebar.selectbox("Historical Data Period", 
                             ["1mo", "3mo", "6mo", "1y", "2y"], 
                             index=3)
days_to_predict = st.sidebar.slider("Days to Predict", 7, 60, 30)
lookback = st.sidebar.slider("Lookback Period (days)", 30, 90, 60)

# Function to fetch data
@st.cache_data
def fetch_data(symbol, period):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data

# Function to prepare data
def prepare_data(data, lookback):
    df = data[['Close']].copy()
    df.columns = ['price']
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['price']])
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, df

# Function to build model
def build_model(lookback):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future prices
def predict_future(model, data, scaler, lookback, days):
    last_sequence = data[-lookback:].values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    
    predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

# Main app logic
if st.sidebar.button("Predict"):
    # Fetch data
    with st.spinner('Fetching data...'):
        data = fetch_data(symbol, period)
    
    if data.empty:
        st.error("No data found for this symbol. Please try a different one.")
    else:
        # Prepare data
        with st.spinner('Preparing data...'):
            X, y, scaler, df = prepare_data(data, lookback)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train model
        with st.spinner('Training model...'):
            model = build_model(lookback)
            history = model.fit(X_train, y_train, 
                               batch_size=32, 
                               epochs=50, 
                               validation_data=(X_test, y_test),
                               verbose=0)
        
        # Make predictions
        with st.spinner('Making predictions...'):
            future_prices = predict_future(model, df[['price']], scaler, lookback, days_to_predict)
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['price'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{symbol} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction table
        st.subheader("Prediction Details")
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_prices
        })
        st.dataframe(prediction_df)
        
        # Model performance
        st.subheader("Model Performance")
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        col1, col2 = st.columns(2)
        col1.metric("Training Loss", f"{train_loss:.6f}")
        col2.metric("Validation Loss", f"{val_loss:.6f}")

# Display raw data
if st.sidebar.checkbox("Show Raw Data"):
    data = fetch_data(symbol, period)
    st.subheader("Raw Data")
    st.write(data)