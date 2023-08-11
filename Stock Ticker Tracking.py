import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_additional_features(data):
    data['5D MA'] = data['Close'].rolling(window=5).mean()
    return data

def main():
    st.title("Stock Price Prediction")

    ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")
    start_date = st.date_input("Start Date:", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date:", pd.to_datetime("2023-08-10"))

    if start_date >= end_date:
        st.error("End Date must be after Start Date")
        return

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        st.warning("No data available for the specified period.")
        return

    stock_data = calculate_additional_features(stock_data)

    # Drop rows with missing values
    stock_data.dropna(inplace=True)

    # Display statistics summary
    st.write("Statistics Summary:")
    statistics = stock_data.describe()
    st.write(statistics)

    # Display whisker box plot
    st.write("Whisker Box Plot:")
    fig, ax = plt.subplots()
    sns.boxplot(data=stock_data, y='Close')
    plt.xlabel('Close')
    plt.title('Whisker Box Plot of Close Price')
    st.pyplot(fig)

    # Display data table
    st.write("Stock Data:")
    st.write(stock_data)

    # Model Training and Prediction
    train_size = int(0.8 * len(stock_data))
    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[['Close', '5D MA']])

    X_train, y_train = [], []

    look_back = 10

    for i in range(look_back, len(train_scaled)):
        X_train.append(train_scaled[i - look_back:i])
        y_train.append(train_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=64)

    # Model Evaluation and Prediction
    test_scaled = scaler.transform(test_data[['Close', '5D MA']])
    X_test, y_test = [], []

    for i in range(look_back, len(test_scaled)):
        X_test.append(test_scaled[i - look_back:i])
        y_test.append(test_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(np.column_stack((test_predictions, np.zeros_like(test_predictions))))[:, 0]

    mse = mean_squared_error(test_data['Close'][look_back:], test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data['Close'][look_back:], test_predictions)

    st.write("Model Evaluation:")
    st.write("Mean Squared Error:", mse)
    st.write("Root Mean Squared Error:", rmse)
    st.write("Mean Absolute Error:", mae)

    # Display stock price and moving average chart for different time periods
    st.write("Stock Price and Moving Average Chart:")
    selected_period = st.selectbox("Select Time Period:", ["1W", "1M"])

    fig = go.Figure()

    period_data = stock_data.resample(selected_period).ffill()

    # Create traces for stock price and 5D MA
    fig.add_trace(go.Scatter(x=period_data.index, y=period_data['Close'], mode='lines+markers', name='Stock Price', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=period_data.index, y=period_data['5D MA'], mode='lines', name='5-Day Moving Avg', line=dict(color='orange')))

    # Set up layout with legend at the best location
    fig.update_layout(
        title=f'Stock Price and 5-Day Moving Avg for {ticker} - {selected_period}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0.05, y=1)  # Adjust the x-coordinate value as needed
    )

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
