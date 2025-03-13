import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
import hashlib  # Import hashlib for password hashing

# ---------------- SETUP ----------------
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor App")

USER_FILE = "users.csv"
PREDICTIONS_FILE = "predictions.csv"
STOCK_DATA_FILE = "stock_data.csv"  # New file for stock data
MODEL_PATH = "Latest_stock_price_model.keras"

# Global variable to track Keras availability
keras_available = False
model = None # Initialize model to None

# ---------------- MODEL LOADING FUNCTION ----------------
def load_keras_model():
    global model, keras_available
    try:
        model = load_model(MODEL_PATH)
        keras_available = True
        st.success("✅ Keras model loaded successfully!")
    except Exception as e:
        st.warning(f"⚠️ Keras model not loaded: {e}. ARIMA model will be used.")
        keras_available = False

# ---------------- USER AUTHENTICATION FUNCTIONS ----------------
def init_csv_files():
    if not os.path.exists(USER_FILE):
        pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)
    if not os.path.exists(PREDICTIONS_FILE):
        pd.DataFrame(columns=["username", "ticker", "date", "predictions"]).to_csv(PREDICTIONS_FILE, index=False)
    if not os.path.exists(STOCK_DATA_FILE): # Initialize stock data file
        pd.DataFrame(columns=["ticker", "date", "Open", "High", "Low", "Close", "MA_100", "MA_200", "MA_250"]).to_csv(STOCK_DATA_FILE, index=False)

def register_user(username, password):
    users = pd.read_csv(USER_FILE)
    if username in users["username"].values:
        st.error("❌ Username already exists. Please choose another one.")
        return False
    else:
        hashed_password = hashlib.sha256(password.encode()).hexdigest() # Hash password
        new_user = pd.DataFrame([[username, hashed_password]], columns=["username", "password"])
        new_user.to_csv(USER_FILE, mode='a', header=False, index=False)
        st.success("✅ Registered successfully! You can now log in.")
        return True

def authenticate_user(username, password):
    users = pd.read_csv(USER_FILE)
    user_row = users[users["username"] == username]
    if not user_row.empty:
        stored_password_hash = user_row["password"].iloc[0]
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password == stored_password_hash
    return False

def handle_login_logout():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if not st.session_state.logged_in:
        with st.container(): # Use container for better layout
            st.subheader("🔑 Login or Register")
            col1, col2 = st.columns([1,1]) # Adjust column ratio as needed
            with col1:
                username = st.text_input("Username", key="login_username") # Unique key for username in login
                password = st.text_input("Password", type="password", key="login_password") # Unique key for password in login
                if st.button("Login"):
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
            with col2:
                reg_username = st.text_input("Username", key="reg_username") # Unique key for username in register
                reg_password = st.text_input("Password", type="password", key="reg_password") # Unique key for password in register
                if st.button("Register"):
                    if reg_username and reg_password:
                        register_user(reg_username, reg_password)
                    else:
                        st.error("❌ Please enter a username and password to register.")
        return False # Not logged in
    else: # User is logged in
        st.subheader(f"Welcome, {st.session_state.username}! 🎉")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        return True # Logged in

# ---------------- STOCK DATA FUNCTIONS ----------------
def fetch_stock_data(stock_ticker):
    end_date = datetime.now()
    start_date = datetime(end_date.year - 20, end_date.month, end_date.day)
    try:
        with st.spinner(f"Fetching stock data for {stock_ticker}..."): # Show spinner during data fetch
            df = yf.download(stock_ticker, start=start_date, end=end_date)
        if df.empty:
            st.error(f"⚠️ No stock data available for {stock_ticker}. Please check the ticker or try a different one.")
            return None
        
        # Keep all relevant price data
        df = df[['Open', 'High', 'Low', 'Close']]
        
        # Calculate Moving Averages for Close price
        for days in [100, 200, 250]:
            df[f'MA_{days}'] = df['Close'].rolling(days).mean()

        # Save fetched stock data to CSV
        df_to_save = df.copy()
        df_to_save['ticker'] = stock_ticker
        df_to_save['date'] = df_to_save.index
        df_to_save.reset_index(drop=True, inplace=True)
        df_to_save.to_csv(STOCK_DATA_FILE, mode='a', header=not os.path.exists(STOCK_DATA_FILE), index=False)

        return df
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data for {stock_ticker}: {e}")
        return None

def display_stock_data(df):
    if df is not None:
        st.subheader("Stock Data")
        
        # Create price chart
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['Close'], label='Close', color='blue')
        ax.plot(df.index, df['Open'], label='Open', color='green', alpha=0.6)
        ax.plot(df.index, df['High'], label='High', color='red', alpha=0.6)
        ax.plot(df.index, df['Low'], label='Low', color='orange', alpha=0.6)
        ax.legend()
        ax.set_title('Historical Stock Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        st.pyplot(fig)
        
        # Display raw data in a table
        st.write("Raw Data:")
        st.write(df)

def display_moving_averages(df):
    if df is not None:
        st.subheader('Moving Averages Analysis')
        
        # Create moving averages chart
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        
        ma_cols = [col for col in df.columns if 'MA_' in col]
        colors = ['red', 'green', 'orange']
        
        for ma_col, color in zip(ma_cols, colors):
            ax.plot(df.index, df[ma_col], label=ma_col, color=color, alpha=0.7)
        
        ax.legend()
        ax.set_title('Moving Averages vs Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        st.pyplot(fig)
        
        # Display moving averages data
        st.write("Moving Averages Data:")
        ma_data = df[['Close'] + ma_cols].tail(10)  # Show last 10 days
        st.write(ma_data)

# ---------------- ARIMA PREDICTION FUNCTIONS ----------------
def predict_arima(df, stock_ticker, username): # Added stock_ticker and username
    if df is None:
        return None, None
    try:
        with st.spinner("Running ARIMA prediction..."): # Show spinner during ARIMA prediction
            arima_model = ARIMA(df['Close'].dropna(), order=(5, 1, 0))  # Example order, handle NaN values
            arima_fit = arima_model.fit()
            forecast_steps = 30
            arima_forecast = arima_fit.forecast(steps=forecast_steps)
            arima_dates = pd.date_range(df['Close'].dropna().index[-1], periods=forecast_steps, freq='B')  # Business days, use non-NA index
            arima_predictions_df = pd.DataFrame({'Date': arima_dates, 'Predicted Price': arima_forecast})

            # Save ARIMA predictions
            predictions_to_save = arima_predictions_df.copy()
            predictions_to_save['username'] = username
            predictions_to_save['ticker'] = stock_ticker
            predictions_to_save['date'] = datetime.now().strftime('%Y-%m-%d') # Prediction date
            predictions_to_save.to_csv(PREDICTIONS_FILE, mode='a', header=not os.path.exists(PREDICTIONS_FILE), index=False)


            return arima_predictions_df, arima_fit # Return fit object if needed later
    except Exception as e:
        st.error(f"⚠️ Error with ARIMA prediction: {e}")
        return None, None

def display_arima_predictions(arima_predictions_df, df):
    if arima_predictions_df is not None and df is not None:
        st.subheader("ARIMA Predictions")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['Close'], label='Actual Close')
        ax.plot(arima_predictions_df['Date'], arima_predictions_df['Predicted Price'], label='ARIMA Predictions', color='red')
        ax.legend()
        ax.set_title('ARIMA Predictions')
        st.pyplot(fig)
        st.write(arima_predictions_df)
        st.line_chart(arima_predictions_df.set_index("Date"), height=300, use_container_width=True) # Adjust height and width

# ---------------- KERAS PREDICTION FUNCTIONS ----------------
def predict_keras(df, model, stock_ticker, username): # Added stock_ticker and username
    if not keras_available or model is None or df is None:
        return None, None, None

    try:
        with st.spinner("Running Keras model prediction..."): # Show spinner during Keras prediction
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']])

            time_step = 100  # Adjust as needed
            x_data, y_data = [], []
            for i in range(time_step, len(scaled_data)):
                x_data.append(scaled_data[i - time_step:i])
                y_data.append(scaled_data[i])
            x_data, y_data = np.array(x_data), np.array(y_data)

            predictions = model.predict(x_data, verbose=0) # Disable verbose output during prediction
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)

            ploting_data = pd.DataFrame({'Original': inv_y_test.flatten(), 'Predicted': inv_pre.flatten()},
                                            index=df.index[len(df) - len(y_data):])  # Correct indexing

            future_predictions = []
            last_sequence = scaled_data[-time_step:]  # Use the last 'time_step' values
            current_input = last_sequence.reshape(1, time_step, 1)

            future_days = 30
            for _ in range(future_days):
                predicted_price = model.predict(current_input, verbose=0) # Disable verbose output during prediction
                future_predictions.append(predicted_price[0, 0])
                current_input = np.append(current_input[:, 1:, :], [[[predicted_price[0, 0]]]], axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(df.index[-1], periods=future_days, freq='B')
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})

            # Save Keras predictions (both historical and future) - Future only for now to keep it simple, can add historical later if needed
            predictions_to_save = future_df.copy()
            predictions_to_save['username'] = username
            predictions_to_save['ticker'] = stock_ticker
            predictions_to_save['date'] = datetime.now().strftime('%Y-%m-%d') # Prediction date
            predictions_to_save.to_csv(PREDICTIONS_FILE, mode='a', header=not os.path.exists(PREDICTIONS_FILE), index=False)


            return ploting_data, future_df, df # Return actual df for plotting

    except Exception as e:
        st.error(f"⚠️ Error with Keras model prediction: {e}")
        return None, None, None


def display_keras_predictions(ploting_data, future_df, df):
    if ploting_data is not None and future_df is not None and df is not None:
        st.subheader("Keras Model Predictions")

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['Close'], label='Actual Close')  # Plot actual values
        ax.plot(ploting_data.index, ploting_data['Predicted'], label='Keras Predictions (Historical)', color='red')
        ax.legend()
        ax.set_title('Keras Model - Historical Predictions vs Actual')
        st.pyplot(fig)
        st.write(ploting_data)
        st.line_chart(ploting_data[["Original", "Predicted"]], height=300, use_container_width=True) # Historical prediction chart

        st.subheader("Keras Model - Future Predictions")
        st.write(future_df)
        st.line_chart(future_df.set_index("Date"), height=300, use_container_width=True) # Future prediction chart


# ---------------- MAIN APP FLOW ----------------
def main():
    init_csv_files()
    load_keras_model() # Load model at app start

    if handle_login_logout(): # Only proceed if user is logged in
        stock_ticker_input = st.text_input("Enter Stock ID (e.g., GOOG, AAPL)", "GOOG").upper()
        stock_ticker = stock_ticker_input.strip()

        if not stock_ticker: # Input validation for empty ticker
            st.warning("Please enter a stock ticker symbol.")
            return

        df = fetch_stock_data(stock_ticker)
        display_stock_data(df)
        display_moving_averages(df)

        arima_predictions_df, _ = predict_arima(df, stock_ticker, st.session_state.username) # Pass ticker and username
        display_arima_predictions(arima_predictions_df, df)


        if keras_available:
            ploting_data, future_df, actual_df_keras = predict_keras(df, model, stock_ticker, st.session_state.username) # Pass ticker and username
            display_keras_predictions(ploting_data, future_df, actual_df_keras)


if __name__ == "__main__":
    main()
