import argparse
import os
import sys
import pandas as pd
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import generate_technical_indicators


def predict_next_day(trading_code):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "DSE_Data.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_dse.cbm")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load Model
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # Load and Preprocess Data
    print(f"Loading latest data for '{trading_code}'...")
    df = pd.read_csv(DATA_PATH)
    df = generate_technical_indicators(df)

    stock_df = df[df["Trading_Code"] == trading_code].sort_values("Date")

    if stock_df.empty:
        print(f"Error: Stock '{trading_code}' not found in dataset.")
        return

    # Extract Most Recent Trading Record
    latest_record = stock_df.iloc[-1]
    last_date = latest_record["Date"]
    last_close = latest_record["Close"]

    feature_columns = ["RSI", "MACD", "SMA_20", "Volume"]
    input_features = latest_record[feature_columns].to_frame().T

    # Predict Next Day Close Price
    predicted_close = model.predict(input_features)[0]
    price_change = predicted_close - last_close
    percent_change = (price_change / last_close) * 100
    direction = "UP 📈" if price_change > 0 else "DOWN 📉"

    # Display Results
    print("\n" + "=" * 45)
    print(f" Stock Forecast: {trading_code}")
    print("=" * 45)
    print(f"Last Available Date : {last_date}")
    print(f"Last Closing Price  : {last_close:.2f} BDT")
    print("-" * 45)
    print(f"Predicted Next Close: {predicted_close:.2f} BDT")
    print(
        f"Expected Movement   : {direction} ({price_change:+.2f} BDT / {percent_change:+.2f}%)"
    )
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Next-Day Stock Close Price"
    )
    parser.add_argument(
        "--code",
        type=str,
        default="GP",
        help="Stock Trading_Code (e.g., GP, BATBC, SQURPHARMA)",
    )
    args = parser.parse_args()

    predict_next_day(args.code)