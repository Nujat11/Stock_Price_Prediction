import os
import sys
import pandas as pd
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import generate_technical_indicators


def train_and_save_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "DSE_Data.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "catboost_dse.cbm")

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print("Generating Technical Indicators...")
    df = generate_technical_indicators(df)

    feature_columns = ["RSI", "MACD", "SMA_20", "Volume"]
    target_column = "Close"

    X = df[feature_columns]
    y = df[target_column]

    print(f"Training CatBoost model on full dataset ({len(df)} rows)...")
    model = CatBoostRegressor(verbose=100, random_seed=42, thread_count=-1)
    model.fit(X, y)

    model.save_model(MODEL_PATH)
    print(f"\nModel successfully trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()