import os
import sys
import pandas as pd
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import generate_technical_indicators

def retrain_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "DSE_Data.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_dse.cbm")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at '{DATA_PATH}'")
        return

    print("Loading updated dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Engineering
    df = generate_technical_indicators(df)
    df.dropna(inplace=True)
    
    # 2. Define Features & Target
    feature_columns = ["RSI", "MACD", "SMA_20", "Volume"]
    X = df[feature_columns]
    y = df["Close"]
    
    # 3. Model Training
    print("Retraining CatBoost model on latest data...")
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=100
    )
    model.fit(X, y)
    
    # 4. Save/Overwrite Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model successfully retrained and saved to '{MODEL_PATH}'!\n")

if __name__ == "__main__":
    retrain_model()
