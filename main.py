import os
import sys
import pandas as pd
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.validation.backtest import WalkForwardValidator


def generate_technical_indicators(df):
    """Function to generate technical indicators per stock."""
    df = df.copy()

    # Ensure Date sorting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Trading_Code", "Date"]).reset_index(drop=True)

    # 1. Simple Moving Average (20 periods)
    df["SMA_20"] = df.groupby("Trading_Code")["Close"].transform(
        lambda x: x.rolling(window=20).mean()
    )

    # 2. Relative Strength Index (14 periods)
    delta = df.groupby("Trading_Code")["Close"].diff()
    gain = (
        delta.where(delta > 0, 0)
        .groupby(df["Trading_Code"])
        .transform(lambda x: x.rolling(window=14).mean())
    )
    loss = (
        (-delta.where(delta < 0, 0))
        .groupby(df["Trading_Code"])
        .transform(lambda x: x.rolling(window=14).mean())
    )
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 3. MACD (12, 26)
    exp1 = df.groupby("Trading_Code")["Close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    exp2 = df.groupby("Trading_Code")["Close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df["MACD"] = exp1 - exp2

    # 4. Bollinger Bands (20 periods, 2 std devs)
    rolling_std = df.groupby("Trading_Code")["Close"].transform(
        lambda x: x.rolling(window=20).std()
    )
    df["BB_Upper"] = df["SMA_20"] + (rolling_std * 2)
    df["BB_Lower"] = df["SMA_20"] - (rolling_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"]

    # 5. Stochastic Oscillator (14, 3)
    low_14 = df.groupby("Trading_Code")["Low"].transform(
        lambda x: x.rolling(window=14).min()
    )
    high_14 = df.groupby("Trading_Code")["High"].transform(
        lambda x: x.rolling(window=14).max()
    )
    df["Stoch_%K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["Stoch_%D"] = df.groupby("Trading_Code")["Stoch_%K"].transform(
        lambda x: x.rolling(window=3).mean()
    )

    # Drop NaN rows created by rolling calculations
    indicator_cols = [
        "RSI",
        "MACD",
        "SMA_20",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "Stoch_%K",
        "Stoch_%D",
    ]
    df = df.dropna(subset=indicator_cols).reset_index(drop=True)
    return df


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "DSE_Data.csv")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset missing at: {DATA_PATH}")

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(
        "Generating Features (RSI, MACD, SMA_20, Bollinger Bands, Stochastic)..."
    )
    df = generate_technical_indicators(df)

    # Filter dataset for a single stock (GP) for fast validation
    print("Filtering data for 'GP' stock...")
    df = df[df["Trading_Code"] == "GP"].reset_index(drop=True)

    feature_columns = [
        "RSI",
        "MACD",
        "SMA_20",
        "Volume",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "Stoch_%K",
        "Stoch_%D",
    ]
    target_column = "Close"

    print(f"Dataset ready with {len(df)} rows for GP. Initializing Model...")

    wfv = WalkForwardValidator(
        train_window_size=500, test_window_size=30, step_size=30
    )

    model = CatBoostRegressor(verbose=0, random_seed=42, thread_count=-1)

    print("Running Walk-Forward Validation for GP...")
    metrics = wfv.evaluate_model(model, df, feature_columns, target_column)

    print("\n" + "=" * 40)
    print(" Walk-Forward Validation Results (GP) ")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()