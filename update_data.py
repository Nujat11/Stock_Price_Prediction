import os
import pandas as pd
from bdshare import get_current_trade_data

def update_dse_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "DSE_Data.csv")

    print("Fetching current market trade data from DSE...")
    try:
        # Get latest market data
        latest_df = get_current_trade_data()
    except Exception as e:
        print(f"Error fetching data from DSE: {e}")
        return

    # Standardize column names to match DSE_Data.csv schema
    # Expected schema: Date, Trading_Code, Close, High, Low, Open, Volume, etc.
    if 'symbol' in latest_df.columns:
        latest_df = latest_df.rename(columns={
            'symbol': 'Trading_Code',
            'ltp': 'Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume',
            'date': 'Date'
        })

    # Add today's date if not present
    if 'Date' not in latest_df.columns:
        latest_df['Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Keep relevant columns
    cols_to_keep = ['Date', 'Trading_Code', 'Close', 'High', 'Low', 'Open', 'Volume']
    available_cols = [c for c in cols_to_keep if c in latest_df.columns]
    latest_df = latest_df[available_cols]

    # Clean numeric fields
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        if col in latest_df.columns:
            latest_df[col] = pd.to_numeric(latest_df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Load existing dataset and append
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_csv(DATA_PATH)
        
        # Combine existing and new data, avoiding duplicate Date + Trading_Code records
        combined_df = pd.concat([existing_df, latest_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['Date', 'Trading_Code'], keep='last', inplace=True)
        combined_df.sort_values(by=['Trading_Code', 'Date'], inplace=True)
        
        combined_df.to_csv(DATA_PATH, index=False)
        print(f"Dataset successfully updated at '{DATA_PATH}'!")
    else:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        latest_df.to_csv(DATA_PATH, index=False)
        print(f"New dataset created at '{DATA_PATH}'!")

if __name__ == "__main__":
    update_dse_dataset()
