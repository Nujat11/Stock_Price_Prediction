# Stock Price Prediction Using Statistical Analysis, Machine Learning, and Deep Learning

Forecast the **next-day closing price** of stocks listed on the **Dhaka Stock Exchange (DSE)**, with a primary case study on **BEXIMCO**. The pipeline combines time-series diagnostics, technical features, gradient-boosting models, and sequence networks (BiLSTM and Transformer).

## Project description

Stock prices are non-stationary, noisy, and driven by regime shifts. This project:

1. Cleans multi-decade DSE daily bars (open, high, low, close, volume).
2. Tests stationarity and structural breaks (ADF, Phillips–Perron, CUSUM).
3. Builds next-day close models for a **single ticker** (BEXIMCO) and for **many tickers** together.
4. Compares linear, tree, boosting, and deep models on a chronological hold-out set.

The original notebooks used OLS, Random Forest, XGBoost, and a vanilla LSTM. They have been rebuilt so that splits do not leak future data, neural nets scale the **target** as well as the features, and newer models (LightGBM, CatBoost, BiLSTM, Transformer) are included.

## Dataset

Historical daily trading records from the Dhaka Stock Exchange:

| Column | Description |
| --- | --- |
| `Trading_Code` | Ticker |
| `Date` | Trading date |
| `Open`, `High`, `Low`, `Close` | Prices (BDT) |
| `Volume` | Traded volume |

Source: [Mendeley — DSE daily data](https://data.mendeley.com/datasets/23553sm4tn/4)

Place the raw file at `data/DSE_raw.csv`, or upload it when running the cleaning notebook in Colab.

## Repository layout

```
Stock-Price-Prediction/
├── README.md
├── requirements.txt
├── data/                              # raw + cleaned CSVs (not committed)
├── src/
│   ├── features.py                    # RSI, MACD, ATR, lags, returns, …
│   ├── split.py                       # chronological and calendar-date splits
│   ├── metrics.py                     # R², RMSE, MAE, MAPE, direction accuracy
│   └── models.py                      # Ridge, trees, boosters, BiLSTM, Transformer
└── Stock Price Prediction/
    ├── DSA_Data_Cleaning_.ipynb       # cleaning + EDA
    ├── Single_Stock_Close_Price.ipynb # BEXIMCO next-day close
    └── MultiStock_Prediction.ipynb    # pooled multi-ticker model
```

Notebooks add the repo root to `sys.path` so `src` imports work from Colab or a local Jupyter kernel.

## Setup

```bash
pip install -r requirements.txt
```

Main dependencies: pandas, scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, statsmodels, matplotlib, plotly.

## How to run

1. **Clean** — `DSA_Data_Cleaning_.ipynb`  
   Per-stock outlier clipping, invalid-row removal, tickers with enough history. Writes `data/Cleaned_DSE_Data.csv` and `Cleaned.csv`.

2. **Single stock** — `Single_Stock_Close_Price.ipynb`  
   BEXIMCO (falls back to the most frequent ticker if BEXIMCO is missing). Train / validation / test by time. Fits tabular models, an inverse-RMSE ensemble, BiLSTM, and a Transformer encoder.

3. **Multi-stock** — `MultiStock_Prediction.ipynb`  
   Shared **calendar** cutoffs for every ticker (not a row-index split after sorting by name). Trains Ridge, XGBoost, LightGBM, CatBoost, HistGBM, then an ensemble.

## Features

Built per ticker, then aligned to the next-day close (`Target_Close`):

- Returns, log returns, high–low and close–open spreads  
- Close and return lags (1, 2, 3, 5, 7, 14)  
- SMA / EMA / volume SMA and rolling volatility (7, 14, 21)  
- RSI(14), MACD, Bollinger %B, ATR(14), volume z-score  
- Day of week and month  

## Models

| Family | Models |
| --- | --- |
| Linear | Ridge |
| Trees | Random Forest, HistGradientBoosting |
| Boosting | XGBoost, LightGBM, CatBoost (early stopping on the validation window) |
| Deep | Bidirectional LSTM; Transformer encoder (Huber loss, early stopping, LR schedule) |
| Combine | Inverse-RMSE weighted ensemble of the three best tabular models |

Sequence models scale **X and y on the training window only**, then invert predictions to price space. That is the main reason the old LSTM underperformed a linear baseline.

## Evaluation

Reported on the **test dates** (not shuffled rows):

- R², RMSE, MAE, MAPE  
- **Direction accuracy** — whether the predicted move from today’s close matches the actual next-day move  

Next-day close is highly persistent, so R² on price can look strong even when the **return** is hard to forecast. Use RMSE and direction accuracy to compare models.

## Statistical diagnostics (single-stock notebook)

- Augmented Dickey–Fuller (ADF)  
- Phillips–Perron (regression-based statistic)  
- CUSUM on OLS residuals for structural breaks  
- Price path plot for BEXIMCO  

Raw close is typically non-stationary; that is expected for levels and is why the pipeline uses returns and technical features rather than modelling the raw series alone.

## Notes on the upgrade

- **No ticker-order leakage** in the multi-stock split.  
- **No full-sample scaler** on LSTM inputs or targets.  
- Cleaning clips outliers **inside each stock**, so expensive tickers are not crushed by a global quantile.  
- Colab file upload still works; local CSV paths are checked first.  
