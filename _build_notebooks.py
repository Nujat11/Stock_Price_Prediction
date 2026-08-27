"""Build updated Colab/local notebooks for the stock prediction project."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "Stock Price Prediction"


def cell(cell_type: str, source: str) -> dict:
    lines = source.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src,
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


CLEANING = [
    cell("markdown", """# DSE Data Cleaning

Works on **Google Colab** (file upload) and **locally** (CSV path). Cleaning is done **per stock** so high-priced tickers are not clipped by the global 99.9th percentile."""),
    cell("code", """# Optional on Colab
# %pip install -q pandas numpy scipy plotly"""),
    cell("code", r'''import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from IPython.display import display

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt


def load_raw_dse():
    candidates = [
        Path("data/DSE_raw.csv"),
        Path("../data/DSE_raw.csv"),
        Path("Cleaned.csv"),
        Path("Cleaned_DSE_Data.csv"),
        Path("../Cleaned.csv"),
    ]
    for path in candidates:
        if path.exists():
            print(f"Loaded {path.resolve()}")
            return pd.read_csv(path, low_memory=False)

    try:
        from google.colab import files
        uploaded = files.upload()
        name = next(iter(uploaded))
        return pd.read_csv(name, low_memory=False)
    except Exception as exc:
        raise FileNotFoundError(
            "Place the Mendeley DSE CSV as data/DSE_raw.csv or upload it in Colab."
        ) from exc


df_raw = load_raw_dse()
df_raw.head()'''),
    cell("code", r'''df = df_raw.copy()
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

rename = {}
for col in df.columns:
    low = col.lower()
    if low in {"tradingcode", "trading_code", "ticker", "symbol"}:
        rename[col] = "Trading_Code"
    elif low == "date":
        rename[col] = "Date"
    elif low in {"open", "opening"}:
        rename[col] = "Open"
    elif low in {"high"}:
        rename[col] = "High"
    elif low in {"low"}:
        rename[col] = "Low"
    elif low in {"close", "closing", "ltp"}:
        rename[col] = "Close"
    elif "volume" in low:
        rename[col] = "Volume"
df = df.rename(columns=rename)

needed = ["Trading_Code", "Date", "Open", "High", "Low", "Close", "Volume"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Trading_Code"])
for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
df = df[~((df["Open"] == 0) & (df["High"] == 0) & (df["Low"] == 0) & (df["Close"] == 0))]
df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
df = df.drop_duplicates(subset=["Trading_Code", "Date"])
df = df.sort_values(["Trading_Code", "Date"]).reset_index(drop=True)

print(df.info())
print("Missing:\n", df.isna().sum())'''),
    cell("code", r'''def clip_outliers_per_stock(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        q_low, q_high = g[col].quantile(0.001), g[col].quantile(0.999)
        g[col] = g[col].clip(lower=q_low, upper=q_high)
    return g


df = df.groupby("Trading_Code", group_keys=False).apply(clip_outliers_per_stock)

stock_lengths = df.groupby("Trading_Code").size()
valid = stock_lengths[stock_lengths >= 250].index
df = df[df["Trading_Code"].isin(valid)].reset_index(drop=True)

print("Usable stocks:", df["Trading_Code"].nunique())
print("Rows:", len(df))
print("Date range:", df["Date"].min(), "→", df["Date"].max())'''),
    cell("code", r'''out_dir = Path("data")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "Cleaned_DSE_Data.csv"
df.to_csv(out_path, index=False)
df.to_csv("Cleaned.csv", index=False)
print("Saved", out_path.resolve(), "and Cleaned.csv")

try:
    from google.colab import files
    files.download(str(out_path))
except ImportError:
    pass'''),
    cell("code", r'''summary = pd.DataFrame({
    "Variable": ["Closing Price", "Volume", "Log Close", "Log Volume"],
    "Max": [df["Close"].max(), df["Volume"].max(), np.log1p(df["Close"]).max(), np.log1p(df["Volume"]).max()],
    "Min": [df["Close"].min(), df["Volume"].min(), np.log1p(df["Close"]).min(), np.log1p(df["Volume"]).min()],
    "Mean": [df["Close"].mean(), df["Volume"].mean(), np.log1p(df["Close"]).mean(), np.log1p(df["Volume"]).mean()],
    "SD": [df["Close"].std(), df["Volume"].std(), np.log1p(df["Close"]).std(), np.log1p(df["Volume"]).std()],
    "Skewness": [skew(df["Close"]), skew(df["Volume"]), skew(np.log1p(df["Close"])), skew(np.log1p(df["Volume"]))],
})
display(summary.round(3))'''),
    cell("code", r'''monthly = (
    df.assign(Month=df["Date"].dt.to_period("M").astype(str))
      .groupby("Month", as_index=False)
      .agg(Close=("Close", "mean"), Volume=("Volume", "sum"))
)

if HAS_PLOTLY:
    px.line(monthly, x="Month", y="Close", title="Average monthly closing price").show()
    px.line(monthly, x="Month", y="Volume", title="Total monthly volume", log_y=True).show()
    px.imshow(df[["Open", "High", "Low", "Close", "Volume"]].corr(), text_auto=True,
              color_continuous_scale="Blues", title="Price/volume correlation").show()
else:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(monthly["Month"]), monthly["Close"])
    ax.set_title("Average monthly closing price")
    plt.show()'''),
]

SINGLE = [
    cell("markdown", """# BEXIMCO Next-Day Close Prediction

Updated pipeline:
- richer technical features
- chronological train / validation / test split
- **Ridge, Random Forest, XGBoost, LightGBM, CatBoost, HistGBM**
- **BiLSTM** and a **Transformer encoder** with *train-only* feature **and target** scaling
- stacked ensemble of the best tabular models

Neural nets previously underperformed mainly because the close price target was unscaled."""),
    cell("code", """# %pip install -q pandas numpy scikit-learn xgboost lightgbm catboost statsmodels tensorflow matplotlib"""),
    cell("code", r'''import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import adfuller

ROOT = Path.cwd()
for candidate in [ROOT, ROOT.parent, ROOT.parent.parent]:
    if (candidate / "src" / "features.py").exists():
        sys.path.insert(0, str(candidate))
        break

from src.features import FEATURE_COLUMNS, add_technical_features
from src.metrics import regression_metrics
from src.models import build_bilstm, build_tabular_models, build_transformer, create_sequences
from src.split import time_split'''),
    cell("code", r'''def load_cleaned():
    for path in [
        Path("data/Cleaned_DSE_Data.csv"),
        Path("../data/Cleaned_DSE_Data.csv"),
        Path("Cleaned.csv"),
        Path("Cleaned_DSE_Data.csv"),
        Path("../Cleaned.csv"),
    ]:
        if path.exists():
            print("Loaded", path)
            return pd.read_csv(path, parse_dates=["Date"])
    raise FileNotFoundError("Run the cleaning notebook first, or place Cleaned.csv next to this notebook.")


df = load_cleaned()
stock = "BEXIMCO" if "BEXIMCO" in set(df["Trading_Code"]) else df["Trading_Code"].value_counts().idxmax()
print("Using ticker:", stock)

bex = df[df["Trading_Code"] == stock].sort_values("Date").reset_index(drop=True)
bex = add_technical_features(bex)
bex = bex.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
print("Rows after features:", len(bex))
bex.tail()'''),
    cell("code", r'''feature_cols = [c for c in FEATURE_COLUMNS if c in bex.columns]
train_df, val_df, test_df = time_split(bex, train_ratio=0.70, val_ratio=0.15)
print(len(train_df), "train |", len(val_df), "val |", len(test_df), "test")

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(train_df[feature_cols])
X_val = x_scaler.transform(val_df[feature_cols])
X_test = x_scaler.transform(test_df[feature_cols])

y_train = train_df["Target_Close"].to_numpy()
y_val = val_df["Target_Close"].to_numpy()
y_test = test_df["Target_Close"].to_numpy()
close_test = test_df["Close"].to_numpy()

X_fit = np.vstack([X_train, X_val])
y_fit = np.concatenate([y_train, y_val])'''),
    cell("code", r'''models = build_tabular_models()
rows = []
preds = {}

for name, model in models.items():
    fit_kwargs = {}
    if name == "XGBoost":
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    elif name == "LightGBM":
        fit_kwargs = {"eval_set": [(X_val, y_val)]}
        try:
            from lightgbm import early_stopping
            fit_kwargs["callbacks"] = [early_stopping(80, verbose=False)]
        except Exception:
            pass
    elif name == "CatBoost":
        fit_kwargs = {"eval_set": (X_val, y_val)}

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_fit, y_fit)

    pred = model.predict(X_test)
    preds[name] = np.asarray(pred).ravel()
    metrics = regression_metrics(y_test, preds[name], close_today=close_test)
    metrics["Model"] = name
    rows.append(metrics)
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if k != "Model"})

tabular = pd.DataFrame(rows).set_index("Model")
tabular.sort_values("RMSE")'''),
    cell("code", r'''# Weighted ensemble of the three lowest-RMSE tabular models
top3 = tabular.nsmallest(3, "RMSE").index.tolist()
weights = 1.0 / tabular.loc[top3, "RMSE"]
weights = weights / weights.sum()
ensemble_pred = sum(weights[m] * preds[m] for m in top3)
preds["Ensemble"] = ensemble_pred
ens_metrics = regression_metrics(y_test, ensemble_pred, close_today=close_test)
ens_metrics["Model"] = "Ensemble"
print("Top-3:", top3)
print("Ensemble", {k: round(v, 4) for k, v in ens_metrics.items() if k != "Model"})'''),
    cell("markdown", """## Sequence models

Targets are standardized on the **training window only**, then inverted after prediction. That is the main LSTM upgrade versus the original notebook."""),
    cell("code", r'''LOOKBACK = 30
y_scaler.fit(y_train.reshape(-1, 1))

X_all = x_scaler.transform(bex[feature_cols])
y_all_scaled = y_scaler.transform(bex["Target_Close"].to_numpy().reshape(-1, 1)).ravel()

X_seq, y_seq = create_sequences(X_all, y_all_scaled, LOOKBACK)
idx = np.arange(LOOKBACK, len(bex))
train_mask = idx < len(train_df)
val_mask = (idx >= len(train_df)) & (idx < len(train_df) + len(val_df))
test_mask = idx >= len(train_df) + len(val_df)

X_tr_s, y_tr_s = X_seq[train_mask], y_seq[train_mask]
X_va_s, y_va_s = X_seq[val_mask], y_seq[val_mask]
X_te_s, y_te_s = X_seq[test_mask], y_seq[test_mask]
y_te_price = bex["Target_Close"].to_numpy()[idx[test_mask]]
close_te_seq = bex["Close"].to_numpy()[idx[test_mask]]
print(X_tr_s.shape, X_va_s.shape, X_te_s.shape)'''),
    cell("code", r'''from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
]

bilstm = build_bilstm(LOOKBACK, X_tr_s.shape[-1])
bilstm.fit(
    X_tr_s, y_tr_s,
    validation_data=(X_va_s, y_va_s),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

lstm_pred = y_scaler.inverse_transform(bilstm.predict(X_te_s, verbose=0)).ravel()
lstm_metrics = regression_metrics(y_te_price, lstm_pred, close_today=close_te_seq)
lstm_metrics["Model"] = "BiLSTM"
print(lstm_metrics)'''),
    cell("code", r'''transformer = build_transformer(LOOKBACK, X_tr_s.shape[-1])
transformer.fit(
    X_tr_s, y_tr_s,
    validation_data=(X_va_s, y_va_s),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

tf_pred = y_scaler.inverse_transform(transformer.predict(X_te_s, verbose=0)).ravel()
tf_metrics = regression_metrics(y_te_price, tf_pred, close_today=close_te_seq)
tf_metrics["Model"] = "Transformer"
print(tf_metrics)'''),
    cell("code", r'''results = pd.concat(
    [
        tabular.reset_index(),
        pd.DataFrame([ens_metrics, lstm_metrics, tf_metrics]),
    ],
    ignore_index=True,
)[["Model", "R2", "RMSE", "MAE", "MAPE", "Direction_Acc"]]
results = results.sort_values("RMSE").reset_index(drop=True)
display(results.round(4))

best_name = results.iloc[0]["Model"]
print("Best test RMSE:", best_name)'''),
    cell("code", r'''plot_pred = preds.get(best_name)
plot_dates = test_df["Date"]
plot_actual = y_test
if best_name == "BiLSTM":
    plot_pred, plot_dates, plot_actual = lstm_pred, bex["Date"].to_numpy()[idx[test_mask]], y_te_price
elif best_name == "Transformer":
    plot_pred, plot_dates, plot_actual = tf_pred, bex["Date"].to_numpy()[idx[test_mask]], y_te_price
elif best_name == "Ensemble":
    plot_pred = ensemble_pred

plt.figure(figsize=(14, 5))
plt.plot(plot_dates, plot_actual, label="Actual next close", linewidth=1.2)
plt.plot(plot_dates, plot_pred, label=f"{best_name} predicted", linewidth=1.2)
plt.title(f"{stock}: actual vs predicted next-day close")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()'''),
    cell("markdown", """## Statistical diagnostics on the raw close series"""),
    cell("code", r'''close = bex["Close"]
adf = adfuller(close)
print("ADF statistic:", adf[0], "p-value:", adf[1])

y = close.diff().dropna()
x = sm.add_constant(close.shift(1).dropna())
pp_stat = sm.OLS(y.iloc[1:], x.iloc[1:]).fit().tvalues.iloc[1]
print("PP statistic (custom):", float(pp_stat))

X_chow = sm.add_constant(bex[["Open", "High", "Low", "Volume"]])
chow_model = sm.OLS(bex["Close"], X_chow).fit()
print("CUSUM:", breaks_cusumolsresid(chow_model.resid))

plt.figure(figsize=(12, 4))
plt.plot(bex["Date"], bex["Close"])
plt.title(f"{stock} closing price")
plt.show()'''),
]

MULTI = [
    cell("markdown", """# Multi-stock next-day close prediction

The original notebook split by **row index after sorting by ticker**, so entire companies leaked into train or test. This version splits on **calendar dates** shared across all stocks, then trains LightGBM / XGBoost / CatBoost / Ridge."""),
    cell("code", """# %pip install -q pandas numpy scikit-learn xgboost lightgbm catboost matplotlib"""),
    cell("code", r'''import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path.cwd()
for candidate in [ROOT, ROOT.parent, ROOT.parent.parent]:
    if (candidate / "src" / "features.py").exists():
        sys.path.insert(0, str(candidate))
        break

from src.features import FEATURE_COLUMNS, add_grouped_features
from src.metrics import regression_metrics
from src.models import build_tabular_models
from src.split import global_date_split'''),
    cell("code", r'''def load_cleaned():
    for path in [
        Path("data/Cleaned_DSE_Data.csv"),
        Path("../data/Cleaned_DSE_Data.csv"),
        Path("Cleaned.csv"),
        Path("Cleaned_DSE_Data.csv"),
    ]:
        if path.exists():
            print("Loaded", path)
            return pd.read_csv(path, parse_dates=["Date"])
    raise FileNotFoundError("Run the cleaning notebook first.")


df = load_cleaned().sort_values(["Trading_Code", "Date"]).reset_index(drop=True)

# Keep liquid names so training stays tractable
counts = df.groupby("Trading_Code").size()
keep = counts[counts >= 800].index
df = df[df["Trading_Code"].isin(keep)].copy()
print("Stocks:", df["Trading_Code"].nunique(), "rows:", len(df))

df = add_grouped_features(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

le = LabelEncoder()
df["Stock_ID"] = le.fit_transform(df["Trading_Code"])
feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + ["Stock_ID"]
print("Features:", len(feature_cols))'''),
    cell("code", r'''train_df, val_df, test_df, train_cut, val_cut = global_date_split(df, train_ratio=0.70, val_ratio=0.15)
print("Train ≤", train_cut.date(), "| Val ≤", val_cut.date(), "| Test after that")
print(len(train_df), len(val_df), len(test_df))

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_val = scaler.transform(val_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
y_train = train_df["Target_Close"].to_numpy()
y_val = val_df["Target_Close"].to_numpy()
y_test = test_df["Target_Close"].to_numpy()
close_test = test_df["Close"].to_numpy()'''),
    cell("code", r'''wanted = ["Ridge", "XGBoost", "LightGBM", "CatBoost", "HistGBM"]
models = {k: v for k, v in build_tabular_models().items() if k in wanted}

rows, preds = [], {}
for name, model in models.items():
    fit_kwargs = {}
    if name == "XGBoost":
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    elif name == "LightGBM":
        try:
            from lightgbm import early_stopping
            fit_kwargs = {"eval_set": [(X_val, y_val)], "callbacks": [early_stopping(60, verbose=False)]}
        except Exception:
            fit_kwargs = {"eval_set": [(X_val, y_val)]}
    elif name == "CatBoost":
        fit_kwargs = {"eval_set": (X_val, y_val)}

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)

    pred = np.asarray(model.predict(X_test)).ravel()
    preds[name] = pred
    metrics = regression_metrics(y_test, pred, close_today=close_test)
    metrics["Model"] = name
    rows.append(metrics)
    print(name, {k: round(v, 4) for k, v in metrics.items() if k != "Model"})

results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
display(results)'''),
    cell("code", r'''top = results.nsmallest(3, "RMSE")["Model"].tolist()
w = 1.0 / results.set_index("Model").loc[top, "RMSE"]
w = w / w.sum()
ensemble = sum(w[m] * preds[m] for m in top)
ens = regression_metrics(y_test, ensemble, close_today=close_test)
ens["Model"] = "Ensemble"
results = pd.concat([results, pd.DataFrame([ens])], ignore_index=True).sort_values("RMSE")
display(results.round(4))

best = results.iloc[0]["Model"]
test_df = test_df.copy()
test_df["Predicted_Close"] = ensemble if best == "Ensemble" else preds[best]'''),
    cell("code", r'''ticker = "BEXIMCO" if "BEXIMCO" in set(test_df["Trading_Code"]) else test_df["Trading_Code"].iloc[0]
plot_df = test_df[test_df["Trading_Code"] == ticker].sort_values("Date")

plt.figure(figsize=(12, 5))
plt.plot(plot_df["Date"], plot_df["Target_Close"], label="Actual next close")
plt.plot(plot_df["Date"], plot_df["Predicted_Close"], label=f"{best} predicted")
plt.title(f"{ticker} hold-out period")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plot_df[["Date", "Trading_Code", "Close", "Target_Close", "Predicted_Close"]].head(15)'''),
]


def write_nb(name: str, cells: list[dict]) -> None:
    path = NB_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    write_nb("DSA_Data_Cleaning_.ipynb", CLEANING)
    write_nb("Single_Stock_Close_Price.ipynb", SINGLE)
    write_nb("MultiStock_Prediction.ipynb", MULTI)
