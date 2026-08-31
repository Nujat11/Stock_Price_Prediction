# 📈 Stock Price Prediction

A machine learning and deep learning project for **next-day stock price prediction** using historical data from the **Dhaka Stock Exchange (DSE)**.

The project supports both **single-stock** and **multi-stock** prediction. **BEXIMCO** is used as the primary single-stock case study.

---

## 📌 Project Overview

This project analyzes historical DSE stock data and predicts the **next trading day's closing price** using:

* Technical indicators
* Statistical analysis
* Machine learning
* Gradient boosting
* Deep learning
* Ensemble learning

The project is divided into two main prediction tasks:

### Single-Stock Prediction

Focuses on **BEXIMCO** and applies statistical analysis, machine learning, BiLSTM, Transformer, and ensemble models.

### Multi-Stock Prediction

Predicts next-day closing prices across multiple DSE-listed stocks using tree-based and boosting models with a shared calendar-based time split.

---

## 📊 Dataset

The dataset contains historical daily trading information from the Dhaka Stock Exchange.

| Feature        | Description       |
| -------------- | ----------------- |
| `Trading_Code` | Stock ticker/code |
| `Date`         | Trading date      |
| `Open`         | Opening price     |
| `High`         | Highest price     |
| `Low`          | Lowest price      |
| `Close`        | Closing price     |
| `Volume`       | Trading volume    |

**Primary Stock:** BEXIMCO

---

## ⚙️ Features

The project uses historical market data along with engineered features:

* Price features
* Daily and log returns
* Lag features
* Moving averages
* Volatility
* RSI
* MACD
* Bollinger Bands
* ATR
* Volume indicators
* Calendar features

---

## 🤖 Models

### Machine Learning

* Ridge Regression
* Random Forest
* HistGradientBoosting
* XGBoost
* LightGBM
* CatBoost

### Deep Learning

* Bidirectional LSTM (BiLSTM)
* Transformer Encoder

### Ensemble

* Inverse-RMSE Weighted Ensemble

---

## 🔬 Statistical Analysis

For the single-stock analysis, the project includes:

* Augmented Dickey-Fuller (ADF) Test
* Phillips-Perron Test
* CUSUM Structural Break Analysis

---

## ⏱️ Time-Series Validation

The project uses **chronological train/validation/test splitting** instead of random splitting.

For multi-stock prediction, a **shared calendar-based split** is used across stocks to prevent temporal and ticker-order leakage.

---

## 📏 Evaluation Metrics

Model performance is evaluated using:

* **R² Score**
* **RMSE**
* **MAE**
* **MAPE**
* **Directional Accuracy**

---

## 📂 Project Structure

```text
Stock_Price_Prediction/
│
├── 📁 catboost_info/
│   └── CatBoost training information
│
├── 📁 dataset/
│   ├── Raw DSE datasets
│   └── Processed datasets
│
├── 📁 models/
│   └── Trained model files
│
├── 📁 results/
│   └── Prediction and evaluation results
│
├── 📁 src/
│   ├── features.py
│   ├── split.py
│   ├── metrics.py
│   └── models.py
│
├── 📄 DSA_Data_Cleaning_.ipynb
├── 📄 Single_Stock_Close_Price.ipynb
├── 📄 MultiStock_Prediction.ipynb
│
├── 📄 app.py
├── 📄 main.py
├── 📄 predict.py
├── 📄 retrain.py
├── 📄 train_final.py
├── 📄 update_data.py
├── 📄 visualize.py
├── 📄 _build_notebooks.py
├── 📄 run_update.bat
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 📓 Notebooks

### `DSA_Data_Cleaning_.ipynb`

Data cleaning and exploratory analysis of DSE stock data.

### `Single_Stock_Close_Price.ipynb`

Complete **BEXIMCO single-stock forecasting** pipeline including statistical analysis, machine learning, BiLSTM, Transformer, and ensemble prediction.

### `MultiStock_Prediction.ipynb`

**Multi-stock forecasting** pipeline using multiple machine learning and boosting models.

---

## 🛠️ Tech Stack

**Language**

* Python

**Data Processing**

* Pandas
* NumPy

**Machine Learning**

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

**Deep Learning**

* TensorFlow
* Keras

**Statistical Analysis**

* Statsmodels

**Visualization**

* Matplotlib
* Plotly

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Nujat11/Stock_Price_Prediction.git
cd Stock_Price_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Update Data

```bash
python update_data.py
```

### Train Model

```bash
python train_final.py
```

### Retrain Model

```bash
python retrain.py
```

### Generate Predictions

```bash
python predict.py
```

### Visualize Results

```bash
python visualize.py
```

### Run Application

```bash
python app.py
```

---

## 🔐 Data Leakage Prevention

The project takes several measures to prevent future information from entering the training process:

* Chronological train/validation/test splitting
* Training-only scaling
* Proper target shifting
* Calendar-based splitting for multi-stock prediction
* Time-aware feature generation

---

## 🔮 Future Work

* Real-time DSE data integration
* News and sentiment analysis
* Macroeconomic features
* Walk-forward validation
* Hyperparameter optimization
* Advanced Transformer models
* Explainable AI
* Real-time prediction dashboard

---
