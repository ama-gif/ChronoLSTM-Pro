# ChronoLSTM-Pro
# 🚀 LSTM Forecaster Pro  
### AI-Powered Multi-Step Time Series Forecasting Platform

LSTM Forecaster Pro is an advanced **end-to-end time series forecasting application** built using **PyTorch, Streamlit, and Plotly**.  
It enables users to perform **EDA, train deep learning models, analyze residuals, compare models, and export results** — all through an interactive UI.

---

## 🔥 Key Features

- 📊 **Advanced Exploratory Data Analysis (EDA)**
  - Distribution plots, box plots, correlation heatmaps
  - Time-series visualization with rolling statistics
  - Normality & outlier detection tests

- 🤖 **Deep Learning Forecasting (LSTM)**
  - Multi-step forecasting using custom LSTM architecture
  - Configurable lag steps, forecast horizon, layers & neurons
  - GPU acceleration support (CUDA)

- 📈 **Model Evaluation & Diagnostics**
  - RMSE, MAE, MSE, R², MAPE, SMAPE
  - Directional accuracy measurement
  - Residual analysis with statistical tests

- 🎯 **Feature Importance Analysis**
  - Permutation-based feature importance
  - Visual ranking of influential features

- 📊 **Model Comparison Dashboard**
  - Compare multiple trained models
  - Automatically recommends best-performing model

- 💾 **Export & Reusability**
  - Download trained PyTorch models (.pth)
  - Export predictions and residuals as CSV
  - Save multiple models in-session

---

## 🛠 Tech Stack

- **Frontend & UI:** Streamlit, Custom CSS  
- **Visualization:** Plotly, Plotly Express  
- **Deep Learning:** PyTorch  
- **Data Processing:** Pandas, NumPy, Scikit-learn  
- **Statistics:** SciPy, Statsmodels  
- **Deployment Ready:** CPU / GPU compatible  

---

## 📁 Input Data Format

- CSV file
- Must contain:
  - A **date/time column**
  - One or more **numeric input features**
  - One **target variable**

---

## ⚙️ How It Works

1. Upload a CSV dataset  
2. Perform EDA & statistical analysis  
3. Configure LSTM architecture & training parameters  
4. Train the model and visualize forecasts  
5. Analyze residuals & feature importance  
6. Compare models and export results  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
