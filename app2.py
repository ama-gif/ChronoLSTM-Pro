import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from datetime import timedelta
import gc
import plotly.figure_factory as ff
import plotly.express as px
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera
import io

# Page Configuration with Custom Theme
st.set_page_config(
    layout="wide",
    page_title="LSTM Forecaster Pro | AI-Powered Time Series Analysis",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        background: linear-gradient(to right, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #e0e0e0;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .feature-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.85rem;
        color: #fff;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Navigation Radio Buttons */
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stRadio > div {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Container Styling */
    [data-testid="stContainer"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #b0b0b0;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Select Box & Input */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        color: white;
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        color: white;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess {
        background: rgba(0, 255, 127, 0.1);
        border-left: 4px solid #00ff7f;
        border-radius: 8px;
        padding: 1rem;
        color: #00ff7f;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        color: #667eea;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        color: #ffc107;
    }
    
    .stError {
        background: rgba(255, 82, 82, 0.1);
        border-left: 4px solid #ff5252;
        border-radius: 8px;
        padding: 1rem;
        color: #ff5252;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 1.5rem 0;
    }
    
    /* Stats Card */
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .stat-label {
        color: #b0b0b0;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Icon Styling */
    .icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    h2 {
        border-bottom: 2px solid rgba(102, 126, 234, 0.5);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.4);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMForecasting(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, linear_hidden_size, lstm_num_layers, linear_num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.linear_hidden_size = linear_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.linear_num_layers = linear_num_layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.linear_layers = nn.ModuleList()
        self.linear_num_layers-=1
        self.linear_layers.append(nn.Linear(self.lstm_hidden_size, self.linear_hidden_size))

        for _ in range(linear_num_layers): 
            self.linear_layers.append(nn.Linear(self.linear_hidden_size, int(self.linear_hidden_size/1.5)))
            self.linear_hidden_size = int(self.linear_hidden_size/1.5)
        
        self.fc = nn.Linear(self.linear_hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0)) 

        for linear_layer in self.linear_layers:
            out = linear_layer(out)
        
        out = self.fc(out[:, -1, :])
        return out

# Initialize session states
if 'sd_click' not in st.session_state:
    st.session_state.sd_click = False
if 'train_click' not in st.session_state:
    st.session_state.train_click = False
if 'disable_opt' not in st.session_state:
    st.session_state.disable_opt = False
if 'model_save' not in st.session_state:
    st.session_state.model_save = None
if 'saved_models' not in st.session_state:
    st.session_state.saved_models = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return torch.from_numpy(array(X)).float(), torch.from_numpy(array(y)).float()

def onClickSD():
    st.session_state.sd_click = True

def onClickTrain():
    st.session_state.train_click = True

def preProcessData(date_f, input_f, output_f, df):
    preProcessDataList = input_f.copy()
    preProcessDataList.insert(-1, output_f)
    preProcessDF = df[list(dict.fromkeys(preProcessDataList))].copy()
    preProcessDF = preProcessDF.astype(float)
    preProcessDF = preProcessDF.replace(0, np.nan)
    preProcessDF = preProcessDF.interpolate(method='linear')
    preProcessDF = preProcessDF.bfill()
    
    preProcessDF.insert(0, date_f, df[date_f])
    if str(preProcessDF.at[0, date_f]).isdigit():
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f], format='%Y')
    else:
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f])
    return preProcessDF

def check_date_frequency(date_series):
    dates = pd.to_datetime(date_series)
    differences = (dates - dates.shift(1)).dropna()
    
    daily_count = (differences == timedelta(days=1)).sum()
    hourly_count = (differences == timedelta(hours=1)).sum()
    weekly_count = (differences == timedelta(weeks=1)).sum()
    monthly_count = (differences >= timedelta(days=28, hours=23, minutes=59)).sum()
    
    if daily_count > max(monthly_count, hourly_count, weekly_count):
        return 365
    elif monthly_count > max(daily_count, hourly_count, weekly_count):
        return 12
    elif weekly_count > max(daily_count, hourly_count, monthly_count):
        return 52
    elif hourly_count > max(daily_count, weekly_count, monthly_count):
        return 24*365
    else:
        return 1

def calculate_advanced_metrics(actual, predicted):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # MAPE - Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Directional Accuracy
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # Symmetric MAPE
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R²': r2,
        'MAPE': mape,
        'SMAPE': smape,
        'Directional Accuracy': directional_accuracy
    }

def perform_residual_analysis(actual, predicted, dates):
    """Analyze residuals to understand model errors"""
    st.markdown("## 🔬 Residual Analysis")
    
    residuals = actual - predicted
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Mean Residual", f"{np.mean(residuals):.4f}")
    with col2:
        st.metric("📈 Std Residual", f"{np.std(residuals):.4f}")
    with col3:
        st.metric("📉 Min Residual", f"{np.min(residuals):.4f}")
    with col4:
        st.metric("📊 Max Residual", f"{np.max(residuals):.4f}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Residual Plots", "📊 Distribution", "🔍 Diagnostics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=residuals, mode='lines+markers', 
                                    name='Residuals', line=dict(color='#667eea')))
            fig.add_hline(y=0, line_dash="dash", line_color="#ff5252")
            fig.update_layout(
                title='Residuals Over Time',
                xaxis_title='Date',
                yaxis_title='Residual',
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers', 
                                    name='Predictions', marker=dict(color='#667eea', size=8)))
            fig.add_trace(go.Scatter(x=[actual.min(), actual.max()], 
                                    y=[actual.min(), actual.max()], 
                                    mode='lines', name='Perfect Fit', 
                                    line=dict(dash='dash', color='#ff5252')))
            fig.update_layout(
                title='Actual vs Predicted',
                xaxis_title='Actual',
                yaxis_title='Predicted',
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals',
                                      marker=dict(color='#667eea')))
            fig.update_layout(
                title='Residual Distribution',
                xaxis_title='Residual',
                yaxis_title='Frequency',
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            from scipy.stats import probplot
            qq = probplot(residuals, dist="norm")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', 
                                    name='Q-Q', marker=dict(color='#667eea')))
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0], 
                                    mode='lines', name='Theoretical', 
                                    line=dict(dash='dash', color='#ff5252')))
            fig.update_layout(
                title='Q-Q Plot',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        
        shapiro_stat, shapiro_p = shapiro(residuals)
        with col1:
            st.markdown("**Shapiro-Wilk Test**")
            st.metric("p-value", f"{shapiro_p:.4f}")
            st.write("✅ Normal" if shapiro_p > 0.05 else "❌ Not Normal")
        
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        with col2:
            st.markdown("**Ljung-Box Test**")
            st.metric("p-value", f"{lb_test['lb_pvalue'].values[0]:.4f}")
            st.write("✅ No Autocorrelation" if lb_test['lb_pvalue'].values[0] > 0.05 else "❌ Autocorrelation Present")
        
        with col3:
            st.markdown("**Heteroscedasticity**")
            correlation = np.corrcoef(actual, np.abs(residuals))[0, 1]
            st.metric("Correlation", f"{correlation:.4f}")
            st.write("✅ Homoscedastic" if abs(correlation) < 0.3 else "❌ Heteroscedastic")

def feature_importance_analysis(model, input_features, X_test, y_test):
    """Analyze feature importance"""
    st.markdown("## 🎯 Feature Importance Analysis")
    
    st.info("📊 This analysis shows which input features have the most impact on predictions.")
    
    baseline_loss = nn.MSELoss()(model(X_test.to(device)), y_test.to(device)).item()
    
    importances = []
    feature_names = input_features
    
    progress_bar = st.progress(0)
    for i, feature in enumerate(feature_names):
        X_permuted = X_test.clone()
        X_permuted[:, :, i] = X_permuted[:, torch.randperm(X_permuted.size(0)), i]
        
        with torch.no_grad():
            permuted_loss = nn.MSELoss()(model(X_permuted.to(device)), y_test.to(device)).item()
        
        importance = ((permuted_loss - baseline_loss) / baseline_loss) * 100
        importances.append(importance)
        progress_bar.progress((i + 1) / len(feature_names))
    
    progress_bar.empty()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance (%)': importances
    }).sort_values('Importance (%)', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance_df['Importance (%)'],
            y=importance_df['Feature'],
            orientation='h',
            marker=dict(color=importance_df['Importance (%)'], 
                       colorscale='Viridis')
        ))
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance (%)',
            yaxis_title='Feature',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(importance_df, use_container_width=True, height=400)
    
    st.success(f"🏆 Most important feature: **{importance_df.iloc[0]['Feature']}** ({importance_df.iloc[0]['Importance (%)']:.2f}%)")

def model_comparison_dashboard():
    """Compare multiple saved models"""
    st.markdown("## 📊 Model Comparison Dashboard")
    
    if len(st.session_state.saved_models) < 2:
        st.warning("⚠️ You need at least 2 saved models to perform comparison.")
        return
    
    model_names = [m['name'] for m in st.session_state.saved_models]
    selected_models = st.multiselect("Select models to compare:", model_names, default=model_names[:2])
    
    if len(selected_models) < 2:
        st.info("Please select at least 2 models.")
        return
    
    comparison_data = []
    for model_name in selected_models:
        model_info = next(m for m in st.session_state.saved_models if m['name'] == model_name)
        comparison_data.append(model_info['metrics'])
    
    comparison_df = pd.DataFrame(comparison_data, index=selected_models)
    
    st.subheader("Metrics Comparison")
    st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE', 'MSE', 'MAPE', 'SMAPE'], color='lightgreen')
                                   .highlight_max(axis=0, subset=['R²', 'Directional Accuracy'], color='lightgreen'),
                use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for metric in ['RMSE', 'MAE', 'MAPE']:
            fig.add_trace(go.Bar(name=metric, x=selected_models, y=comparison_df[metric]))
        fig.update_layout(
            title='Error Metrics Comparison (Lower is Better)',
            barmode='group',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='R²', x=selected_models, y=comparison_df['R²']))
        fig.add_trace(go.Bar(name='Directional Accuracy', x=selected_models, 
                            y=comparison_df['Directional Accuracy']))
        fig.update_layout(
            title='Performance Metrics (Higher is Better)',
            barmode='group',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    best_model = comparison_df['RMSE'].idxmin()
    st.success(f"🏆 **Recommended Model:** {best_model} (Lowest RMSE: {comparison_df.loc[best_model, 'RMSE']:.4f})")

def export_model_and_results(model, model_config, predictions, actual, metrics, model_name):
    """Export model and results"""
    st.markdown("### 💾 Export Model & Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_data = {
            'model_state_dict': model.state_dict(),
            'config': model_config,
            'metrics': metrics,
            'name': model_name
        }
        
        buffer = io.BytesIO()
        torch.save(model_data, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Model (.pth)",
            data=buffer,
            file_name=f"{model_name}.pth",
            mime="application/octet-stream"
        )
    
    with col2:
        results_df = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions,
            'Residual': actual - predictions
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (.csv)",
            data=csv,
            file_name=f"{model_name}_results.csv",
            mime="text/csv"
        )
    
    if st.button("💾 Save Model to Session"):
        st.session_state.saved_models.append({
            'name': model_name,
            'model': model,
            'config': model_config,
            'metrics': metrics,
            'predictions': predictions,
            'actual': actual
        })
        st.success(f"✅ Model '{model_name}' saved successfully!")

def perform_eda(df, date_f, output_f):
    st.markdown("## 📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Distributions", "📉 Time Series", "🔗 Relationships", "📊 Statistics", "🔍 Outliers"])
    
    with tab1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution Plots")
            for col in numeric_cols[:3]:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=30,
                                          marker=dict(color='#667eea')))
                fig.update_layout(
                    title=f'Distribution of {col}',
                    xaxis_title=col,
                    yaxis_title='Frequency',
                    height=300,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Box Plots")
            for col in numeric_cols[:3]:
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[col], name=col, marker=dict(color='#667eea')))
                fig.update_layout(
                    title=f'Box Plot of {col}',
                    yaxis_title=col,
                    height=300,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(255,255,255,0.05)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Time Series Analysis")
        if date_f and output_f:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_f], y=df[output_f], mode='lines', 
                                    name=output_f, line=dict(color='#667eea', width=2)))
            fig.update_layout(
                title=f'{output_f} Over Time',
                xaxis_title='Date',
                yaxis_title=output_f,
                height=400,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Rolling Statistics")
            window = st.slider("Rolling Window Size", 3, 30, 7)
            df_rolling = df.copy()
            df_rolling['Rolling_Mean'] = df[output_f].rolling(window=window).mean()
            df_rolling['Rolling_Std'] = df[output_f].rolling(window=window).std()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_f], y=df[output_f], mode='lines', 
                                    name='Original', line=dict(color='#667eea')))
            fig.add_trace(go.Scatter(x=df[date_f], y=df_rolling['Rolling_Mean'], 
                                    mode='lines', name='Rolling Mean', line=dict(color='#38ef7d')))
            fig.add_trace(go.Scatter(x=df[date_f], y=df_rolling['Rolling_Std'], 
                                    mode='lines', name='Rolling Std', line=dict(color='#ff5252')))
            fig.update_layout(
                title='Rolling Statistics',
                height=400,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Relationships")
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', 
                           color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.update_layout(
                title='Correlation Heatmap',
                height=500,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X axis", numeric_cols, key='scatter_x')
        with col2:
            y_col = st.selectbox("Select Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key='scatter_y')
        
        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols")
        fig.update_layout(
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Descriptive Statistics**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            st.write("**Missing Values**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        st.subheader("Normality Tests")
        selected_col = st.selectbox("Select column for normality test", numeric_cols)
        
        if selected_col:
            col1, col2, col3 = st.columns(3)
            
            stat, p = shapiro(df[selected_col].dropna())
            with col1:
                st.metric("Shapiro-Wilk p-value", f"{p:.4f}")
                st.write("✅ Normal" if p > 0.05 else "❌ Not Normal")
            
            stat, p = normaltest(df[selected_col].dropna())
            with col2:
                st.metric("D'Agostino p-value", f"{p:.4f}")
                st.write("✅ Normal" if p > 0.05 else "❌ Not Normal")
            
            stat, p = jarque_bera(df[selected_col].dropna())
            with col3:
                st.metric("Jarque-Bera p-value", f"{p:.4f}")
                st.write("✅ Normal" if p > 0.05 else "❌ Not Normal")
    
    with tab5:
        st.subheader("Outlier Detection")
        
        selected_col = st.selectbox("Select column for outlier detection", numeric_cols, key='outlier_col')
        
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Total Outliers", len(outliers))
            st.metric("📈 Percentage", f"{(len(outliers)/len(df)*100):.2f}%")
        
        with col2:
            st.metric("📉 Lower Bound", f"{lower_bound:.2f}")
            st.metric("📊 Upper Bound", f"{upper_bound:.2f}")
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[selected_col], name='With Outliers',
                            marker=dict(color='#667eea')))
        fig.update_layout(
            title=f'Outliers in {selected_col}',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        st.plotly_chart(fig, use_container_width=True)

def sea_decomp(date_f, input_f, output_f, df):
    if date_f:
        sea_decomp_data = preProcessData(date_f, input_f, output_f, df)
        corr_df = sea_decomp_data.select_dtypes(include=['int', 'float'])
        correlation_matrix = np.round(corr_df.corr(), 1)
        result = seasonal_decompose(sea_decomp_data.set_index(date_f)[output_f], model='additive', 
                                     period=check_date_frequency(sea_decomp_data[date_f]))
        
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=result.seasonal.index.values, y=result.seasonal.values, 
                                   mode='lines', line=dict(color='#38ef7d', width=2)))
        fig_s.update_layout(
            title='Seasonal Component',
            xaxis_title='Date',
            yaxis_title='Value',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=result.trend.index.values, y=result.trend.values, 
                                   mode='lines', line=dict(color='#667eea', width=2)))
        fig_t.update_layout(
            title='Trend Component',
            xaxis_title='Date',
            yaxis_title='Value',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.05)'
        )
        
        fig_corr = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale='Viridis')
        fig_corr.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        with st.container(border=True):
            st.subheader("🔗 Correlation Matrix")
            st.divider()
            st.plotly_chart(fig_corr, use_container_width=True)

        with st.container(border=True):
            st.subheader("📉 Seasonal Decomposition")
            st.divider()
            st.plotly_chart(fig_t, use_container_width=True)
            st.divider()
            st.plotly_chart(fig_s, use_container_width=True)

        with st.container(border=True):
            st.subheader("📋 Pre-Processed Data Preview")
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="📊 Total Rows", value=sea_decomp_data.shape[0])
            with col2:
                st.metric(label="📈 Total Columns", value=sea_decomp_data.shape[1])
            st.dataframe(sea_decomp_data, use_container_width=True, height=250)
        
        return sea_decomp_data

# Main Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🚀 LSTM Forecaster Pro</h1>
    <p class="main-subtitle">AI-Powered Multi-Step Time Series Forecasting Platform</p>
    <div style="margin-top: 1rem;">
        <span class="feature-badge">📊 Advanced EDA</span>
        <span class="feature-badge">🔬 Residual Analysis</span>
        <span class="feature-badge">🎯 Feature Importance</span>
        <span class="feature-badge">📈 Model Comparison</span>
        <span class="feature-badge">💾 Model Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Select Module:",
        ["🏠 Home & Training", "📊 Exploratory Data Analysis", "🔬 Model Analysis", "📈 Model Comparison"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # System Info
    st.markdown("### ⚙️ System Info")
    device_emoji = "🖥️" if device.type == 'cpu' else "🎮"
    st.info(f"{device_emoji} **Device:** {device.type.upper()}")
    
    if len(st.session_state.saved_models) > 0:
        st.success(f"💾 **Saved Models:** {len(st.session_state.saved_models)}")
    
    st.divider()
    
    # About
    st.markdown("### ℹ️ About")
    st.markdown("""
    **LSTM Forecaster Pro** is a comprehensive platform for time series forecasting using deep learning.
    
    Built with:
    - 🔥 PyTorch
    - 📊 Plotly
    - 🎨 Streamlit
    """)

col1, col2 = st.columns([3, 2])

with col1:
    with st.container(border=True):
        st.subheader("📁 Upload Your Dataset")
        st.divider()
        uploaded_file = st.file_uploader("Choose a CSV file:", type=['csv'], help="Upload your time series data in CSV format")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep='[,;]', engine='python')
    
    if page == "📊 Exploratory Data Analysis":
        with col1:
            with st.container(border=True):
                st.subheader("🎯 Select Features for EDA")
                st.divider()
                date_col_eda = st.selectbox("📅 Date Column (optional):", ["None"] + list(df.columns))
                output_col_eda = st.selectbox("🎯 Target Column (optional):", ["None"] + list(df.columns))
        
        date_f_eda = None if date_col_eda == "None" else date_col_eda
        output_f_eda = None if output_col_eda == "None" else output_col_eda
        perform_eda(df, date_f_eda, output_f_eda)
    
    elif page == "🔬 Model Analysis":
        st.info("💡 Train a model first to access residual analysis and feature importance.")
        
        if st.session_state.model_save is not None:
            model_data = st.session_state.model_save
            
            perform_residual_analysis(
                model_data['actual'],
                model_data['predictions'],
                model_data['dates']
            )
            
            if 'model' in model_data and 'X_test' in model_data:
                feature_importance_analysis(
                    model_data['model'],
                    model_data['input_features'],
                    model_data['X_test'],
                    model_data['y_test']
                )
    
    elif page == "📈 Model Comparison":
        model_comparison_dashboard()
    
    else:  # Home & Training
        with col1:
            with st.container(border=True):
                st.subheader("🎯 Feature Selection")
                st.divider()
                date_f = st.selectbox(label="📅 Date Feature:", options=df.columns, help="Select the column containing date/time information")
                input_f = st.multiselect(label="📊 Input Features (X):", 
                                        options=[element for element in list(df.columns) if element != date_f],
                                        help="Select one or more features to use as model inputs")
                output_f = st.selectbox(label="🎯 Output Feature (Y):", 
                                       options=[element for element in list(df.columns) if element != date_f],
                                       help="Select the target variable you want to predict")
                st.divider()
                st.button('🔄 Pre-Process Data', type="primary", on_click=onClickSD, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.subheader("📋 Dataset Overview")
                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(label="📊 Total Rows", value=df.shape[0])
                with col_b:
                    st.metric(label="📈 Total Columns", value=df.shape[1])
                st.dataframe(df, use_container_width=True, height=250)

        if st.session_state.sd_click:
            data = sea_decomp(date_f, input_f, output_f, df)
            
            with st.container(border=True):
                st.subheader("🤖 LSTM Model Configuration")
                st.divider()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Sequence Configuration**")
                    lag_steps = st.number_input('🔙 Lag Steps:', step=1, min_value=1, value=10,
                                               help="Number of past time steps to use for prediction")
                    forecast_steps = st.number_input('🔮 Forecast Steps:', step=1, min_value=1, value=5,
                                                    help="Number of future time steps to predict")
                
                with col2:
                    st.markdown("**🧠 LSTM Architecture**")
                    lstm_layers = st.slider('📚 LSTM Layers:', 1, 5, 2,
                                           help="Number of LSTM layers in the network")
                    lstm_neurons = st.slider('🔢 LSTM Neurons:', 10, 500, 50,
                                            help="Number of neurons in each LSTM layer")
                
                if (lag_steps + forecast_steps) > (df.shape[0] - forecast_steps):
                    st.error('⚠️ Invalid configuration: Lag + Forecast steps exceed available data', icon="🚨")
                else:
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔗 Dense Layers**")
                        linear_hidden_layers = st.slider('📚 Hidden Layers:', 1, 5, 2,
                                                        help="Number of dense layers after LSTM")
                        linear_hidden_neurons = st.slider('🔢 Hidden Neurons:', lstm_neurons, 500, 100,
                                                         help="Number of neurons in hidden layers")
                    
                    with col2:
                        st.markdown("**⚙️ Training Parameters**")
                        n_epochs = st.number_input('🔄 Epochs:', step=1, min_value=1, value=100,
                                                  help="Number of training iterations")
                        batch_size = st.number_input('📦 Batch Size:', step=1, min_value=1, max_value=100, value=32,
                                                    help="Number of samples per training batch")
                    
                    st.divider()
                    model_name = st.text_input("📝 Model Name:", value=f"LSTM_Model_{len(st.session_state.saved_models)+1}",
                                              help="Give your model a unique name")
                    
                    if st.button('🚀 Train Model', type="primary", use_container_width=True):
                        with st.spinner('🔄 Training in progress...'):
                            df_train = data[:-forecast_steps]
                            df_test = data[-forecast_steps:]
                            
                            scaler = MinMaxScaler(feature_range=(0, 1))
                            datastack = np.empty((df_train.shape[0], 0))
                            
                            for i in range(1, len(data.columns[1:]) + 1):
                                datastack = np.hstack((datastack, scaler.fit_transform(df_train.iloc[:, i].values.reshape((-1, 1)))))
                            datastack = np.hstack((datastack, scaler.fit_transform(df_train[output_f].values.reshape((-1, 1)))))
                            
                            X, y = split_sequences(datastack, lag_steps, forecast_steps)
                            
                            train_size = int(0.8 * len(X))
                            X_train, X_test = X[:train_size], X[train_size:]
                            y_train, y_test = y[:train_size], y[train_size:]
                            
                            X_train, y_train = X_train.to(device), y_train.to(device)
                            X_test, y_test = X_test.to(device), y_test.to(device)
                            
                            dataset = torch.utils.data.TensorDataset(X_train, y_train)
                            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                            
                            model = LSTMForecasting(input_size=X.shape[2], lstm_hidden_size=lstm_neurons, 
                                                   lstm_num_layers=lstm_layers, linear_num_layers=linear_hidden_layers,
                                                   linear_hidden_size=linear_hidden_neurons, output_size=forecast_steps).to(device)
                            
                            criterion = nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                            
                            training_losses = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for epoch in range(1, n_epochs + 1):
                                epoch_loss = 0
                                for inputs, labels in dataloader:
                                    optimizer.zero_grad()
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item()
                                
                                avg_loss = epoch_loss / len(dataloader)
                                training_losses.append(avg_loss)
                                progress_bar.progress(int((epoch / n_epochs) * 100))
                                status_text.text(f"⏳ Epoch {epoch}/{n_epochs} - Loss: {avg_loss:.6f}")
                            
                            progress_bar.empty()
                            status_text.empty()
                        
                        st.success('🎉 Model Training Completed Successfully!', icon="✅")
                        
                        model.eval()
                        with torch.no_grad():
                            test_outputs = model(X_test)
                            test_predictions = scaler.inverse_transform(test_outputs.cpu().numpy().reshape(-1, 1))
                            test_actual = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
                        
                        metrics = calculate_advanced_metrics(test_actual.flatten(), test_predictions.flatten())
                        
                        st.markdown("## 📊 Model Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📉 RMSE", f"{metrics['RMSE']:.4f}")
                            st.metric("📊 MAE", f"{metrics['MAE']:.4f}")
                        with col2:
                            st.metric("📈 MSE", f"{metrics['MSE']:.4f}")
                            st.metric("🎯 R² Score", f"{metrics['R²']:.4f}")
                        with col3:
                            st.metric("📊 MAPE", f"{metrics['MAPE']:.2f}%")
                            st.metric("📉 SMAPE", f"{metrics['SMAPE']:.2f}%")
                        with col4:
                            st.metric("🎯 Directional Accuracy", f"{metrics['Directional Accuracy']:.2f}%")
                        
                        # Training loss plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=training_losses, mode='lines', name='Training Loss',
                                                line=dict(color='#667eea', width=2)))
                        fig.update_layout(
                            title='📉 Training Loss Progression',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            height=350,
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(255,255,255,0.05)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Final predictions
                        with torch.no_grad():
                            final_outputs = model(torch.Tensor(datastack[-lag_steps:, :-1]).float().unsqueeze(0).to(device))
                            final_predictions = scaler.inverse_transform(final_outputs[0].reshape(1, -1).cpu().numpy())
                        
                        # Forecast visualization
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_test[date_f], y=df_test[output_f].values, 
                                                mode='lines+markers', name='Actual', 
                                                line=dict(color='#667eea', width=2),
                                                marker=dict(size=8)))
                        fig.add_trace(go.Scatter(x=df_test[date_f], y=final_predictions[0], 
                                                mode='lines+markers', name='Predicted', 
                                                line=dict(color='#38ef7d', dash='dash', width=2),
                                                marker=dict(size=8)))
                        fig.update_layout(
                            title='🔮 Forecast vs Actual Values',
                            xaxis_title='Date',
                            yaxis_title=output_f,
                            height=450,
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(255,255,255,0.05)',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save model configuration
                        model_config = {
                            'lag_steps': lag_steps,
                            'forecast_steps': forecast_steps,
                            'lstm_layers': lstm_layers,
                            'lstm_neurons': lstm_neurons,
                            'linear_hidden_layers': linear_hidden_layers,
                            'linear_hidden_neurons': linear_hidden_neurons,
                            'n_epochs': n_epochs,
                            'batch_size': batch_size
                        }
                        
                        st.session_state.model_save = {
                            'model': model,
                            'predictions': final_predictions[0],
                            'actual': df_test[output_f].values,
                            'dates': df_test[date_f],
                            'metrics': metrics,
                            'config': model_config,
                            'X_test': X_test,
                            'y_test': y_test,
                            'input_features': input_f,
                            'scaler': scaler
                        }
                        
                        export_model_and_results(model, model_config, final_predictions[0], 
                                                df_test[output_f].values, metrics, model_name)
                        
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2 style='color: #667eea;'>👋 Welcome to LSTM Forecaster Pro!</h2>
        <p style='font-size: 1.2rem; color: #b0b0b0; margin-top: 1rem;'>
            Upload your time series dataset to get started with advanced AI-powered forecasting.
        </p>
        <p style='color: #888; margin-top: 2rem;'>
            📁 Supported format: CSV
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="icon">📊</div>
            <h3 style="color: #667eea; margin: 0;">EDA</h3>
            <p style="color: #b0b0b0; font-size: 0.9rem;">Comprehensive exploratory data analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="icon">🤖</div>
            <h3 style="color: #667eea; margin: 0;">LSTM</h3>
            <p style="color: #b0b0b0; font-size: 0.9rem;">Deep learning forecasting models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="icon">🔬</div>
            <h3 style="color: #667eea; margin: 0;">Analysis</h3>
            <p style="color: #b0b0b0; font-size: 0.9rem;">Residual & feature importance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="icon">📈</div>
            <h3 style="color: #667eea; margin: 0;">Compare</h3>
            <p style="color: #b0b0b0; font-size: 0.9rem;">Multi-model comparison tools</p>
        </div>
        """, unsafe_allow_html=True)
