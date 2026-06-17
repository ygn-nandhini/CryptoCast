import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="CryptoCast", page_icon="₿"
, layout="wide")


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 60, output_size)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x

# ===== DATA LOAD =====
@st.cache_data
def load_data():
    df = pd.read_csv("data/bitcoin.csv")
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    def conv_vol(v):
        v = str(v).strip()
        if 'K' in v: return float(v.replace('K','')) * 1e3
        if 'M' in v: return float(v.replace('M','')) * 1e6
        if 'B' in v: return float(v.replace('B','')) * 1e9
        return float(v)
    df['Vol.']     = df['Vol.'].apply(conv_vol)
    df['Change %'] = df['Change %'].astype(str).str.replace('%','').astype(float)
    df['Date']     = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['Vol.']).reset_index(drop=True)
    return df

def create_sequences(data, window=60, horizon=1):
    X, y = [], []
    for i in range(window, len(data) - horizon + 1):
        X.append(data[i-window:i, 0])
        y.append(data[i:i+horizon, 0])
    return np.array(X), np.array(y)

@st.cache_data
def get_test_data():
    df     = load_data()
    prices = df['Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    split  = int(len(scaled) * 0.8)
    train  = scaled[:split]
    test   = scaled[split:]
    overlap = np.concatenate([train[-60:], test], axis=0)
    return overlap, scaler, df

def get_tensors(overlap, horizon):
    X_te, y_te = create_sequences(overlap, 60, horizon)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(-1)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)
    return X_te_t, y_te_t

# ===== TRAIN FUNCTION =====
@st.cache_data
def train_and_evaluate(model_name, horizon):
    overlap, scaler, df = get_test_data()

    # Train data
    prices = df['Price'].values.reshape(-1, 1)
    scaled = scaler.transform(prices)
    split  = int(len(scaled) * 0.8)
    train  = scaled[:split]

    X_tr, y_tr = create_sequences(train, 60, horizon)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    X_te_t, y_te_t = get_tensors(overlap, horizon)

    # Model create
    if model_name == "LSTM":
        model = LSTMModel(output_size=horizon)
    elif model_name == "RNN":
        model = RNNModel(output_size=horizon)
    elif model_name == "CNN":
        model = CNNModel(output_size=horizon)
    else:
        model = TransformerModel(output_size=horizon)

    # Train
    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

    # Predict
    model.eval()
    with torch.no_grad():
        preds  = model(X_te_t).numpy()
        actual = y_te_t.numpy()

    # ✅ Reshape to (n,1) for inverse_transform
    pred_inv   = scaler.inverse_transform(preds[:, 0].reshape(-1, 1))
    actual_inv = scaler.inverse_transform(actual[:, 0].reshape(-1, 1))

    mae  = mean_absolute_error(actual_inv, pred_inv)
    rmse = np.sqrt(mean_squared_error(actual_inv, pred_inv))
    mape = np.mean(np.abs((actual_inv - pred_inv) / actual_inv)) * 100

    return pred_inv, actual_inv, mae, rmse, mape

# ===== FUTURE FORECAST (last 60 days → next N days) =====
def future_forecast(model_name, horizon, scaler, df):
    prices = df['Price'].values.reshape(-1, 1)
    scaled = scaler.transform(prices)
    last60 = scaled[-60:].reshape(1, 60, 1)
    inp    = torch.tensor(last60, dtype=torch.float32)

    if model_name == "LSTM":
        model = LSTMModel(output_size=horizon)
    elif model_name == "RNN":
        model = RNNModel(output_size=horizon)
    elif model_name == "CNN":
        model = CNNModel(output_size=horizon)
    else:
        model = TransformerModel(output_size=horizon)

    # Quick train
    prices_full = df['Price'].values.reshape(-1, 1)
    scaled_full = scaler.transform(prices_full)
    split = int(len(scaled_full) * 0.8)
    train = scaled_full[:split]
    X_tr, y_tr = create_sequences(train, 60, horizon)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(50):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(inp).numpy()
    return scaler.inverse_transform(out.reshape(-1, 1)).flatten()

# ===== UI =====
st.title("₿ CryptoCast — Bitcoin Price Forecasting")
st.markdown("Multi-Horizon Deep Learning Forecasting System")
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Settings")
model_choice   = st.sidebar.selectbox("🤖 Model",   ["LSTM", "RNN", "CNN", "Transformer"])
horizon_choice = st.sidebar.selectbox("📅 Horizon", ["1 Day", "3 Days", "7 Days"])
horizon_map    = {"1 Day": 1, "3 Days": 3, "7 Days": 7}
horizon        = horizon_map[horizon_choice]
predict_btn    = st.sidebar.button("🚀 Run Forecast")

# Price History
df = load_data()
st.header("📊 Bitcoin Price History (2010–2024)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Year Range",    "2010 – 2024")
c3.metric("All-Time High", f"${df['Price'].max():,.0f}")
c4.metric("All-Time Low",  f"${df['Price'].min():.2f}")

fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(df['Date'], df['Price'], color='orange', linewidth=0.8)
ax1.set_xlabel('Date'); ax1.set_ylabel('Price (USD)')
ax1.set_title('Bitcoin Price History (2010–2024)')
st.pyplot(fig1); plt.close(fig1)
st.markdown("---")

# Model Comparison Table
st.header("🏆 Model Comparison (MAPE %)")
comp = pd.DataFrame({
    'Model':    ['CNN',   'RNN',  'LSTM', 'Transformer'],
    '1D MAPE%': [9.20,   4.77,   5.35,   8.62],
    '3D MAPE%': [19.46,  9.12,   8.48,   7.48],
    '7D MAPE%': [15.92,  7.92,   14.76,  13.15]
})
st.dataframe(comp, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
x, w = np.arange(4), 0.25
ax2.bar(x-w, comp['1D MAPE%'], w, label='1 Day',  color='steelblue')
ax2.bar(x,   comp['3D MAPE%'], w, label='3 Days', color='orange')
ax2.bar(x+w, comp['7D MAPE%'], w, label='7 Days', color='green')
ax2.set_xticks(x); ax2.set_xticklabels(comp['Model'])
ax2.set_title('Model MAPE % Comparison')
ax2.set_ylabel('MAPE %'); ax2.legend()
st.pyplot(fig2); plt.close(fig2)
st.markdown("---")

# ===== FORECAST SECTION =====
st.header(f"🔮 Forecast: {model_choice} | {horizon_choice}")

if predict_btn:
    with st.spinner(f"⏳ Training {model_choice} for {horizon_choice}... (50 epochs)"):
        pred, actual, mae, rmse, mape = train_and_evaluate(model_choice, horizon)

    # Metrics
    st.subheader("📈 Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"${mae:,.2f}")
    m2.metric("RMSE", f"${rmse:,.2f}")
    m3.metric("MAPE", f"{mape:.2f}%")
    st.markdown("---")

    # Future price prediction
    st.subheader(f"💰 Next {horizon_choice} Predicted Price")
    overlap, scaler_obj, _ = get_test_data()
    future_prices = future_forecast(model_choice, horizon, scaler_obj, df)

    if horizon == 1:
        st.success(f"📅 Next Day Predicted Price: **${future_prices[0]:,.2f}**")
    else:
        cols = st.columns(horizon)
        for i, price in enumerate(future_prices):
            cols[i].metric(f"Day {i+1}", f"${price:,.2f}")
    st.markdown("---")

    # Actual vs Predicted chart
    st.subheader(f"📉 Actual vs Predicted — {model_choice} ({horizon_choice})")
    n = min(100, len(actual))
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(actual[:n, 0], label='Actual',    color='blue',   linewidth=1.5)
    ax3.plot(pred[:n, 0],   label='Predicted', color='orange', linewidth=1.5, linestyle='--')
    ax3.set_title(f'{model_choice} — {horizon_choice} (First 100 Test Days)')
    ax3.set_xlabel('Days'); ax3.set_ylabel('Price (USD)'); ax3.legend()
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    st.pyplot(fig3); plt.close(fig3)
    st.markdown("---")

    # 1D vs 3D vs 7D side-by-side
    st.subheader(f"📊 Horizon Comparison — {model_choice}: 1D vs 3D vs 7D")
    fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = [('blue','orange'), ('navy','tomato'), ('darkblue','gold')]

    for idx, (h, title) in enumerate(zip([1,3,7], ['1 Day','3 Days','7 Days'])):
        with st.spinner(f"Loading {title}..."):
            p, a, _, _, mape_h = train_and_evaluate(model_choice, h)
        n  = min(100, len(a))
        ax = axes[idx]
        ax.plot(a[:n,0], label='Actual',    color=colors[idx][0], linewidth=1.5)
        ax.plot(p[:n,0], label='Predicted', color=colors[idx][1], linewidth=1.5, linestyle='--')
        ax.set_title(f'{model_choice} — {title}\nMAPE: {mape_h:.2f}%')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price (USD)' if idx == 0 else '')
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}k'))

    plt.tight_layout()
    st.pyplot(fig4); plt.close(fig4)

else:
    st.info("👈 Sidebar-ல Model & Horizon select  **🚀 Run Forecast** click ")

st.markdown("---")
st.markdown("🎓 CryptoCast | Bitcoin Price Forecasting | Deep Learning Project")