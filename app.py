import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Interactive R-G Model Analysis")

# === Step 1: File Upload ===
uploaded_file = st.file_uploader("Upload Excel file with 'Tickers' and 'Hard' sheets", type="xlsx")

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    df_tickers = xls.parse("Tickers")
    df_raw = xls.parse("Hard", header=None)

    # === Data Preparation ===
    series_list = []
    tickers = df_raw.iloc[0, 1::2].tolist()

    for i, ticker in enumerate(tickers):
        dates = pd.to_datetime(df_raw.iloc[1:, i*2], errors='coerce')
        values = pd.to_numeric(df_raw.iloc[1:, i*2+1], errors='coerce')

        df_series = pd.DataFrame({
            'Date': dates,
            ticker: values
        }).dropna(subset=['Date', ticker]).set_index('Date').resample('ME').ffill()

        series_list.append(df_series)

    df_monthly = pd.concat(series_list, axis=1).sort_index()
    df_monthly.dropna(how='all', inplace=True)

    st.subheader("Monthly Data Preview")
    st.dataframe(df_monthly.head())

    # === Define R and G variables ===
    R_target = 'USOSFR10 Curncy'
    G_target = 'GDP CQoQ Index'

    R_vars = ['FEDL01 Index', 'PCEPILFE Index', 'CONSP5MD Index', 'ACMTP10  Index', 'USGGBE10 Index']
    G_vars = ['CONSSENT Index', 'SBOICAPS Index', 'OUMFCEF  Index', 'NAPMNEWO Index', 'IP       Index', 'PAYEMS_PCH', 'VIX Index', 'LEI BP Index', 'PCE DEFM Index']

    R_vars = [var for var in R_vars if var in df_monthly.columns]
    G_vars = [var for var in G_vars if var in df_monthly.columns]

    # === Normalization Function ===
    def normalize_data(df, tickers_info):
        pct_change_vars = tickers_info[tickers_info['Units'] == '% CHANGE']['Ticker'].tolist()
        index_vars = tickers_info[tickers_info['Units'] == 'INDEX VALUE']['Ticker'].tolist()
        rate_vars = tickers_info[tickers_info['Units'] == 'INTEREST RATE']['Ticker'].tolist()

        df_norm = pd.DataFrame(index=df.index)
        vars_to_normalize = index_vars + rate_vars
        rolling_mean = df[vars_to_normalize].rolling(window=120, min_periods=24).mean()
        rolling_std = df[vars_to_normalize].rolling(window=120, min_periods=24).std()
        df_norm[vars_to_normalize] = (df[vars_to_normalize] - rolling_mean) / rolling_std
        df_norm[pct_change_vars] = df[pct_change_vars]

        if G_target in df.columns:
            df_norm[G_target] = df[G_target].interpolate(method='linear')

        return df_norm.dropna()

    df_norm = normalize_data(df_monthly, df_tickers)

    # === Weight Calculation Function ===
    def calculate_weights(df_norm, target_var, explanatory_vars):
        dependent_var = df_norm[target_var].shift(-1)
        weights = {}
        for var in explanatory_vars:
            X, Y_aligned = df_norm[var].dropna(), dependent_var.loc[df_norm[var].dropna().index]
            valid_idx = Y_aligned.dropna().index.intersection(X.index)
            coef = np.cov(X.loc[valid_idx], Y_aligned.loc[valid_idx])[0, 1] / np.var(X.loc[valid_idx])
            weights[var] = coef
        weights_sum = sum(abs(w) for w in weights.values())
        return {k: v / weights_sum for k, v in weights.items()}

    weights_R = calculate_weights(df_norm, R_target, R_vars)
    weights_G = calculate_weights(df_norm, G_target, G_vars)

    # === Visualization ===
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[R_target], mode='lines', name='R Score'))
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[G_target], mode='lines', name='G Score'))

    fig.update_layout(title='R and G Scores Over Time', plot_bgcolor='white')

    st.plotly_chart(fig, use_container_width=True)

    # === Display Weights ===
    st.subheader("R Variable Weights")
    st.dataframe(pd.DataFrame.from_dict(weights_R, orient='index', columns=['Weight']))

    st.subheader("G Variable Weights")
    st.dataframe(pd.DataFrame.from_dict(weights_G, orient='index', columns=['Weight']))
