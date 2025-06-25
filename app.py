import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import openai
from datetime import datetime, timedelta
import requests

# Load API keys from Streamlit Secrets
openai.api_key = st.secrets["openai"]["api_key"]
FMP_API_KEY = st.secrets["fmp"]["api_key"]

st.title("Comprehensive R-G Model Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload Excel file with 'Tickers' and 'Hard' sheets", type="xlsx")

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    df_tickers = xls.parse("Tickers")
    df_raw = xls.parse("Hard", header=None)

    # Data Preparation
    series_list = []
    tickers = df_raw.iloc[0, 1::2].tolist()

    for i, ticker in enumerate(tickers):
        dates = pd.to_datetime(df_raw.iloc[1:, i*2], errors='coerce')
        values = pd.to_numeric(df_raw.iloc[1:, i*2+1], errors='coerce')

        df_series = pd.DataFrame({'Date': dates, ticker: values}).dropna(subset=['Date', ticker]).set_index('Date').resample('ME').ffill()
        series_list.append(df_series)

    df_monthly = pd.concat(series_list, axis=1).sort_index().dropna(how='all')

    # Define R and G variables explicitly
    R_target = 'USOSFR10 Curncy'
    G_target = 'GDP CQoQ Index'

    R_vars = ['FEDL01 Index', 'PCEPILFE Index', 'CONSP5MD Index', 'ACMTP10  Index', 'USGGBE10 Index']
    G_vars = ['CONSSENT Index', 'SBOICAPS Index', 'OUMFCEF  Index', 'NAPMNEWO Index', 'IP       Index', 'PAYEMS_PCH', 'VIX Index', 'LEI BP Index', 'PCE DEFM Index']

    df_norm = df_monthly.copy()  # Simplified normalization for brevity

    short_names_map = df_tickers.set_index('Ticker')['Short_Name'].to_dict()

    # Weight calculations
    def calculate_weights(df, target, vars):
        dep_var = df[target].shift(-1)
        weights, correlations, betas = {}, {}, {}
        for var in vars:
            X = df[var].dropna()
            Y = dep_var.loc[X.index].dropna()
            aligned_idx = X.index.intersection(Y.index)
            coef = np.cov(X.loc[aligned_idx], Y.loc[aligned_idx])[0, 1] / np.var(X.loc[aligned_idx])
            corr = np.corrcoef(X.loc[aligned_idx], Y.loc[aligned_idx])[0, 1]
            weights[var] = coef
            correlations[var] = corr
            betas[var] = coef
        total = sum(abs(w) for w in weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}
        return correlations, betas, normalized_weights

    correlations_R, betas_R, weights_R = calculate_weights(df_norm, R_target, R_vars)
    correlations_G, betas_G, weights_G = calculate_weights(df_norm, G_target, G_vars)

    # Display comprehensive tables
    st.subheader("R Variables Analysis")
    df_R = pd.DataFrame({
        'Short_Name': pd.Series(R_vars).map(short_names_map),
        'Correlation with R_target': pd.Series(correlations_R),
        'Coefficient (Beta) with R_target': pd.Series(betas_R),
        'Weight': pd.Series(weights_R)
    }).sort_values(by='Correlation with R_target', ascending=False)
    st.dataframe(df_R)

    st.subheader("G Variables Analysis")
    df_G = pd.DataFrame({
        'Short_Name': pd.Series(G_vars).map(short_names_map),
        'Correlation with G_target': pd.Series(correlations_G),
        'Coefficient (Beta) with G_target': pd.Series(betas_G),
        'Weight': pd.Series(weights_G)
    }).sort_values(by='Correlation with G_target', ascending=False)
    st.dataframe(df_G)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[R_target], mode='lines', name='R Score'))
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[G_target], mode='lines', name='G Score'))
    fig.update_layout(title='R and G Scores Over Time', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    # GenAI Comprehensive Queries
    st.subheader("Comprehensive Economic Analysis")

    detailed_prompt = f"""Provide an extensive economic analysis based on the following R and G scores:

    R Score: {df_norm[R_target].iloc[-1]:.4f}
    G Score: {df_norm[G_target].iloc[-1]:.4f}

    Include implications, historical context, methodological critique, and actionable recommendations."""

    if st.button("Generate Comprehensive Analysis"):
        with st.spinner("Generating comprehensive economic analysis..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert macroeconomist providing detailed, structured, and educational economic analyses."},
                    {"role": "user", "content": detailed_prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            st.markdown(response.choices[0].message.content)
