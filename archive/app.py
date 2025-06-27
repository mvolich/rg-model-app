import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("R-G Model Analysis")

uploaded_file = st.file_uploader("Upload Excel file with 'Tickers' and 'Hard' sheets", type="xlsx")

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    # Explicitly parse Excel sheets
    df_tickers = xls.parse("Tickers")
    df_raw = xls.parse("Hard", header=None)

    tickers = df_raw.iloc[0, 1::2].tolist()

    series_list = []

    for i, ticker in enumerate(tickers):
        dates = pd.to_datetime(df_raw.iloc[1:, i * 2], errors='coerce')
        values = pd.to_numeric(df_raw.iloc[1:, i * 2 + 1], errors='coerce')

        df_series = pd.DataFrame({'Date': dates, ticker: values}).dropna(subset=['Date', ticker]).set_index('Date').resample('ME').ffill()
        series_list.append(df_series)

    df_monthly = pd.concat(series_list, axis=1).sort_index().dropna(how='all')

    R_target = 'USOSFR10 Curncy'
    G_target = 'GDP CQoQ Index'

    R_vars = ['FEDL01 Index', 'PCEPILFE Index', 'CONSP5MD Index', 'ACMTP10  Index', 'USGGBE10 Index']
    G_vars = ['CONSSENT Index', 'SBOICAPS Index', 'OUMFCEF  Index', 'NAPMNEWO Index', 'IP       Index', 'PAYEMS_PCH', 'VIX Index', 'LEI BP Index', 'PCE DEFM Index']

    short_names_map = df_tickers.set_index('Ticker')['Short_Name'].to_dict()

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

    df_norm = df_monthly.rolling(120, min_periods=1).apply(lambda x: (x.iloc[-1] - np.mean(x)) / np.std(x) if np.std(x) else 0)

    correlations_R, betas_R, weights_R = calculate_weights(df_norm, R_target, R_vars)
    correlations_G, betas_G, weights_G = calculate_weights(df_norm, G_target, G_vars)

    df_norm['R_score'] = df_norm[R_vars].mul(pd.Series(weights_R)).sum(axis=1)
    df_norm['G_score'] = df_norm[G_vars].mul(pd.Series(weights_G)).sum(axis=1)

    st.subheader("R Variables Analysis")
    df_R = pd.DataFrame({
        'Short_Name': [short_names_map.get(var, var) for var in R_vars],
        'Correlation with R_target': [correlations_R[var] for var in R_vars],
        'Coefficient (Beta) with R_target': [betas_R[var] for var in R_vars],
        'Weight': [weights_R[var] for var in R_vars]
    }).sort_values(by='Correlation with R_target', ascending=False)
    st.dataframe(df_R)

    st.subheader("G Variables Analysis")
    df_G = pd.DataFrame({
        'Short_Name': [short_names_map.get(var, var) for var in G_vars],
        'Correlation with G_target': [correlations_G[var] for var in G_vars],
        'Coefficient (Beta) with G_target': [betas_G[var] for var in G_vars],
        'Weight': [weights_G[var] for var in G_vars]
    }).sort_values(by='Correlation with G_target', ascending=False)
    st.dataframe(df_G)

    # Corrected Plotly Plot
    fig = go.Figure()

    color_tightening = '#E15759'
    color_loosening = '#59A14F'

    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['R_score'], mode='lines',
                             line=dict(color=color_tightening), name='Net R'))

    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['G_score'], mode='lines',
                             line=dict(color=color_loosening), name='Net G'))

    window_ma = 6
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['R_score'].rolling(window=window_ma).mean(),
                             mode='lines', line=dict(color=color_tightening, dash='dot'), name='Net R MA (6-month)'))

    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['G_score'].rolling(window=window_ma).mean(),
                             mode='lines', line=dict(color=color_loosening, dash='dot'), name='Net G MA (6-month)'))

    fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    fig.update_layout(title='Financial Conditions: Net R and G Scores', plot_bgcolor='white',
                      hovermode='x unified', legend_title_text='Legend')

    st.plotly_chart(fig, use_container_width=True)

