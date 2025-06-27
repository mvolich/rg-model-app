import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="R-G Model Streamlit App", layout="wide")
st.title("R-G Financial Conditions Model")

# --- Sidebar for file upload ---
st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

if not uploaded_file:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# --- Load Data ---
xls = pd.ExcelFile(uploaded_file)
df_tickers = xls.parse("Tickers")
df_raw = xls.parse("Hard", header=None)

# --- Prepare Data ---
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

# --- Dynamic fix for USOSFR10 Curncy ---
usosfr10_col_index = df_raw.iloc[0, :][df_raw.iloc[0, :] == 'USOSFR10 Curncy'].index[0]
date_col = pd.to_datetime(df_raw.iloc[1:, usosfr10_col_index - 1], errors='coerce', dayfirst=True)
value_col = pd.to_numeric(df_raw.iloc[1:, usosfr10_col_index], errors='coerce')
usosfr10_df = pd.DataFrame({
    'Date': date_col,
    'USOSFR10 Curncy': value_col
}).dropna(subset=['Date']).set_index('Date').resample('ME').ffill()
df_monthly['USOSFR10 Curncy'] = usosfr10_df['USOSFR10 Curncy']

# --- Identify R and G Variables ---
R_target = 'USOSFR10 Curncy'
G_target = 'GDP CQoQ Index'
R_vars = ['FEDL01 Index', 'PCEPILFE Index', 'CONSP5MD Index', 'ACMTP10  Index', 'USGGBE10 Index']
G_vars = ['CONSSENT Index', 'SBOICAPS Index', 'OUMFCEF  Index', 'NAPMNEWO Index', 'IP       Index', 'PAYEMS_PCH', 'VIX Index', 'LEI BP Index', 'PCE DEFM Index']
R_vars = [var for var in R_vars if var in df_monthly.columns]
G_vars = [var for var in G_vars if var in df_monthly.columns]

# --- Monthly GDP Imputation ---
if G_target in df_monthly.columns:
    df_monthly['GDP_CQoQ_Monthly'] = df_monthly[G_target].interpolate(method='linear')
    df_monthly.drop(columns=[G_target], inplace=True)
else:
    st.error(f"{G_target} column not found in DataFrame. Please verify data input.")
    st.stop()

# --- Differentiated Normalization ---
pct_change_vars = df_tickers[df_tickers['Units'] == '% CHANGE']['Ticker'].tolist()
index_vars = df_tickers[df_tickers['Units'] == 'INDEX VALUE']['Ticker'].tolist()
rate_vars = df_tickers[df_tickers['Units'] == 'INTEREST RATE']['Ticker'].tolist()
pct_change_vars = [var for var in pct_change_vars if var != 'GDP CQoQ Index']
df_norm = pd.DataFrame(index=df_monthly.index)
vars_to_normalize = index_vars + rate_vars
rolling_mean = df_monthly[vars_to_normalize].rolling(window=120, min_periods=24).mean()
rolling_std = df_monthly[vars_to_normalize].rolling(window=120, min_periods=24).std()
df_norm[vars_to_normalize] = (df_monthly[vars_to_normalize] - rolling_mean) / rolling_std
df_norm[pct_change_vars] = df_monthly[pct_change_vars]
if 'GDP_CQoQ_Monthly' in df_monthly:
    df_norm['GDP_CQoQ_Monthly'] = df_monthly['GDP_CQoQ_Monthly']
df_norm.dropna(inplace=True)

# --- Weights Calculation ---
dependent_var_R = df_norm[R_target].shift(-1)
weights_R = {}
for var in R_vars:
    X, Y_aligned = df_norm[var].dropna(), dependent_var_R.loc[df_norm[var].dropna().index]
    valid_idx = Y_aligned.dropna().index.intersection(X.index)
    coef = np.cov(X.loc[valid_idx], Y_aligned.loc[valid_idx])[0, 1] / np.var(X.loc[valid_idx])
    weights_R[var] = coef
weights_sum_R = sum(abs(w) for w in weights_R.values())
weights_R = {k: v / weights_sum_R for k, v in weights_R.items()}
dependent_var_G = df_norm['GDP_CQoQ_Monthly'].shift(-1)
weights_G = {}
for var in G_vars:
    X, Y_aligned = df_norm[var].dropna(), dependent_var_G.loc[df_norm[var].dropna().index]
    valid_idx = Y_aligned.dropna().index.intersection(X.index)
    coef = np.cov(X.loc[valid_idx], Y_aligned.loc[valid_idx])[0, 1] / np.var(X.loc[valid_idx])
    weights_G[var] = coef
weights_sum_G = sum(abs(w) for w in weights_G.values())
weights_G = {k: v / weights_sum_G for k, v in weights_G.items()}
df_norm['R_score'] = df_norm[R_vars].mul(pd.Series(weights_R)).sum(axis=1)
df_norm['G_score'] = df_norm[G_vars].mul(pd.Series(weights_G)).sum(axis=1)
df_weights_R = pd.DataFrame.from_dict(weights_R, orient='index', columns=['Weight'])
df_weights_R['Short_Name'] = df_weights_R.index.map(short_names_map)
df_weights_R_sorted = df_weights_R[['Short_Name', 'Weight']].sort_values('Weight', ascending=False)
df_weights_G = pd.DataFrame.from_dict(weights_G, orient='index', columns=['Weight'])
df_weights_G['Short_Name'] = df_weights_G.index.map(short_names_map)
df_weights_G_sorted = df_weights_G[['Short_Name', 'Weight']].sort_values('Weight', ascending=False)

# --- Table of R_vars Correlation, Coefficient (Beta), and Weights ---
R_correlations = df_norm[R_vars].corrwith(df_norm[R_target])
R_coefs = {}
dependent_var_R_lagged = df_norm[R_target].shift(-1).dropna()
for var in R_vars:
    X_aligned = df_norm[var].loc[dependent_var_R_lagged.index].dropna()
    Y_aligned = dependent_var_R_lagged.loc[X_aligned.index]
    if len(X_aligned) > 1 and np.var(X_aligned) != 0:
        coef = np.cov(X_aligned, Y_aligned)[0, 1] / np.var(X_aligned)
        R_coefs[var] = coef
    else:
        R_coefs[var] = np.nan
R_table_data = {
    'Correlation with R_target': R_correlations,
    'Coefficient (Beta) with R_target': pd.Series(R_coefs),
    'Weight': pd.Series(weights_R)
}
df_R_table_full = pd.DataFrame(R_table_data)
df_R_table_full['Short_Name'] = df_R_table_full.index.map(short_names_map)
df_R_table_full = df_R_table_full[['Short_Name', 'Correlation with R_target', 'Coefficient (Beta) with R_target', 'Weight']].sort_values('Correlation with R_target', ascending=False)
with st.expander("Table of R_vars Correlation, Coefficient (Beta) with R_target, and their Weights"):
    st.dataframe(df_R_table_full)

# --- Table of G_vars Correlation, Coefficient (Beta), and Weights ---
G_correlations = df_norm[G_vars].corrwith(df_norm['GDP_CQoQ_Monthly'])
G_coefs = {}
dependent_var_G_lagged = df_norm['GDP_CQoQ_Monthly'].shift(-1).dropna()
for var in G_vars:
    X_aligned = df_norm[var].loc[dependent_var_G_lagged.index].dropna()
    Y_aligned = dependent_var_G_lagged.loc[X_aligned.index]
    if len(X_aligned) > 1 and np.var(X_aligned) != 0:
        coef = np.cov(X_aligned, Y_aligned)[0, 1] / np.var(X_aligned)
        G_coefs[var] = coef
    else:
        G_coefs[var] = np.nan
G_table_data_full = {
    'Correlation with G_target': G_correlations,
    'Coefficient (Beta) with G_target': pd.Series(G_coefs),
    'Weight': pd.Series(weights_G)
}
df_G_table_full = pd.DataFrame(G_table_data_full)
df_G_table_full['Short_Name'] = df_G_table_full.index.map(short_names_map)
df_G_table_full = df_G_table_full[['Short_Name', 'Correlation with G_target', 'Coefficient (Beta) with G_target', 'Weight']].sort_values('Correlation with G_target', ascending=False)
with st.expander("Table of G_vars Correlation, Coefficient (Beta) with G_target, and their Weights"):
    st.dataframe(df_G_table_full)

# --- R Variables Plot ---
with st.expander("Normalized R Variables and R_target Over Time"):
    fig_R = go.Figure()
    for var in R_vars:
        if var in df_norm.columns:
            fig_R.add_trace(go.Scatter(x=df_norm.index, y=df_norm[var], mode='lines', name=short_names_map.get(var, var)))
    if R_target in df_norm.columns:
        fig_R.add_trace(go.Scatter(x=df_norm.index, y=df_norm[R_target], mode='lines', name=short_names_map.get(R_target, R_target), line=dict(color='red', dash='dash')))
    fig_R.update_layout(title='Normalized R Variables and R_target Over Time', xaxis_title='Date', yaxis_title='Normalized Value', hovermode='x unified')
    st.plotly_chart(fig_R, use_container_width=True)

# --- G Variables Plot ---
with st.expander("Normalized G Variables and Imputed GDP Over Time"):
    fig_G = go.Figure()
    for var in G_vars:
        if var in df_norm.columns:
            fig_G.add_trace(go.Scatter(x=df_norm.index, y=df_norm[var], mode='lines', name=short_names_map.get(var, var)))
    if 'GDP_CQoQ_Monthly' in df_norm.columns:
        fig_G.add_trace(go.Scatter(x=df_norm.index, y=df_norm['GDP_CQoQ_Monthly'], mode='lines', name=short_names_map.get(G_target, G_target) + ' (Imputed)', line=dict(color='red', dash='dash')))
    fig_G.update_layout(title='Normalized G Variables and Imputed GDP Over Time', xaxis_title='Date', yaxis_title='Normalized Value', hovermode='x unified')
    st.plotly_chart(fig_G, use_container_width=True)

# --- Financial Conditions Plot ---
with st.expander("Financial Conditions: Net R and G Scores"):
    fig = go.Figure()
    color_tightening = '#E15759'
    color_loosening = '#59A14F'
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['R_score'], mode='lines', line=dict(color=color_tightening), name='Net R'))
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['G_score'], mode='lines', line=dict(color=color_loosening), name='Net G'))
    window_ma = 6
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['R_score'].rolling(window=window_ma).mean(), mode='lines', line=dict(color=color_tightening, dash='dot'), name='Net R MA (6-month)'))
    fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm['G_score'].rolling(window=window_ma).mean(), mode='lines', line=dict(color=color_loosening, dash='dot'), name='Net G MA (6-month)'))
    current_state = df_norm['R_score'].iloc[0] > df_norm['G_score'].iloc[0]
    segment_x, segment_upper, segment_lower = [], [], []
    for date, r_score, g_score in zip(df_norm.index, df_norm['R_score'], df_norm['G_score']):
        tightening = r_score > g_score
        upper, lower = max(r_score, g_score), min(r_score, g_score)
        if tightening != current_state and segment_x:
            color = 'rgba(225,87,89,0.3)' if current_state else 'rgba(89,161,79,0.3)'
            fig.add_trace(go.Scatter(x=segment_x + segment_x[::-1], y=segment_upper + segment_lower[::-1], fill='toself', fillcolor=color, mode='none', showlegend=False))
            segment_x, segment_upper, segment_lower = [], [], []
            current_state = tightening
        segment_x.append(date)
        segment_upper.append(upper)
        segment_lower.append(lower)
    color = 'rgba(225,87,89,0.3)' if current_state else 'rgba(89,161,79,0.3)'
    fig.add_trace(go.Scatter(x=segment_x + segment_x[::-1], y=segment_upper + segment_lower[::-1], fill='toself', fillcolor=color, mode='none', showlegend=False))
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    fig.update_layout(title='Financial Conditions: Net R and G Scores', plot_bgcolor='white', hovermode='x unified', legend_title_text='Legend')
    st.plotly_chart(fig, use_container_width=True)

# --- Attribution and Visualization Sections ---

# Monetary Conditions Attribution Plot
with st.expander("Monetary Conditions Attribution (Weighted and Normalized)"):
    R_vars_no_target = [var for var in R_vars if var != 'USOSFR10 Curncy']
    R_weighted = df_norm[R_vars_no_target].copy()
    for var in R_vars_no_target:
        R_weighted[var] *= weights_R[var]
    ticker_to_name = df_tickers.set_index('Ticker')['Long_Name'].to_dict()
    R_weighted.rename(columns=ticker_to_name, inplace=True)
    R_weighted['Net_R_Score'] = R_weighted.sum(axis=1)
    fig = go.Figure()
    positive_color = '#E15759'
    negative_color = '#59A14F'
    for col in R_weighted.columns[:-1]:
        fig.add_trace(go.Bar(
            x=R_weighted.index,
            y=R_weighted[col],
            marker_color=[positive_color if val >= 0 else negative_color for val in R_weighted[col]],
            opacity=0.7,
            hoverinfo='skip',
            showlegend=False
        ))
    hover_text = []
    for idx, row in R_weighted.iterrows():
        sorted_row = row[:-1].sort_values(ascending=False)
        hover_label = f"<b>{idx.strftime('%Y-%m')}</b><br>"
        for col, val in sorted_row.items():
            text_color = positive_color if val >= 0 else negative_color
            hover_label += f"<span style='color:{text_color}'>{col}: {val:.2f}</span><br>"
        hover_text.append(hover_label)
    fig.add_trace(go.Scatter(
        x=R_weighted.index,
        y=[0]*len(R_weighted.index),
        mode='markers',
        opacity=0,
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=R_weighted.index,
        y=R_weighted['Net_R_Score'],
        mode='lines',
        line=dict(color='grey', width=2),
        name='Net Monetary Conditions',
        hovertemplate='Net Score: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title='Monetary Conditions Attribution (Weighted and Normalized)',
        xaxis_title='Date',
        yaxis_title='Weighted Normalized Contribution',
        plot_bgcolor='white',
        barmode='relative',
        hovermode='x unified',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Growth Conditions Attribution Plot
with st.expander("Growth Conditions Attribution (Weighted and Normalized)"):
    G_vars_no_target = [var for var in G_vars if var != 'GDP CQoQ Index']
    G_weighted = df_norm[G_vars_no_target].copy()
    for var in G_vars_no_target:
        G_weighted[var] *= weights_G[var]
    ticker_to_name = df_tickers.set_index('Ticker')['Long_Name'].to_dict()
    G_weighted.rename(columns=ticker_to_name, inplace=True)
    G_weighted['Net_Growth_Score'] = G_weighted.sum(axis=1)
    fig = go.Figure()
    positive_color = '#59A14F'
    negative_color = '#E15759'
    for col in G_weighted.columns[:-1]:
        fig.add_trace(go.Bar(
            x=G_weighted.index,
            y=G_weighted[col],
            marker_color=[positive_color if val >= 0 else negative_color for val in G_weighted[col]],
            opacity=0.7,
            hoverinfo='skip',
            showlegend=False
        ))
    hover_text = []
    for idx, row in G_weighted.iterrows():
        sorted_row = row[:-1].sort_values(ascending=False)
        hover_label = f"<b>{idx.strftime('%Y-%m')}</b><br>"
        for col, val in sorted_row.items():
            text_color = positive_color if val >= 0 else negative_color
            hover_label += f"<span style='color:{text_color}'>{col}: {val:.2f}</span><br>"
        hover_text.append(hover_label)
    fig.add_trace(go.Scatter(
        x=G_weighted.index,
        y=[0]*len(G_weighted.index),
        mode='markers',
        opacity=0,
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=G_weighted.index,
        y=G_weighted['Net_Growth_Score'],
        mode='lines',
        line=dict(color='grey', width=2),
        name='Net Growth Conditions',
        hovertemplate='Net Score: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title='Growth Conditions Attribution (Weighted and Normalized)',
        xaxis_title='Date',
        yaxis_title='Weighted Normalized Contribution',
        plot_bgcolor='white',
        barmode='relative',
        hovermode='x unified',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Latest Period Attribution Bar Charts ---
with st.expander("Latest Period Attribution Analysis"):
    # Latest Monetary Conditions Attribution
    latest_period_R = R_weighted.iloc[-1][:-1].sort_values()
    short_names_R = [df_tickers.set_index('Long_Name').loc[name, 'Short_Name'] if name in df_tickers.set_index('Long_Name').index else name for name in latest_period_R.index]
    colors_R = ['#E15759' if val >= 0 else '#59A14F' for val in latest_period_R]
    fig_R_latest = go.Figure(go.Bar(
        x=latest_period_R.values,
        y=short_names_R,
        orientation='h',
        marker=dict(color=colors_R, opacity=0.7),
        text=[f"{val:.2f}" for val in latest_period_R.values],
        textposition='auto',
        textfont=dict(size=11)
    ))
    fig_R_latest.update_layout(
        title=f'Monetary Conditions Attribution ({R_weighted.index[-1].strftime("%Y-%m")})',
        xaxis_title='Contribution',
        yaxis_title='Factors',
        plot_bgcolor='white',
        margin=dict(l=150, r=50, t=60, b=60),
        yaxis=dict(automargin=True, tickfont=dict(size=16), showgrid=True, gridcolor='lightgray', gridwidth=0.5),
        xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    )
    st.plotly_chart(fig_R_latest, use_container_width=True)
    
    # Latest Growth Conditions Attribution
    latest_period_G = G_weighted.iloc[-1][:-1].sort_values()
    short_names_G = [df_tickers.set_index('Long_Name').loc[name, 'Short_Name'] if name in df_tickers.set_index('Long_Name').index else name for name in latest_period_G.index]
    colors_G = ['#59A14F' if val >= 0 else '#E15759' for val in latest_period_G]
    fig_G_latest = go.Figure(go.Bar(
        x=latest_period_G.values,
        y=short_names_G,
        orientation='h',
        marker=dict(color=colors_G, opacity=0.7),
        text=[f"{val:.2f}" for val in latest_period_G.values],
        textposition='auto'
    ))
    fig_G_latest.update_layout(
        title=f'Growth Conditions Attribution ({G_weighted.index[-1].strftime("%Y-%m")})',
        xaxis_title='Contribution',
        yaxis_title='Factors',
        plot_bgcolor='white',
        margin=dict(l=150, r=50, t=60, b=60),
        yaxis=dict(automargin=True, tickfont=dict(size=16), showgrid=True, gridcolor='lightgray', gridwidth=0.5),
        xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    )
    st.plotly_chart(fig_G_latest, use_container_width=True)

# --- Comprehensive Dashboard ---
with st.expander("Comprehensive Dashboard (All Charts)"):
    max_range = max(abs(latest_period_R).max(), abs(latest_period_G).max()) * 1.1
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monetary Conditions Attribution (Last 12 Months)",
            f"Monetary Conditions Attribution ({R_weighted.index[-1].strftime('%Y-%m')})",
            "Growth Conditions Attribution (Last 12 Months)",
            f"Growth Conditions Attribution ({G_weighted.index[-1].strftime('%Y-%m')})"
        ),
        horizontal_spacing=0.20,
        vertical_spacing=0.2,
        column_widths=[0.55, 0.45]
    )
    
    # Column Charts (Left side)
    for col in R_weighted.columns[:-1]:
        fig.add_trace(go.Bar(
            x=R_weighted.tail(12).index.strftime('%Y-%m'),
            y=R_weighted.tail(12)[col],
            marker_color='#E15759',
            opacity=0.7,
            hoverinfo='skip',
            showlegend=False
        ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=R_weighted.tail(12).index.strftime('%Y-%m'),
        y=R_weighted.tail(12)['Net_R_Score'],
        mode='lines',
        line=dict(color='grey'),
        showlegend=False
    ), row=1, col=1)
    
    for col in G_weighted.columns[:-1]:
        fig.add_trace(go.Bar(
            x=G_weighted.tail(12).index.strftime('%Y-%m'),
            y=G_weighted.tail(12)[col],
            marker_color=['#59A14F' if val >= 0 else '#E15759' for val in G_weighted.tail(12)[col]],
            opacity=0.7,
            hoverinfo='skip',
            showlegend=False
        ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=G_weighted.tail(12).index.strftime('%Y-%m'),
        y=G_weighted.tail(12)['Net_Growth_Score'],
        mode='lines',
        line=dict(color='grey'),
        showlegend=False
    ), row=2, col=1)
    
    # Bar Charts (Right side)
    fig.add_trace(go.Bar(
        x=latest_period_R.values,
        y=short_names_R,
        orientation='h',
        marker=dict(color=colors_R, opacity=0.7),
        text=[f"{val:.2f}" for val in latest_period_R],
        textposition='auto',
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=latest_period_G.values,
        y=short_names_G,
        orientation='h',
        marker=dict(color=colors_G, opacity=0.7),
        text=[f"{val:.2f}" for val in latest_period_G],
        textposition='auto',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        height=1000,
        barmode='relative',
        plot_bgcolor='white',
        margin=dict(l=100, r=50, t=100, b=100),
        xaxis2=dict(range=[-max_range, max_range], showgrid=True, gridcolor='lightgray'),
        xaxis4=dict(range=[-max_range, max_range], showgrid=True, gridcolor='lightgray'),
        yaxis2=dict(automargin=True, tickfont=dict(size=12)),
        yaxis4=dict(automargin=True, tickfont=dict(size=12))
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Weighted Normalized Contribution", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Weighted Normalized Contribution", row=2, col=1)
    fig.update_xaxes(title_text="Contribution", row=1, col=2)
    fig.update_xaxes(title_text="Contribution", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

# --- Model Summary ---
with st.expander("Model Summary and Latest Metrics"):
    latest_metrics = df_norm.iloc[-1].to_dict()
    r_score = latest_metrics.get('R_score', 'Not available')
    g_score = latest_metrics.get('G_score', 'Not available')
    r_g_score_diff = r_score - g_score if isinstance(r_score, (int, float)) else 'Not available'
    
    st.markdown("## 📈 Latest Model Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monetary Conditions (R)", f"{r_score:.4f}")
    with col2:
        st.metric("Growth Conditions (G)", f"{g_score:.4f}")
    with col3:
        st.metric("Difference (R - G)", f"{r_g_score_diff:.4f}")
    
    st.markdown("### Model Overview")
    st.markdown("""
    The R-G financial conditions model provides insights into monetary and growth economic conditions:
    
    **Monetary Conditions (R):**
    - Reflect monetary tightening using indicators such as interest rates, liquidity, volatility, and credit spreads
    - Weighted using regression analysis with the US 10-Year Swap Rate as the target indicator
    
    **Growth Conditions (G):**
    - Reflect economic activity, including employment metrics, consumer confidence, and production indices
    - Weighted using regression analysis with monthly interpolated GDP growth as the target indicator
    
    **Data Processing:**
    - Monthly resampling, forward-filling, and Z-score normalization over a 120-month rolling window
    - A positive R-G difference indicates tighter monetary conditions relative to economic growth
    """)
    
    st.markdown("### Historical Context (Last 12 Months)")
    st.dataframe(df_norm[['R_score', 'G_score']].tail(12)) 