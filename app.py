import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai

st.set_page_config(page_title="R-G Model Streamlit App", layout="wide")
st.title("R-G Financial Conditions Model")

# --- Sidebar for file upload and OpenAI key ---
st.sidebar.header("Configuration")

# OpenAI API Key (from secrets or sidebar input)
def get_openai_api_key():
    try:
        return st.secrets["openai"]["api_key"]
    except Exception:
        return st.sidebar.text_input("OpenAI API Key", type="password")

openai_api_key = get_openai_api_key()

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

# --- Model Summary ---
with st.expander("Model Summary and Latest Metrics"):
    latest_metrics = df_norm.iloc[-1].to_dict()
    r_score = latest_metrics.get('R_score', 'Not available')
    g_score = latest_metrics.get('G_score', 'Not available')
    r_g_score_diff = r_score - g_score if isinstance(r_score, (int, float)) else 'Not available'
    
    st.markdown("## 📈 Latest Model Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monetary Conditions (R)", f"{r_score:.4f}" if isinstance(r_score, (int, float)) else r_score)
    with col2:
        st.metric("Growth Conditions (G)", f"{g_score:.4f}" if isinstance(g_score, (int, float)) else g_score)
    with col3:
        st.metric("Difference (R - G)", f"{r_g_score_diff:.4f}" if isinstance(r_g_score_diff, (int, float)) else r_g_score_diff)
    
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
df_R_table_full['Short_Name'] = df_R_table_full.index.map(df_tickers.set_index('Ticker')['Short_Name'].to_dict())
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
df_G_table_full['Short_Name'] = df_G_table_full.index.map(df_tickers.set_index('Ticker')['Short_Name'].to_dict())
df_G_table_full = df_G_table_full[['Short_Name', 'Correlation with G_target', 'Coefficient (Beta) with G_target', 'Weight']].sort_values('Correlation with G_target', ascending=False)
with st.expander("Table of G_vars Correlation, Coefficient (Beta) with G_target, and their Weights"):
    st.dataframe(df_G_table_full)

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

# --- GenAI Expert Review Section (OpenAI GPT-4) ---
with st.expander("AI Expert Review and Model Analysis (OpenAI GPT-4)"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar or Streamlit secrets to enable expert review.")
    else:
        openai.api_key = openai_api_key
        # Methodology Summary
        methodology_summary = """
Model Overview:

The R-G financial conditions model provides insights into monetary and growth economic conditions:

Monetary Conditions (R):
Reflect monetary tightening using indicators such as interest rates, liquidity, volatility, and credit spreads.
Weighted using regression analysis with the US 10-Year Swap Rate as the target indicator.

Growth Conditions (G):
Reflect economic activity, including employment metrics, consumer confidence, and production indices.
Weighted using regression analysis with monthly interpolated GDP growth as the target indicator.

Data undergoes monthly resampling, forward-filling, and Z-score normalization over a 120-month rolling window.
"""
        # Latest Metrics
        latest_metrics = df_norm.iloc[-1].to_dict()
        r_score = latest_metrics.get('R_score', 'Not available')
        g_score = latest_metrics.get('G_score', 'Not available')
        r_g_score_diff = r_score - g_score if isinstance(r_score, (int, float)) else 'Not available'
        metrics_summary = f"""
📈 Latest Model Metrics:

| Metric                    | Score   |
|---------------------------|---------|
| Monetary Conditions (R)   | {r_score:.4f} |
| Growth Conditions (G)     | {g_score:.4f} |
| Difference (R - G)        | {r_g_score_diff:.4f} |

A positive R-G difference indicates tighter monetary conditions relative to economic growth.
"""
        # Historical Attribution
        historical_scores = df_norm[['R_score', 'G_score']].tail(12).to_markdown()
        historical_summary = f"""
📊 Historical Context (Last 12 Months):

{historical_scores}
"""
        # Regression Weights
        weights_R_table = df_weights_R_sorted.to_markdown()
        weights_G_table = df_weights_G_sorted.to_markdown()
        regression_summary = f"""
🔍 Regression Analysis Findings:

Monetary Conditions (R) - Variable Weights:
{weights_R_table}

Growth Conditions (G) - Variable Weights:
{weights_G_table}

These regression results clearly highlight key indicators influencing monetary and growth conditions.
"""
        # Indicator Short Names
        short_names_dict = df_tickers.set_index('Ticker')['Short_Name'].to_dict()
        indicator_details = '\n'.join([f"- {ticker}: {name}" for ticker, name in short_names_dict.items()])
        indicator_summary = f"""
📚 Indicator Guide:
{indicator_details}
"""
        # Economic Significance & Interpretation Query
        prompt_economic_interpretation = f"""
You are a PhD-level macroeconomist providing a highly detailed, structured briefing on the R-G Model, explicitly contrasting Monetary (R) and Growth (G) conditions.

{methodology_summary}

{metrics_summary}

{indicator_summary}

Structured Briefing Outline:

1. Economic Significance & Interpretation (Quantitative Detail):
   - Summarise the Model explicitly.
   - Clearly explain the economic significance of R, G, and the R-G difference explicitly.
   - Provide intuitive explanations supported explicitly by recent quantitative data (latest metrics, historical values, regression weights).
   - Illustrate explicitly the implications of positive vs. negative R-G differences using current numerical examples.
"""
        try:
            response_economic_interpretation = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert macroeconomist providing quantitatively detailed, structured, and highly educational economic analyses."},
                    {"role": "user", "content": prompt_economic_interpretation}
                ],
                temperature=0.2,
                max_tokens=3500
            )
            expert_review_economic_interpretation = response_economic_interpretation["choices"][0]["message"]["content"]
            st.markdown("### 1. Economic Significance & Interpretation")
            st.markdown(expert_review_economic_interpretation)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
        # Calculation Methodology & Indicator Selection Query
        prompt_calculation_methodology = f"""
You are a PhD-level macroeconomist writing a highly detailed, structured briefing on the R-G Model, explicitly contrasting Monetary (R) and Growth (G) conditions.

{methodology_summary}

{metrics_summary}

{historical_summary}

{regression_summary}

{indicator_summary}

Structured Briefing Outline:

2. Calculation Methodology & Indicator Selection (Quantitative Detail):
   - Clearly outline the regression-based weighting method for both R and G indicators, explicitly referencing the provided regression weights tables.
   - Provide a detailed rationale for selecting each indicator, emphasizing their quantitative contributions explicitly.
   - Explicitly explain the normalization approach (Z-score normalization with a 120-month rolling window) and its economic rationale.
"""
        try:
            response_calculation_methodology = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert macroeconomist providing quantitatively detailed, structured, and highly educational economic analyses."},
                    {"role": "user", "content": prompt_calculation_methodology}
                ],
                temperature=0.2,
                max_tokens=3500
            )
            expert_review_calculation_methodology = response_calculation_methodology["choices"][0]["message"]["content"]
            st.markdown("### 2. Calculation Methodology & Indicator Selection")
            st.markdown(expert_review_calculation_methodology)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
        # Historical Trends & Attribution Analysis Query
        expanded_historical_summary = df_norm[['R_score', 'G_score']].tail(12).to_markdown()
        expanded_regression_summary = f"""
Top Monetary Conditions:
{df_weights_R_sorted.head(5).to_markdown()}

Top Growth Conditions:
{df_weights_G_sorted.head(5).to_markdown()}
"""
        prompt_historical_attribution = f"""
You are a PhD-level macroeconomist providing a highly detailed, structured briefing on the R-G Model, explicitly contrasting Monetary (R) and Growth (G) conditions.

{expanded_regression_summary}

Historical Scores (Past 12 Months):
{expanded_historical_summary}

📖 Structured Briefing Outline (Part 3):

3. Historical Trends & Attribution Analysis:
   - Explicitly identify key periods (past 12 months) with significant shifts in R, G, and R-G scores.
   - Explicitly quantify and discuss main contributing indicators using provided regression weights.
   - Clearly highlight indicators driving major shifts, providing numeric comparisons and contributions.
   - Explicitly reference months of highest and lowest scores, and largest R-G shifts.
   - Provide detailed analysis of economic implications of observed trends.
"""
        try:
            response_historical_attribution = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert macroeconomist providing quantitatively detailed, structured, accurate, and educational economic analyses."},
                    {"role": "user", "content": prompt_historical_attribution}
                ],
                temperature=0.2,
                max_tokens=5000
            )
            expert_review_historical_attribution = response_historical_attribution["choices"][0]["message"]["content"]
            st.markdown("### 3. Historical Trends & Attribution Analysis")
            st.markdown(expert_review_historical_attribution)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
        # Critical Methodology Evaluation & Recommendations Query
        prompt_standard = f"""
You are a PhD-level macroeconomist providing a concise and practical evaluation and recommendations for the R-G Model, based on the provided methodology and regression results.

{methodology_summary}

{regression_summary}

Structured Briefing Outline:

4. Critical Methodology Evaluation (Concise)

Provide a concise evaluation of indicator selection, normalization, and regression target suitability.

5. Actionable Recommendations (Concise & Direct)

Suggest clear, actionable methodological improvements to enhance model accuracy, interpretability, and predictive power, supported by economic rationale.

Your response should be structured, insightful, and clearly incorporate the provided regression findings.
"""
        try:
            response_standard = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert macroeconomist providing concise, structured, and actionable economic analyses and recommendations."},
                    {"role": "user", "content": prompt_standard}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            expert_review_standard = response_standard["choices"][0]["message"]["content"]
            st.markdown("### 4. Critical Methodology Evaluation & 5. Actionable Recommendations")
            st.markdown(expert_review_standard)
        except Exception as e:
            st.error(f"OpenAI API error: {e}")

