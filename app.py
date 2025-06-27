import streamlit as st
import pandas as pd
import numpy as np


# Streamlit App title
st.title("R-G Model Analysis")

# File uploader widget explicitly asking for Excel files
uploaded_file = st.file_uploader(
    "Upload Excel file with 'Tickers' and 'Hard' sheets", type="xlsx"
)

if uploaded_file:
    try:
        # Load Excel file
        xls = pd.ExcelFile(uploaded_file)
        
        # Read specific sheets
        df_tickers = xls.parse("Tickers")
        df_raw = xls.parse("Hard", header=None)

        # Displaying previews of sheets explicitly for user confirmation
        st.subheader("Preview of 'Tickers' Sheet")
        st.dataframe(df_tickers.head())

    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")

if uploaded_file:
    try:
        # --- Data Preparation ---

        # Extract tickers explicitly from df_raw (first row, every second column starting from index 1)
        tickers = df_raw.iloc[0, 1::2].tolist()

        series_list = []

        # Process each ticker explicitly
        for i, ticker in enumerate(tickers):
            dates = pd.to_datetime(df_raw.iloc[1:, i * 2], errors='coerce')
            values = pd.to_numeric(df_raw.iloc[1:, i * 2 + 1], errors='coerce')

            # Create a monthly resampled DataFrame explicitly for each ticker
            df_series = (
                pd.DataFrame({'Date': dates, ticker: values})
                .dropna(subset=['Date', ticker])
                .set_index('Date')
                .resample('ME')
                .ffill()
            )

            series_list.append(df_series)

        # Combine all tickers into one DataFrame explicitly
        df_monthly = pd.concat(series_list, axis=1).sort_index().dropna(how='all')


        # --- Explicit Z-score Normalization ---

        # Explicitly define a normalization function
        def zscore_normalize(df, window=120):
            return df.rolling(window, min_periods=1).apply(
                lambda x: (x[-1] - np.mean(x)) / np.std(x) if np.std(x) else 0,
                raw=False
            )

        df_norm = zscore_normalize(df_monthly)

    except Exception as e:
        st.error(f"Error during data preparation: {e}")

# === Financial Conditions Plot (Enhanced) ===
fig_financial_conditions = go.Figure()

color_tightening = '#E15759'
color_loosening = '#59A14F'

# Net R Score
fig_financial_conditions.add_trace(go.Scatter(
    x=df_norm.index, 
    y=df_norm['R_score'], 
    mode='lines',
    line=dict(color=color_tightening), 
    name='Net R'
))

# Net G Score
fig_financial_conditions.add_trace(go.Scatter(
    x=df_norm.index, 
    y=df_norm['G_score'], 
    mode='lines',
    line=dict(color=color_loosening), 
    name='Net G'
))

# 6-month moving averages
window_ma = 6
fig_financial_conditions.add_trace(go.Scatter(
    x=df_norm.index,
    y=df_norm['R_score'].rolling(window=window_ma).mean(),
    mode='lines',
    line=dict(color=color_tightening, dash='dot'),
    name='Net R MA (6-month)'
))

fig_financial_conditions.add_trace(go.Scatter(
    x=df_norm.index,
    y=df_norm['G_score'].rolling(window=window_ma).mean(),
    mode='lines',
    line=dict(color=color_loosening, dash='dot'),
    name='Net G MA (6-month)'
))

# Shaded areas for tightening and loosening
current_state = df_norm['R_score'].iloc[0] > df_norm['G_score'].iloc[0]
segment_x, segment_upper, segment_lower = [], [], []

for date, r_score, g_score in zip(df_norm.index, df_norm['R_score'], df_norm['G_score']):
    tightening = r_score > g_score
    upper, lower = max(r_score, g_score), min(r_score, g_score)

    if tightening != current_state and segment_x:
        color = 'rgba(225,87,89,0.3)' if current_state else 'rgba(89,161,79,0.3)'
        fig_financial_conditions.add_trace(go.Scatter(
            x=segment_x + segment_x[::-1], 
            y=segment_upper + segment_lower[::-1],
            fill='toself', 
            fillcolor=color, 
            mode='none', 
            showlegend=False
        ))
        segment_x, segment_upper, segment_lower = [], [], []
        current_state = tightening

    segment_x.append(date)
    segment_upper.append(upper)
    segment_lower.append(lower)

# Final shaded segment
color = 'rgba(225,87,89,0.3)' if current_state else 'rgba(89,161,79,0.3)'
fig_financial_conditions.add_trace(go.Scatter(
    x=segment_x + segment_x[::-1], 
    y=segment_upper + segment_lower[::-1],
    fill='toself', 
    fillcolor=color, 
    mode='none', 
    showlegend=False
))

# Layout adjustments for clarity
fig_financial_conditions.update_yaxes(
    showgrid=True, 
    gridcolor='lightgray', 
    zeroline=True, 
    zerolinewidth=2, 
    zerolinecolor='gray'
)

fig_financial_conditions.update_layout(
    title='Financial Conditions: Net R and G Scores',
    plot_bgcolor='white',
    hovermode='x unified',
    legend_title_text='Legend'
)

st.plotly_chart(fig_financial_conditions, use_container_width=True)

