import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from datetime import datetime, timedelta

# --- 1. 頁面與 UI 設定 ---
st.set_page_config(page_title="主力狙擊儀表板 Pro Max", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    h1, h2, h3, h4, p, label, span { color: #ffffff !important; }
    .stNumberInput input { color: white; }
    div[data-testid="stMetricValue"] { color: #00e676; }
    .metric-card { background: linear-gradient(135deg, #1e2631 0%, #262d3a 100%); 
                   padding: 15px; border-radius: 10px; border: 1px solid #2d3748; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心演算法與函數 ---

# A. VWAP 計算
def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']
    df['VWAP'] = df['TP_Volume'].cumsum() / df['Volume'].cumsum()
    return df

# B. 蠟燭模式識別
def detect_patterns(df):
    """Detect candlestick patterns"""
    df['Pattern'] = ''
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        body = abs(current['Close'] - current['Open'])
        range_candle = current['High'] - current['Low']
        upper_shadow = current['High'] - max(current['Open'], current['Close'])
        lower_shadow = min(current['Open'], current['Close']) - current['Low']
        
        if (lower_shadow > 2 * body and upper_shadow < body * 0.3 and 
            current['Close'] > current['Open'] and range_candle > 0):
            df.at[df.index[i], 'Pattern'] = 'Hammer'
        elif (upper_shadow > 2 * body and lower_shadow < body * 0.3 and 
              current['Close'] < current['Open'] and range_candle > 0):
            df.at[df.index[i], 'Pattern'] = 'Shooting Star'
        elif (prev['Close'] < prev['Open'] and current['Close'] > current['Open'] and
              current['Open'] < prev['Close'] and current['Close'] > prev['Open']):
            df.at[df.index[i], 'Pattern'] = 'Bullish Engulfing'
        elif (prev['Close'] > prev['Open'] and current['Close'] < current['Open'] and
              current['Open'] > prev['Close'] and current['Close'] < prev['Open']):
            df.at[df.index[i], 'Pattern'] = 'Bearish Engulfing'
        elif body < range_candle * 0.1 and range_candle > 0:
            df.at[df.index[i], 'Pattern'] = 'Doji'
    
    return df

# C. AI 支撐壓力識別
def calculate_sr_levels(df):
    levels = []
    for i in range(2, len(df) - 2):
        if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i-2] and \
           df['High'].iloc[i] > df['High'].iloc[i+1] and df['High'].iloc[i] > df['High'].iloc[i+2]:
            levels.append((df.index[i], df['High'].iloc[i], "Resistance"))
        elif df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i-2] and \
             df['Low'].iloc[i] < df['Low'].iloc[i+1] and df['Low'].iloc[i] < df['Low'].iloc[i+2]:
            levels.append((df.index[i], df['Low'].iloc[i], "Support"))
    
    consolidated_levels = []
    for date, level, type_ in levels:
        is_far = True
        for _, existing_level, _ in consolidated_levels:
            if abs(level - existing_level) < (level * 0.015):
                is_far = False
                break
        if is_far:
            consolidated_levels.append((date, level, type_))
    return consolidated_levels

# D. SMC FVG 偵測
def check_fvg(df):
    df['FVG_Bullish'] = False
    for i in range(2, len(df)):
        prev_high = df['High'].iloc[i-2]
        curr_low = df['Low'].iloc[i]
        mid_close = df['Close'].iloc[i-1]
        mid_open = df['Open'].iloc[i-1]
        if mid_close > mid_open and curr_low > prev_high:
            df.iloc[i, df.columns.get_loc('FVG_Bullish')] = True
    return df

# E. 數據加載與指標計算
@st.cache_data(ttl=300)
def load_data(symbol, interval='1d', period='6mo'):
    try:
        df = yf.Ticker(symbol).history(interval=interval, period=period)
        if df.empty: return None
        
        df = calculate_vwap(df)
        df['EMA20'] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df['EMA50'] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
        
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        df['SpanA'] = ichimoku.ichimoku_a()
        df['SpanB'] = ichimoku.ichimoku_b()

        indicator_macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = indicator_macd.macd()
        df['MACD_Signal'] = indicator_macd.macd_signal()
        df['MACD_Hist'] = indicator_macd.macd_diff()
        
        df = check_fvg(df)
        df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
        df['Ant_Buy'] = (df['Volume'] > 1.5 * df['Vol_MA']) & (df['Close'] > df['Open'])
        df['Ant_Sell'] = (df['Volume'] > 1.5 * df['Vol_MA']) & (df['Close'] < df['Open'])
        df = detect_patterns(df)

        return df
    except Exception as e:
        st.error(f"無法獲取數據 ({symbol}): {e}")
        return None

# F. 計算評分
def calculate_score(df):
    if df is None or len(df) == 0:
        return 0, {}
    
    last_row = df.iloc[-1]
    current_price = last_row['Close']
    
    conditions = {}
    conditions['EMA多頭'] = (current_price > last_row['EMA20']) and (last_row['EMA20'] > last_row['EMA50'])
    cloud_top = max(last_row['SpanA'], last_row['SpanB'])
    conditions['Ichimoku雲上'] = current_price > cloud_top
    conditions['SMC FVG'] = df['FVG_Bullish'].tail(5).any()
    conditions['Ants資金'] = df['Ant_Buy'].tail(3).any()
    conditions['MACD多頭'] = last_row['MACD_Hist'] > 0
    conditions['VWAP之上'] = current_price > last_row['VWAP']
    
    score = 30
    if conditions['EMA多頭']: score += 15
    if conditions['Ichimoku雲上']: score += 10
    if conditions['SMC FVG']: score += 10
    if conditions['Ants資金']: score += 15
    if conditions['MACD多頭']: score += 10
    if conditions['VWAP之上']: score += 10
    
    return min(score, 100), conditions

# G. 回測引擎
def run_backtest(df, threshold=60):
    if df is None or len(df) < 50:
        return None
    
    results = []
    position = None
    
    for i in range(50, len(df)):
        temp_df = df.iloc[:i+1].copy()
        score, _ = calculate_score(temp_df)
        current_price = df.iloc[i]['Close']
        
        if position is None and score >= threshold:
            position = {'entry_price': current_price, 'entry_date': df.index[i]}
        elif position is not None and score < (threshold - 10):
            exit_price = current_price
            profit_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            results.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[i],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit_pct': profit_pct
            })
            position = None
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results)
    win_rate = len(results_df[results_df['profit_pct'] > 0]) / len(results_df) * 100
    avg_profit = results_df['profit_pct'].mean()
    total_return = results_df['profit_pct'].sum()
    max_drawdown = results_df['profit_pct'].min()
    
    return {
        'trades': len(results_df),
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'details': results_df
    }

# H. 股票篩選器
@st.cache_data(ttl=600)
def screen_stocks(tickers, interval='1d', period='3mo'):
    results = []
    for ticker in tickers:
        df = load_data(ticker, interval, period)
        if df is not None:
            score, conditions = calculate_score(df)
            last_price = df.iloc[-1]['Close']
            prev_close = df.iloc[-2]['Close'] if len(df) > 1 else last_price
            pct_change = ((last_price - prev_close) / prev_close) * 100
            
            results.append({
                '股票代號': ticker,
                '評分': score,
                '當前價格': round(last_price, 2),
                '漲跌幅%': round(pct_change, 2),
                'VWAP之上': '✅' if conditions.get('VWAP之上', False) else '❌',
                'EMA多頭': '✅' if conditions.get('EMA多頭', False) else '❌'
            })
    
    return pd.DataFrame(results).sort_values('評分', ascending=False)

# --- 3. 側邊欄設定 ---
st.sidebar.header("🛡️ 狙擊參數設定")
ticker = st.sidebar.text_input("股票代號", value="AAPL")

st.sidebar.divider()
st.sidebar.header("🤖 圖表圖層控制")
show_vwap = st.sidebar.checkbox("顯示 VWAP", value=True)
show_patterns = st.sidebar.checkbox("顯示蠟燭模式", value=True)
show_sr = st.sidebar.checkbox("顯示 AI 支撐/壓力線", value=True)
show_ants = st.sidebar.checkbox("顯示 Ants 資金螞蟻", value=True)
show_fvg = st.sidebar.checkbox("顯示 SMC FVG 缺口", value=True)

st.sidebar.divider()
st.sidebar.header("💰 部位計算器")
account_size = st.sidebar.number_input("帳戶總額 ($)", value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("風險比例 (%)", 1, 5, 2)
stop_loss_pct = st.sidebar.slider("停損 (%)", 1, 10, 3)

# --- 4. 主程式：多時間週期 Tabs ---
st.title("📊 主力狙擊儀表板 Pro Max")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 主儀表板", "⏱️ 5分鐘", "🕐 1小時", "🕓 4小時", "📅 日線", "🔍 回測 & 篩選器"])

timeframes = {
    "5分鐘": ("5m", "5d"),
    "1小時": ("1h", "1mo"),
    "4小時": ("1h", "3mo"),
    "日線": ("1d", "6mo")
}

# 共用函數：繪製圖表
def render_chart(df, ticker, timeframe_name, show_vwap, show_patterns, show_sr, show_ants, show_fvg):
    if df is None:
        st.warning("無法載入數據")
        return
    
    last_row = df.iloc[-1]
    current_price = last_row['Close']
    prev_price = df.iloc[-2]['Close'] if len(df) > 1 else current_price
    pct_change = ((current_price - prev_price) / prev_price) * 100
    
    score, conditions = calculate_score(df)
    sr_levels = calculate_sr_levels(df)
    
    # 上方儀表板卡片
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #888; font-size: 14px; margin: 0;">當前價格</p>
            <p style="color: #00e676; font-size: 28px; font-weight: bold; margin: 5px 0;">${current_price:.2f}</p>
            <p style="color: {'#00e676' if pct_change > 0 else '#ff4b4b'}; font-size: 14px; margin: 0;">
                {'+' if pct_change > 0 else ''}{pct_change:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #888; font-size: 14px; margin: 0;">多頭評分</p>
            <p style="color: {'#00e676' if score > 60 else '#ff4b4b' if score < 40 else '#ffa726'}; font-size: 28px; font-weight: bold; margin: 5px 0;">{score}</p>
            <p style="color: #888; font-size: 14px; margin: 0;">{'強勢' if score > 60 else '弱勢' if score < 40 else '中性'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        vol_change = ((last_row['Volume'] - last_row['Vol_MA']) / last_row['Vol_MA'] * 100) if last_row['Vol_MA'] > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #888; font-size: 14px; margin: 0;">成交量</p>
            <p style="color: #2196f3; font-size: 28px; font-weight: bold; margin: 5px 0;">{last_row['Volume']:,.0f}</p>
            <p style="color: {'#00e676' if vol_change > 0 else '#ff4b4b'}; font-size: 14px; margin: 0;">
                {'+' if vol_change > 0 else ''}{vol_change:.0f}% vs 均量
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        high_52w = df['High'].max()
        low_52w = df['Low'].min()
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #888; font-size: 14px; margin: 0;">區間高/低</p>
            <p style="color: #ff4b4b; font-size: 18px; font-weight: bold; margin: 5px 0;">${high_52w:.2f}</p>
            <p style="color: #00e676; font-size: 18px; font-weight: bold; margin: 0;">${low_52w:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.markdown(f"### 🎯 訊號分析")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={'text': "多頭強度", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00e676" if score > 60 else ("#ff4b4b" if score < 40 else "gray")},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 75, 75, 0.2)'},
                    {'range': [40, 60], 'color': 'rgba(255, 167, 38, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(0, 230, 118, 0.2)'}
                ]
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20,r=20,t=30,b=10), paper_bgcolor="#0e1117", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{timeframe_name}")
        
        st.markdown("#### ✅ 訊號清單")
        for name, condition in conditions.items():
            icon = "✅" if condition else "⬜"
            st.markdown(f"**{icon} {name}**")
        
        st.divider()
        
        st.markdown("#### 💰 建議部位")
        risk_amount = account_size * (risk_pct / 100)
        stop_loss_amount = current_price * (stop_loss_pct / 100)
        if stop_loss_amount > 0:
            shares = int(risk_amount / stop_loss_amount)
            position_value = shares * current_price
            st.success(f"**建議股數**: {shares} 股")
            st.info(f"**部位金額**: ${position_value:,.2f}")
            st.warning(f"**風險金額**: ${risk_amount:,.2f}")
            st.error(f"**停損價**: ${current_price - stop_loss_amount:.2f}")
    
    with col2:
        st.markdown(f"### 📈 {ticker} - {timeframe_name}")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        
        if show_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#ffeb3b', width=2, dash='dot'),
                                     name='VWAP'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#00e676', width=1),
                                 name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2962ff', width=1),
                                 name='EMA 50'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(width=0),
                                 showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], line=dict(width=0), fill='tonexty',
                                 fillcolor='rgba(255, 255, 255, 0.05)', name='Cloud'), row=1, col=1)
        
        if show_sr:
            for date, level, type_ in sr_levels:
                if current_price * 0.85 < level < current_price * 1.15:
                    color = "rgba(255, 82, 82, 0.6)" if level > current_price else "rgba(0, 230, 118, 0.6)"
                    fig.add_shape(type="line", x0=date, x1=df.index[-1], y0=level, y1=level,
                                  line=dict(color=color, width=1, dash="dash"), row=1, col=1)
        
        if show_ants:
            buy_ants = df[df['Ant_Buy']]
            if not buy_ants.empty:
                fig.add_trace(go.Scatter(x=buy_ants.index, y=buy_ants['Low']*0.995, mode='markers',
                                         marker=dict(symbol='circle', size=6, color='#00e676'),
                                         name='Buy Ants'), row=1, col=1)
            sell_ants = df[df['Ant_Sell']]
            if not sell_ants.empty:
                fig.add_trace(go.Scatter(x=sell_ants.index, y=sell_ants['High']*1.005, mode='markers',
                                         marker=dict(symbol='x', size=6, color='#ff4b4b'),
                                         name='Sell Ants'), row=1, col=1)
        
        if show_fvg:
            fvg_dates = df[df['FVG_Bullish']].index
            if not fvg_dates.empty:
                fig.add_trace(go.Scatter(x=fvg_dates, y=df.loc[fvg_dates, 'Low']*0.99, mode='markers',
                                         marker=dict(symbol='triangle-up', size=8, color='yellow'),
                                         name='SMC FVG'), row=1, col=1)
        
        if show_patterns:
            pattern_df = df[df['Pattern'] != '']
            if not pattern_df.empty:
                for idx, row in pattern_df.iterrows():
                    pattern_color = '#00e676' if 'Bullish' in row['Pattern'] or 'Hammer' in row['Pattern'] else '#ff4b4b'
                    fig.add_annotation(x=idx, y=row['High']*1.02, text=row['Pattern'][:3],
                                       showarrow=True, arrowhead=2, arrowcolor=pattern_color,
                                       font=dict(size=10, color=pattern_color), row=1, col=1)
        
        colors = ['#00e676' if val >= 0 else '#ff4b4b' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors,
                             name='MACD Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff', width=1),
                                 name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#ff6d00', width=1),
                                 name='Signal'), row=2, col=1)
        
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark",
                          hovermode='x unified', margin=dict(t=30, b=30))
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{timeframe_name}")

# Tab 1: 主儀表板
with tab1:
    df_main = load_data(ticker, '1d', '6mo')
    render_chart(df_main, ticker, "主儀表板 (日線)", show_vwap, show_patterns, show_sr, show_ants, show_fvg)

# Tab 2-5: 不同時間週期
for tab, (name, (interval, period)) in zip([tab2, tab3, tab4, tab5], timeframes.items()):
    with tab:
        df_tf = load_data(ticker, interval, period)
        render_chart(df_tf, ticker, name, show_vwap, show_patterns, show_sr, show_ants, show_fvg)

# Tab 6: 回測 & 篩選器
with tab6:
    st.header("🔬 回測模組")
    
    col_bt1, col_bt2 = st.columns([2, 1])
    
    with col_bt1:
        threshold = st.slider("進場評分門檻", 40, 80, 60, 5)
        
        if st.button("🚀 執行回測", type="primary"):
            df_backtest = load_data(ticker, '1d', '1y')
            with st.spinner("回測中..."):
                backtest_results = run_backtest(df_backtest, threshold)
            
            if backtest_results:
                st.success("✅ 回測完成!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("交易次數", backtest_results['trades'])
                col2.metric("勝率", f"{backtest_results['win_rate']:.1f}%")
                col3.metric("平均獲利", f"{backtest_results['avg_profit']:.2f}%")
                col4.metric("總報酬", f"{backtest_results['total_return']:.2f}%")
                
                st.dataframe(backtest_results['details'], use_container_width=True)
            else:
                st.warning("此參數無交易訊號，請調整門檻")
    
    with col_bt2:
        st.info("""
        **回測邏輯**:
        - 評分 ≥ 門檻 → 進場
        - 評分 < 門檻-10 → 出場
        - 計算歷史績效
        """)
    
    st.divider()
    st.header("🔍 股票篩選器")
    
    # S&P 500 熱門股票清單 (前100支)
    SP500_TOP_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
        "V", "XOM", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "KO", "PEP", "AVGO", "COST", "LLY", "ADBE", "MCD", "CSCO", "TMO", "ACN",
        "NFLX", "ABT", "NKE", "DIS", "VZ", "CMCSA", "INTC", "CRM", "WFC", "AMD",
        "DHR", "TXN", "NEE", "UPS", "PM", "QCOM", "INTU", "RTX", "BMY", "UNP",
        "HON", "AMGN", "LOW", "SPGI", "COP", "BA", "SBUX", "CAT", "IBM", "GE",
        "DE", "AXP", "BLK", "GILD", "ELV", "MDLZ", "ADI", "MMC", "AMT", "ISRG",
        "PLD", "LMT", "NOW", "REGN", "VRTX", "TJX", "SYK", "CI", "MO", "ZTS",
        "BKNG", "PGR", "TMUS", "CVS", "BDX", "DUK", "CB", "SO", "MMM", "AON",
        "GS", "TGT", "SCHW", "EQIX", "APD", "ADM", "C", "ITW", "SLB", "HUM"
    ]
    
    # 篩選模式選擇
    scan_mode = st.radio(
        "選擇篩選模式：",
        ["📝 手動輸入", "🚀 掃描 S&P 500 熱門股 (100支)", "🌐 掃描完整 S&P 500"],
        horizontal=True
    )
    
    if scan_mode == "📝 手動輸入":
        default_watchlist = "AAPL,MSFT,GOOGL,TSLA,NVDA,AMD,META,AMZN"
        watchlist_input = st.text_area("輸入股票代號 (逗號分隔)", value=default_watchlist)
        
        col_screen1, col_screen2 = st.columns([1, 3])
        with col_screen1:
            if st.button("🔎 開始篩選", type="primary"):
                tickers = [t.strip().upper() for t in watchlist_input.split(',') if t.strip()]
                with st.spinner(f"正在篩選 {len(tickers)} 檔股票..."):
                    screener_results = screen_stocks(tickers, '1d', '3mo')
                
                st.session_state['screener_results'] = screener_results
        
        with col_screen2:
            st.info("💡 點擊篩選後，結果會按評分排序。評分越高代表多頭訊號越強。")
    
    elif scan_mode == "🚀 掃描 S&P 500 熱門股 (100支)":
        st.info(f"📊 將自動掃描 {len(SP500_TOP_STOCKS)} 支 S&P 500 熱門股票")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            min_score = st.number_input("最低評分", 0, 100, 60, 5)
        with col2:
            if st.button("🚀 開始掃描", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner(f"正在掃描 {len(SP500_TOP_STOCKS)} 支股票..."):
                    screener_results = screen_stocks(SP500_TOP_STOCKS, '1d', '3mo')
                    
                    # 只顯示評分大於門檻的
                    if not screener_results.empty:
                        screener_results = screener_results[screener_results['評分'] >= min_score]
                    
                    st.session_state['screener_results'] = screener_results
                    progress_bar.progress(100)
                    status_text.success(f"✅ 掃描完成！找到 {len(screener_results)} 支評分 ≥ {min_score} 的股票")
        
        with col3:
            st.info("💡 建議：評分 ≥ 70 為強勢股，評分 ≥ 60 為中性偏多")
    
    else:  # 完整 S&P 500
        st.warning("⚠️ 掃描完整 S&P 500 需要較長時間（約5-10分鐘），建議先使用熱門股模式")
        st.info("📊 此功能會掃描所有 500+ 支 S&P 500 成分股")
        
        if st.button("🌐 開始完整掃描", type="primary"):
            st.error("⚠️ 完整掃描功能開發中。目前建議使用「熱門股」模式，已包含市值最大的100支股票。")
    
    # 顯示結果
    if 'screener_results' in st.session_state:
        st.divider()
        results = st.session_state['screener_results']
        
        if not results.empty:
            st.success(f"📊 找到 {len(results)} 支符合條件的股票")
            
            # 分類顯示
            col_a, col_b, col_c  = st.columns(3)
            strong_stocks = results[results['評分'] >= 70]
            medium_stocks = results[(results['評分'] >= 60) & (results['評分'] < 70)]
            weak_stocks = results[results['評分'] < 60]
            
            col_a.metric("🔥 強勢股 (≥70)", len(strong_stocks))
            col_b.metric("📊 中性股 (60-69)", len(medium_stocks))
            col_c.metric("⚠️ 弱勢股 (<60)", len(weak_stocks))
            
            st.dataframe(results, use_container_width=True, height=400)
        else:
            st.warning("未找到符合條件的股票，請降低評分門檻")