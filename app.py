import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import MACD, EMAIndicator, IchimokuIndicator, ADXIndicator
from ta.volatility import KeltnerChannel, AverageTrueRange
from ta.volume import MFIIndicator
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (run this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- 1. 頁面與 UI 設定 ---
st.set_page_config(page_title="主力狙擊儀表板 Pro Max", layout="wide")

# 強制暗色系風格
st.markdown("""
<style>
    /* Global Theme */
    .stApp { background-color: #0b0e11; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* Typography */
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: 0.5px; }
    h4, h5, h6 { color: #a0a0a0 !important; font-weight: 500; }
    p, label, span { color: #cccccc !important; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-family: 'Courier New', monospace; font-weight: bold; }
    
    /* Cards */
    .metric-card { 
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363d; }
    
    /* Buttons */
    .stButton button { 
        background-color: #238636; 
        color: white; 
        border: none; 
        border-radius: 6px; 
        font-weight: bold;
        transition: all 0.2s;
    }
    .stButton button:hover { background-color: #2ea043; box-shadow: 0 0 10px rgba(46, 160, 67, 0.5); }
    
    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { 
        background-color: #0d1117; 
        color: white; 
        border: 1px solid #30363d; 
        border-radius: 6px; 
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb;
        color: white;
    }
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

# B. 蠟燭模式識別 (Custom Pattern Recognition)
def detect_patterns(df):
    """Detect candlestick patterns without TA-Lib"""
    df['Pattern'] = ''
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        body = abs(current['Close'] - current['Open'])
        range_candle = current['High'] - current['Low']
        upper_shadow = current['High'] - max(current['Open'], current['Close'])
        lower_shadow = min(current['Open'], current['Close']) - current['Low']
        
        # Hammer (看漲)
        if (lower_shadow > 2 * body and upper_shadow < body * 0.3 and 
            current['Close'] > current['Open'] and range_candle > 0):
            df.at[df.index[i], 'Pattern'] = 'Hammer'
        
        # Shooting Star (看跌)
        elif (upper_shadow > 2 * body and lower_shadow < body * 0.3 and 
              current['Close'] < current['Open'] and range_candle > 0):
            df.at[df.index[i], 'Pattern'] = 'Shooting Star'
        
        # Bullish Engulfing (看漲吞沒)
        elif (prev['Close'] < prev['Open'] and current['Close'] > current['Open'] and
              current['Open'] < prev['Close'] and current['Close'] > prev['Open']):
            df.at[df.index[i], 'Pattern'] = 'Bullish Engulfing'
        
        # Bearish Engulfing (看跌吞沒)
        elif (prev['Close'] > prev['Open'] and current['Close'] < current['Open'] and
              current['Open'] > prev['Close'] and current['Close'] < prev['Open']):
            df.at[df.index[i], 'Pattern'] = 'Bearish Engulfing'
        
        # Doji (十字星)
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
    
    # 合併相近的線
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
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(symbol, interval='1d', period='6mo'):
    try:
        df = yf.Ticker(symbol).history(interval=interval, period=period)
        if df.empty: return None
        
        # VWAP
        df = calculate_vwap(df)
        
        # EMA
        df['EMA20'] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df['EMA50'] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
        
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
        df['SpanA'] = ichimoku.ichimoku_a()
        df['SpanB'] = ichimoku.ichimoku_b()

        # MACD
        indicator_macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = indicator_macd.macd()
        df['MACD_Signal'] = indicator_macd.macd_signal()
        df['MACD_Hist'] = indicator_macd.macd_diff()
        
        # FVG
        df = check_fvg(df)

        # Ants
        df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
        df['Ant_Buy'] = (df['Volume'] > 1.5 * df['Vol_MA']) & (df['Close'] > df['Open'])
        df['Ant_Sell'] = (df['Volume'] > 1.5 * df['Vol_MA']) & (df['Close'] < df['Open'])
        
        # Pattern Detection
        df = detect_patterns(df)

        # --- Advanced Indicators (Pro Max) ---
        # 1. ADX (Trend Strength)
        adx_indicator = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx_indicator.adx()
        
        # 2. Keltner Channels (Volatility)
        keltner = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
        df['KC_High'] = keltner.keltner_channel_hband()
        df['KC_Low'] = keltner.keltner_channel_lband()
        df['KC_Mid'] = keltner.keltner_channel_mband()
        
        # 3. MFI (Money Flow)
        mfi_indicator = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
        df['MFI'] = mfi_indicator.money_flow_index()

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
    
    # --- Advanced Conditions ---
    conditions['ADX強勢'] = last_row['ADX'] > 25
    conditions['MFI資金流入'] = last_row['MFI'] > 50
    conditions['Keltner突破'] = current_price > last_row['KC_High']
    
    score = 30  # 基礎分
    if conditions['EMA多頭']: score += 15
    if conditions['Ichimoku雲上']: score += 10
    if conditions['SMC FVG']: score += 10
    if conditions['Ants資金']: score += 15
    if conditions['MACD多頭']: score += 10
    if conditions['VWAP之上']: score += 10
    
    # Advanced Boosters
    if conditions['ADX強勢']: score += 5
    if conditions['MFI資金流入']: score += 5
    if conditions['Keltner突破']: score += 5
    
    # Penalties
    if last_row['MFI'] > 80: score -= 5 # Overbought
    if last_row['ADX'] < 20: score -= 5 # Weak Trend
    
    return min(score, 100), conditions

# === ENHANCED TRADING FUNCTIONS ===

# G. 計算 ATR (Average True Range)
def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility measurement"""
    if df is None or len(df) < period:
        return None
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

# H. 支撐壓力計算 (增強版)
def calculate_support_resistance_levels(df, lookback=20, tolerance=0.02):
    """Calculate support and resistance levels using swing highs/lows"""
    if df is None or len(df) < lookback:
        return {'support': [], 'resistance': []}
    
    resistance_levels = []
    support_levels = []
    
    # 使用最近的數據
    recent_df = df.tail(lookback * 3)
    
    for i in range(2, len(recent_df) - 2):
        # 阻力位 (swing high)
        if (recent_df['High'].iloc[i] > recent_df['High'].iloc[i-1] and 
            recent_df['High'].iloc[i] > recent_df['High'].iloc[i-2] and
            recent_df['High'].iloc[i] > recent_df['High'].iloc[i+1] and 
            recent_df['High'].iloc[i] > recent_df['High'].iloc[i+2]):
            resistance_levels.append(recent_df['High'].iloc[i])
        
        # 支撐位 (swing low)
        if (recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i-1] and 
            recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i-2] and
            recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i+1] and 
            recent_df['Low'].iloc[i] < recent_df['Low'].iloc[i+2]):
            support_levels.append(recent_df['Low'].iloc[i])
    
    # 合併接近的水平
    def consolidate_levels(levels, tolerance):
        if not levels:
            return []
        levels = sorted(levels)
        consolidated = [levels[0]]
        for level in levels[1:]:
            if abs(level - consolidated[-1]) / consolidated[-1] > tolerance:
                consolidated.append(level)
        return consolidated
    
    support_levels = consolidate_levels(support_levels, tolerance)
    resistance_levels = consolidate_levels(resistance_levels, tolerance)
    
    return {'support': support_levels, 'resistance': resistance_levels}

# I. 生成交易訊號
def generate_trade_signal(df):
    """Generate buy/sell/hold signals based on momentum and technical indicators"""
    if df is None or len(df) < 50:
        return '觀望', {}
    
    score, conditions = calculate_score(df)
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row
    
    # 計算前一根K線的評分
    prev_df = df.iloc[:-1]
    prev_score, _ = calculate_score(prev_df)
    
    signal_details = {
        'current_score': score,
        'prev_score': prev_score,
        'momentum_shift': score - prev_score
    }
    
    # 買入訊號條件
    buy_conditions = [
        score >= 60,  # 當前評分高
        score - prev_score >= 10,  # 評分快速上升
        last_row['MACD_Hist'] > 0,  # MACD為正
        last_row['Close'] > last_row['EMA20'],  # 價格在均線上
    ]
    
    # 賣出訊號條件
    sell_conditions = [
        score < 40,  # 評分低
        score - prev_score <= -15,  # 評分快速下降
        last_row['MACD_Hist'] < 0 and prev_row['MACD_Hist'] > 0,  # MACD死叉
    ]
    
    # 決定訊號
    if sum(buy_conditions) >= 3:
        signal = '買入'
    elif sum(sell_conditions) >= 2:
        signal = '賣出'
    else:
        signal = '觀望'
    
    signal_details['buy_conditions_met'] = sum(buy_conditions)
    signal_details['sell_conditions_met'] = sum(sell_conditions)
    
    return signal, signal_details

# J. AI 價格推薦
def calculate_price_recommendations(df):
    """Calculate recommended entry, target, and stop-loss prices"""
    if df is None or len(df) < 50:
        return None
    
    current_price = df.iloc[-1]['Close']
    
    # 計算ATR
    df_with_atr = df.copy()
    df_with_atr['ATR'] = calculate_atr(df_with_atr, 14)
    atr = df_with_atr['ATR'].iloc[-1]
    
    if pd.isna(atr) or atr == 0:
        atr = current_price * 0.02  # 使用2%作為後備
    
    # 獲取支撐壓力位
    levels = calculate_support_resistance_levels(df, lookback=20)
    
    # 尋找最近的支撐位作為買入參考
    recent_supports = [s for s in levels['support'] if s < current_price and s > current_price * 0.90]
    if recent_supports:
        nearest_support = max(recent_supports)
        entry_price = nearest_support + (atr * 0.3)  # 支撐位上方一點
    else:
        # 如果沒有明顯支撐，使用當前價格下方的ATR
        entry_price = current_price - (atr * 0.5)
    
    # 停損價：支撐位下方或使用ATR
    if recent_supports:
        stop_loss = max(recent_supports) - (atr * 0.8)
    else:
        stop_loss = entry_price - (atr * 2.0)  # 2 ATR止損
    
    # 目標價：尋找最近的阻力位
    recent_resistances = [r for r in levels['resistance'] if r > current_price and r < current_price * 1.15]
    if recent_resistances:
        target_price = min(recent_resistances)
    else:
        # 如果沒有明顯阻力，使用風險報酬比
        risk = entry_price - stop_loss
        target_price = entry_price + (risk * 2.5)  # 2.5:1 風險報酬比
    
    # 計算風險報酬比
    risk = entry_price - stop_loss
    reward = target_price - entry_price
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    return {
        'entry_price': round(entry_price, 2),
        'target_price': round(target_price, 2),
        'stop_loss': round(stop_loss, 2),
        'risk_reward_ratio': round(risk_reward_ratio, 2),
        'atr': round(atr, 2),
        'current_price': round(current_price, 2),
        'support_levels': [round(s, 2) for s in levels['support']],
        'resistance_levels': [round(r, 2) for r in levels['resistance']]
    }

# K. 回測引擎 (Enhanced)
def run_backtest(df):
    """Advanced backtesting using signals and ATR-based risk management"""
    if df is None or len(df) < 50:
        return None
    
    results = []
    position = None
    
    # Iterate through data (start after enough data for indicators)
    for i in range(50, len(df)):
        current_date = df.index[i]
        current_row = df.iloc[i]
        current_price = current_row['Close']
        
        # We need the full history up to this point for the signal function to work (requires > 50 rows)
        window_df = df.iloc[:i+1] 
        
        # Check signals
        signal, _ = generate_trade_signal(window_df)
        
        # --- Entry Logic ---
        if position is None:
            if signal == '買入':
                # Calculate dynamic stop loss and target
                recs = calculate_price_recommendations(df.iloc[:i+1])
                
                if recs:
                    position = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'stop_loss': recs['stop_loss'],
                        'target_price': recs['target_price'],
                        'shares': 100 
                    }
        
        # --- Exit Logic ---
        elif position is not None:
            # 1. Stop Loss Hit
            if current_row['Low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                reason = 'Stop Loss'
                
            # 2. Target Hit
            elif current_row['High'] >= position['target_price']:
                exit_price = position['target_price']
                reason = 'Target Hit'
                
            # 3. Sell Signal (Trend Reversal)
            elif signal == '賣出':
                exit_price = current_price
                reason = 'Signal Reversal'
            
            else:
                continue # Hold position
                
            # Execute Exit
            profit_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            results.append({
                'entry_date': position['entry_date'],
                'exit_date': current_date,
                'entry_price': round(position['entry_price'], 2),
                'exit_price': round(exit_price, 2),
                'stop_loss': round(position['stop_loss'], 2),
                'target': round(position['target_price'], 2),
                'profit_pct': round(profit_pct, 2),
                'reason': reason
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

# L. 股票篩選器 (Enhanced)
@st.cache_data(ttl=600)
def screen_stocks(tickers, interval='1d', period='3mo'):
    """Screen multiple stocks and return scores with recommendations"""
    results = []
    for ticker in tickers:
        df = load_data(ticker, interval, period)
        if df is not None:
            score, conditions = calculate_score(df)
            signal, _ = generate_trade_signal(df)  # NEW
            prices = calculate_price_recommendations(df)  # NEW
            
            last_price = df.iloc[-1]['Close']
            prev_close = df.iloc[-2]['Close'] if len(df) > 1 else last_price
            pct_change = ((last_price - prev_close) / prev_close) * 100
            
            result = {
                '股票代號': ticker,
                '訊號': signal,  # NEW
                '評分': score,
                '當前價格': round(last_price, 2),
                '漲跌幅%': round(pct_change, 2),
            }
            
            # Add price recommendations if available
            if prices:
                result.update({
                    '建議買入價': prices['entry_price'],
                    '目標價': prices['target_price'],
                    '停損價': prices['stop_loss'],
                    '風險報酬比': f"{prices['risk_reward_ratio']}:1"
                })
            
            results.append(result)
    
    return pd.DataFrame(results).sort_values('評分', ascending=False)

# --- M. Intelligence Layer (New Features) ---

def get_sentiment_analysis(ticker_symbol):
    """
    Fetches news and calculates a sentiment score (-1 to 1).
    Returns: Score, List of Headlines
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            return 0, []

        analyzer = SentimentIntensityAnalyzer()
        scores = []
        headlines = []

        for item in news[:7]: # Analyze top 7 recent articles
            title = item.get('title', '')
            headlines.append(title)
            # Compound score gives a metric from -1 (Negative) to +1 (Positive)
            sentiment = analyzer.polarity_scores(title)
            scores.append(sentiment['compound'])

        if not scores:
            return 0, []

        avg_score = sum(scores) / len(scores)
        return avg_score, headlines

    except Exception as e:
        # st.error(f"Error in Sentiment Engine: {e}") # Suppress error to avoid UI clutter
        return 0, []

def plot_sentiment_gauge(score):
    """
    Visualizes sentiment score on a Gauge Chart (-1 to 1)
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI News Sentiment", 'font': {'size': 20, 'color': 'white'}},
        delta = {'reference': 0, 'increasing': {'color': "#00e676"}, 'decreasing': {'color': "#ff4b4b"}},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#2962ff"},
            'bgcolor': "#0e1117",
            'borderwidth': 2,
            'bordercolor': "#30363d",
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(255, 75, 75, 0.3)'},
                {'range': [-0.3, 0.3], 'color': 'rgba(128, 128, 128, 0.3)'},
                {'range': [0.3, 1], 'color': 'rgba(0, 230, 118, 0.3)'}],
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="#0e1117", font={'color': "white"})
    return fig

def get_institutional_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    # --- 1. Major Holders (Ownership Breakdown) ---
    major_holders_df = pd.DataFrame()
    try:
        data = ticker.major_holders
        if data is not None and not data.empty:
            # yfinance returns a DataFrame where the first column is the value/percentage
            # and the second column is the description.
            major_holders_df = data.copy()
            major_holders_df.columns = ['Value', 'Category']
            major_holders_df = major_holders_df.set_index('Category')
            # Clean up the index names for better display
            major_holders_df.index = major_holders_df.index.str.replace(r'[\d\.]+% of | outstanding shares', '', regex=True).str.strip()
    except Exception as e:
        # print(f"Warning: Failed to fetch major holders for {ticker_symbol}. Error: {e}")
        pass # Return empty dataframe on failure

    # --- 2. Institutional Holders (Specific Funds) ---
    fund_holders_df = pd.DataFrame()
    try:
        data = ticker.institutional_holders
        if data is not None and not data.empty:
            fund_holders_df = data[['Holder', 'Shares', 'Date Reported', '% Out']].head(10).copy()
    except Exception as e:
        # print(f"Warning: Failed to fetch fund holders for {ticker_symbol}. Error: {e}")
        pass # Return empty dataframe on failure
        
    return major_holders_df, fund_holders_df

# --- 3. 側邊欄設定 ---
# --- 3. 側邊欄設定 (Sidebar) ---
with st.sidebar:
    st.markdown("## 🛡️ 狙擊控制台")
    
    # 1. 股票選擇 (Ticker Selection)
    with st.expander("🔍 股票設定", expanded=True):
        # Quick Select History (Mockup for now, could be dynamic later)
        quick_picks = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "GOOGL", "META", "AMZN"]
        selected_quick = st.selectbox("快速選擇", ["自定義"] + quick_picks, index=0)
        
        if selected_quick != "自定義":
            st.session_state['ticker'] = selected_quick
            
        # Text Input
        if 'ticker' not in st.session_state:
            st.session_state['ticker'] = 'AAPL'
            
        ticker = st.text_input("輸入代號 (例如: COIN)", value=st.session_state['ticker']).upper()
        st.session_state['ticker'] = ticker # Sync back
        
    # 2. 圖表圖層 (Visuals)
    with st.expander("🎨 圖表顯示", expanded=False):
        show_vwap = st.checkbox("VWAP (成交量加權均價)", value=True)
        show_patterns = st.checkbox("K線型態識別", value=True)
        show_sr = st.checkbox("AI 支撐/壓力線", value=True)
        show_ants = st.checkbox("主力資金 (Ants)", value=True)
        show_fvg = st.checkbox("SMC 缺口 (FVG)", value=True)
        show_kc = st.checkbox("Keltner Channels (波動通道)", value=True)

    # 3. 風險管理 (Risk)
    with st.expander("💰 資金與風險", expanded=False):
        account_size = st.number_input("帳戶總額 ($)", value=10000.0, step=1000.0)
        risk_pct = st.slider("單筆風險 (%)", 0.5, 5.0, 2.0, 0.1)
        stop_loss_pct = st.slider("預設停損 (%)", 1.0, 10.0, 3.0, 0.5)
        
    st.divider()
    st.markdown("### 🚀 Pro Max v2.0")
    st.caption("Powered by Gemini 2.0 Flash")

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
def render_chart(df, ticker, timeframe_name, show_vwap, show_patterns, show_sr, show_ants, show_fvg, show_kc):
    if df is None:
        st.warning("無法載入數據")
        return
    
    last_row = df.iloc[-1]
    current_price = last_row['Close']
    prev_price = df.iloc[-2]['Close'] if len(df) > 1 else current_price
    pct_change = ((current_price - prev_price) / prev_price) * 100
    
    score, conditions = calculate_score(df)
    sr_levels = calculate_sr_levels(df)
    recommendations = calculate_price_recommendations(df)
    
    # --- 1. Hero Banner ---
    status_color = "#00e676" if score > 60 else "#ff4b4b" if score < 40 else "#ffa726"
    status_text = "STRONG BUY" if score > 75 else "BUY" if score > 60 else "SELL" if score < 40 else "NEUTRAL"
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(14,17,23,1) 0%, rgba(22,27,34,1) 100%); 
                padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 20px;
                display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h2 style="margin:0; color:white;">{ticker} <span style="font-size: 18px; color: #888;">{timeframe_name}</span></h2>
            <h1 style="margin:0; font-size: 48px; color: {status_color};">${current_price:.2f}</h1>
        </div>
        <div style="text-align: right;">
            <div style="background-color: {status_color}20; padding: 5px 15px; border-radius: 20px; border: 1px solid {status_color}; display: inline-block;">
                <span style="color: {status_color}; font-weight: bold; font-size: 16px;">{status_text}</span>
            </div>
            <p style="margin: 5px 0 0 0; font-size: 14px; color: #888;">Score: <span style="color: white; font-weight: bold;">{score}/100</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- 2. Key Metrics Row ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("漲跌幅", f"{pct_change:+.2f}%", delta_color="normal")
    with col_m2:
        vol_change = ((last_row['Volume'] - last_row['Vol_MA']) / last_row['Vol_MA'] * 100) if last_row['Vol_MA'] > 0 else 0
        st.metric("成交量變動", f"{vol_change:+.0f}%", f"{last_row['Volume']:,.0f}")
    with col_m3:
        atr = recommendations['atr'] if recommendations else 0
        st.metric("ATR (波動率)", f"{atr:.2f}")
    with col_m4:
        adx = last_row['ADX']
        st.metric("ADX (趨勢強度)", f"{adx:.1f}", delta="強勢" if adx > 25 else "盤整", delta_color="normal")
        
    st.divider()
    
    # --- 3. Main Content Area ---
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        # Chart Logic
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
        
        # Indicators
        if show_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#ffeb3b', width=2, dash='dot'), name='VWAP'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#00e676', width=1), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#2962ff', width=1), name='EMA 50'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name='Cloud'), row=1, col=1)
        
        # Keltner Channels
        if show_kc:
            fig.add_trace(go.Scatter(x=df.index, y=df['KC_High'], line=dict(color='rgba(0, 230, 118, 0.3)', width=1), name='KC High'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['KC_Low'], line=dict(color='rgba(255, 82, 82, 0.3)', width=1), name='KC Low'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['KC_Mid'], line=dict(color='rgba(41, 98, 255, 0.5)', width=1, dash='dot'), name='KC Mid'), row=1, col=1)
        
        if show_sr:
            for date, level, type_ in sr_levels:
                if current_price * 0.85 < level < current_price * 1.15:
                    color = "rgba(255, 82, 82, 0.6)" if level > current_price else "rgba(0, 230, 118, 0.6)"
                    fig.add_shape(type="line", x0=date, x1=df.index[-1], y0=level, y1=level, line=dict(color=color, width=1, dash="dash"), row=1, col=1)
        
        if show_ants:
            buy_ants = df[df['Ant_Buy']]
            if not buy_ants.empty:
                fig.add_trace(go.Scatter(x=buy_ants.index, y=buy_ants['Low']*0.995, mode='markers', marker=dict(symbol='circle', size=6, color='#00e676'), name='Buy Ants'), row=1, col=1)
            sell_ants = df[df['Ant_Sell']]
            if not sell_ants.empty:
                fig.add_trace(go.Scatter(x=sell_ants.index, y=sell_ants['High']*1.005, mode='markers', marker=dict(symbol='x', size=6, color='#ff4b4b'), name='Sell Ants'), row=1, col=1)
        
        if show_fvg:
            fvg_dates = df[df['FVG_Bullish']].index
            if not fvg_dates.empty:
                fig.add_trace(go.Scatter(x=fvg_dates, y=df.loc[fvg_dates, 'Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=8, color='yellow'), name='SMC FVG'), row=1, col=1)
        
        if show_patterns:
            pattern_df = df[df['Pattern'] != '']
            if not pattern_df.empty:
                for idx, row in pattern_df.iterrows():
                    pattern_color = '#00e676' if 'Bullish' in row['Pattern'] or 'Hammer' in row['Pattern'] else '#ff4b4b'
                    fig.add_annotation(x=idx, y=row['High']*1.02, text=row['Pattern'][:3], showarrow=True, arrowhead=2, arrowcolor=pattern_color, font=dict(size=10, color=pattern_color), row=1, col=1)
        
        # MACD
        colors = ['#00e676' if val >= 0 else '#ff4b4b' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff', width=1), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#ff6d00', width=1), name='Signal'), row=2, col=1)
        
        fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", hovermode='x unified', margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="#0e1117")
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{timeframe_name}")
        
        # --- Intelligence Layer Integration ---
        st.markdown("---")
        st.subheader("📡 Pro Max Intelligence Layer")
        
        col_intel1, col_intel2 = st.columns([1, 1])
        
        # Sentiment
        with col_intel1:
            st.markdown("#### 📰 市場情緒脈動 (News Sentiment)")
            sentiment_score, headlines = get_sentiment_analysis(ticker)
            st.plotly_chart(plot_sentiment_gauge(sentiment_score), use_container_width=True, key=f"sent_{timeframe_name}")
            
            with st.expander("最新新聞頭條", expanded=False):
                if headlines:
                    for h in headlines:
                        st.caption(f"• {h}")
                else:
                    st.caption("暫無新聞數據")

        # Institutional Data
        with col_intel2:
            st.markdown("#### 🏢 機構與內部人籌碼 (Institutional Radar)")
            major_holders, fund_holders = get_institutional_data(ticker)
            
            if not major_holders.empty:
                st.dataframe(major_holders, use_container_width=True, height=180)
            else:
                st.warning("無法獲取主要持有者數據")
                
            st.markdown("---")
            
            if not fund_holders.empty:
                st.markdown("**頂級機構持倉**")
                st.dataframe(fund_holders.style.background_gradient(cmap="Greens", subset=['% Out']), height=250, use_container_width=True)
            else:
                st.info("無法獲取具體機構持倉數據")
        
    with col_side:
        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': status_color}, 'bgcolor': "white",
                   'steps': [{'range': [0, 40], 'color': 'rgba(255, 75, 75, 0.2)'},
                             {'range': [40, 60], 'color': 'rgba(255, 167, 38, 0.2)'},
                             {'range': [60, 100], 'color': 'rgba(0, 230, 118, 0.2)'}]}
        ))
        fig_gauge.update_layout(height=180, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="#0e1117", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{timeframe_name}")
        
        # Checklist
        with st.expander("✅ 訊號清單", expanded=True):
            for name, condition in conditions.items():
                st.markdown(f"{'✅' if condition else '⬜'} {name}")
                
        # Recommendations
        if recommendations:
            with st.expander("🤖 AI 建議", expanded=True):
                st.markdown(f"**進場**: ${recommendations['entry_price']}")
                st.markdown(f"**目標**: ${recommendations['target_price']}")
                st.markdown(f"**停損**: ${recommendations['stop_loss']}")
                
        # Position
        with st.expander("💰 部位試算", expanded=False):
            risk_amount = account_size * (risk_pct / 100)
            stop_loss_amount = current_price * (stop_loss_pct / 100)
            if stop_loss_amount > 0:
                shares = int(risk_amount / stop_loss_amount)
                st.info(f"建議股數: {shares}")
                st.warning(f"風險: ${risk_amount:.0f}")


# === Tab 1: 主儀表板 (日線) ===
with tab1:
    df_main = load_data(ticker, '1d', '6mo')
    render_chart(df_main, ticker, "主儀表板 (日線)", show_vwap, show_patterns, show_sr, show_ants, show_fvg, show_kc)

# === Tab 2-5: 不同時間週期 ===
for tab, (name, (interval, period)) in zip([tab2, tab3, tab4, tab5], timeframes.items()):
    with tab:
        df_tf = load_data(ticker, interval, period)
        render_chart(df_tf, ticker, name, show_vwap, show_patterns, show_sr, show_ants, show_fvg, show_kc)

# === Tab 6: 回測 & 篩選器 ===
with tab6:
    st.header("🔬 智能回測模組")
    
    col_bt1, col_bt2 = st.columns([3, 1])
    
    with col_bt1:
        st.info("🤖 回測系統已升級：使用 AI 訊號 + ATR 動態停損停利")
        
        if st.button("🚀 執行智能回測", type="primary", use_container_width=True):
            df_backtest = load_data(ticker, '1d', '1y')
            with st.spinner("正在執行 AI 策略回測..."):
                backtest_results = run_backtest(df_backtest)
            
            st.session_state['backtest_results'] = backtest_results
            
        if 'backtest_results' in st.session_state and st.session_state['backtest_results']:
            res = st.session_state['backtest_results']
            st.success("✅ 回測完成!")
            
            # Metrics Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總交易次數", res['trades'])
            m2.metric("勝率", f"{res['win_rate']:.1f}%")
            m3.metric("平均獲利", f"{res['avg_profit']:.2f}%")
            m4.metric("總報酬率", f"{res['total_return']:.2f}%")
            
            st.subheader("📝 交易明細")
            st.dataframe(
                res['details'],
                use_container_width=True,
                height=400,
                column_config={
                    "entry_date": st.column_config.DateColumn("進場日期"),
                    "exit_date": st.column_config.DateColumn("出場日期"),
                    "entry_price": st.column_config.NumberColumn("進場價", format="$%.2f"),
                    "exit_price": st.column_config.NumberColumn("出場價", format="$%.2f"),
                    "profit_pct": st.column_config.NumberColumn("獲利 %", format="%.2f%%"),
                    "reason": st.column_config.TextColumn("出場原因"),
                }
            )
        elif 'backtest_results' in st.session_state:
             st.warning("此期間無符合策略的交易訊號")

    with col_bt2:
        st.markdown("### 🧠 策略邏輯")
        with st.expander("策略說明", expanded=True):
            st.markdown("""
            **進場條件**:
            - 多頭評分 > 60
            - 動能轉強 (Score Delta > 10)
            - MACD 黃金交叉
            
            **出場條件**:
            1. **停損**: 觸發 ATR 動態停損
            2. **停利**: 達到阻力位目標
            3. **反轉**: 出現賣出訊號
            """)
    
    st.divider()
    st.divider()
    st.header("🔍 股票篩選器")
    
    # Predefined Lists
    tech_giants = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
    semis = ["NVDA", "AMD", "TSM", "AVGO", "QCOM", "INTC", "MU"]
    crypto_stocks = ["COIN", "MSTR", "MARA", "RIOT", "HOOD"]
    top_100_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "V", "JNJ", "WMT", "JPM", "MA", "PG", "UNH", "DIS", "HD", "VZ", "BAC", "KO", "PFE", "INTC", "CSCO", "CMCSA", "PEP", "WFC", "XOM", "CVX", "MRK", "ABT", "T", "ADBE", "CRM", "AVGO", "NKE", "ACN", "TMO", "MCD", "ABBV", "DHR", "NEE", "LIN", "TXN", "PM", "COST", "UNP", "QCOM", "BMY", "UPS", "LOW", "MS", "HON", "AMGN", "SBUX", "IBM", "GE", "DE", "CAT", "GS", "MMM", "INT", "AMT", "BLK", "C", "SCHW", "CVS", "LMT", "AXP", "TGT", "ISRG", "MDT", "PYPL", "SYK", "ZTS", "NOW", "ADP", "BKNG", "ADI", "AMD", "GILD", "MU", "LRCX", "TJX", "CB", "MMC", "CSX", "CI", "PNC", "USB", "TFC", "MO", "COP", "EOG", "SLB", "OXY", "VLO", "PSX", "KMI", "WMB"]

    # Quick Load Buttons
    st.markdown("##### ⚡ 快速載入清單")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    if col_q1.button("科技巨頭 (Mag 7)"): st.session_state['watchlist_selected'] = tech_giants
    if col_q2.button("半導體 (Semis)"): st.session_state['watchlist_selected'] = semis
    if col_q3.button("加密概念 (Crypto)"): st.session_state['watchlist_selected'] = crypto_stocks
    if col_q4.button("熱門 Top 100"): st.session_state['watchlist_selected'] = top_100_tickers
    
    # Multiselect Input
    if 'watchlist_selected' not in st.session_state:
        st.session_state['watchlist_selected'] = tech_giants
        
    selected_tickers = st.multiselect("選擇或輸入股票代號", 
                                      options=list(set(top_100_tickers + tech_giants + semis + crypto_stocks)),
                                      default=st.session_state['watchlist_selected'])
    
    # Action Button
    if st.button("🔎 開始掃描", type="primary", use_container_width=True):
        if not selected_tickers:
            st.warning("請至少選擇一檔股票")
        else:
            with st.spinner(f"正在分析 {len(selected_tickers)} 檔標的..."):
                screener_results = screen_stocks(selected_tickers, '1d', '3mo')
                st.session_state['screener_results'] = screener_results
    
    # Results Display
    if 'screener_results' in st.session_state:
        results = st.session_state['screener_results']
        if not results.empty:
            st.markdown(f"### 🎯 篩選結果 ({len(results)} 檔)")
            st.caption("💡 點擊表格中的股票可直接切換主儀表板")
            
            # Configure Columns
            event = st.dataframe(
                results,
                use_container_width=True,
                height=500,
                column_config={
                    "股票代號": st.column_config.TextColumn("代號", width="small"),
                    "訊號": st.column_config.TextColumn("AI 訊號", width="medium"),
                    "評分": st.column_config.ProgressColumn("多頭評分", format="%d", min_value=0, max_value=100, width="medium"),
                    "當前價格": st.column_config.NumberColumn("價格", format="$%.2f"),
                    "漲跌幅%": st.column_config.NumberColumn("漲跌幅", format="%.2f%%"),
                    "建議買入價": st.column_config.NumberColumn("買入", format="$%.2f"),
                    "目標價": st.column_config.NumberColumn("目標", format="$%.2f"),
                    "停損價": st.column_config.NumberColumn("停損", format="$%.2f"),
                },
                on_select="rerun",
                selection_mode="single-row"
            )
            
            # Handle click
            if event.selection.rows:
                selected_idx = event.selection.rows[0]
                selected_ticker = results.iloc[selected_idx]['股票代號']
                if st.session_state['ticker'] != selected_ticker:
                    st.session_state['ticker'] = selected_ticker
                    st.rerun()
