import nltk
nltk.download('vader_lexicon', quiet=True)
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MarianMTModel, MarianTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from pytrends.request import TrendReq
import numpy as np
import urllib.parse
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Setup Sentiment Analyzer ---
analyzer = SentimentIntensityAnalyzer()

# --- API Key ---
API_KEY_NEWSAPI = "eff4370ce4f9491ca58fafe81567324b"

# --- Sidebar ---
st.sidebar.title("📊 ML Trading Cockpit")
ticker = st.sidebar.text_input("📈 Ticker-Symbol")
company_name = None

if ticker:
    try:
        ticker_obj = yf.Ticker(ticker)
        company_name = ticker_obj.info.get('longName', 'Name not found')
        st.sidebar.markdown(f"**Company Name:** {company_name}")
    except Exception as e:
        st.sidebar.error(f"Error fetching company name: {e}")

# --- News Source Selection ---
st.sidebar.markdown("### 📰 News Sources")
use_yahoo_news = st.sidebar.checkbox("✅ Yahoo Finance", value=True)
use_google_news = st.sidebar.checkbox("✅ Google News RSS", value=True)
use_investing_rss = st.sidebar.checkbox("✅ Investing.com RSS", value=True)
use_google_trends = st.sidebar.checkbox("📈 Google Trends", value=True)
use_sentiment_in_optuna = st.sidebar.checkbox("🧠 Use Sentiment in Optuna", value=False)

# --- Advanced Settings ---
with st.sidebar.expander("⚙️ Advanced Trading Settings", expanded=False):
    trading_capital = st.number_input("💰 Trading Capital", min_value=100.0, value=10000.0, step=1000.0)
    col1, col2 = st.columns(2)
    allow_long = col1.checkbox("✅ Long", value=True)
    allow_short = col2.checkbox("✅ Short", value=True)
    trailing_stop_loss_pct = st.number_input("🚨 Trailing Stop Loss (%)", 0.1, 10.0, 2.5, 0.1) / 100
    use_take_profit = st.checkbox("Enable Take Profit", value=False)
    take_profit_pct = st.number_input("🎯 Take Profit (%)", 0.1, 50.0, 5.0, 0.1) / 100
    transaction_cost_amount = st.number_input("💸 Transaction Cost per Trade", 0.0, value=4.9, step=0.1)
    tax_rate_pct = st.number_input("📊 Tax Rate (%) on Gains", 0.0, 50.0, 30.0, step=0.1) / 100

train_default = st.sidebar.slider("📅 Training Period (Years)", 1, 10, 3)
test_default_months = st.sidebar.slider("📅 Test Period (Months)", 1, 24, 6)
n_trials = st.sidebar.slider("🔁 Optuna Trials", 1, 200, 100)

start = st.sidebar.button("🚀 Start Analysis")

# --- Sentiment Functions ---

def analyze_article_sentiment(article):
    text = f"{article.get('title', '')} {article.get('description', article.get('summary', ''))}".strip()
    if not text:
        return None
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]

def classify_sentiment(score):
    if score is None:
        return "unknown"
    elif score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def show_sentiment_chart(articles):
    data = []
    for article in articles:
        published_at = article.get("publishedAt") or article.get("published") or ""
        try:
            dt = pd.to_datetime(published_at, errors="coerce")
            if dt is pd.NaT:
                continue
            dt = dt.tz_localize(None) if dt.tzinfo else dt
            date = dt.normalize()
        except Exception:
            continue
        score = analyze_article_sentiment(article)
        if score is not None:
            data.append((date, score))
    if not data:
        st.info("No valid sentiment data available.")
        return
    df = pd.DataFrame(data, columns=["date", "sentiment"])
    daily_avg = df.groupby("date")["sentiment"].mean().sort_index()
    smoothed = daily_avg.rolling(window=3, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_avg.index, daily_avg.values, marker='o', linestyle='-', alpha=0.5, label="Daily Average")
    ax.plot(smoothed.index, smoothed.values, color='orange', linestyle='-', label="Smoothed (Rolling 3)")
    ax.set_title("Sentiment Score Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment Score (-1 to +1)")
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()
    ax.legend()
    st.pyplot(fig)

# --- News Fetching Functions ---

def fetch_yahoo_news_rss(ticker):
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(url)
    return [{"title": e.title, "link": e.link, "published": e.get("published", ""), "summary": e.get("summary", "")} for e in feed.entries]

def fetch_google_news_rss(ticker, company_name):
    query = f"{ticker} OR {company_name}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [{"title": e.title, "link": e.link, "published": e.get("published", ""), "summary": e.get("summary", "")} for e in feed.entries]

def fetch_investing_rss(url):
    feed = feedparser.parse(url)
    return [{"title": e.title, "link": e.link, "published": e.get("published", ""), "summary": e.get("summary", "")} for e in feed.entries]

def get_google_trends(ticker):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq()
        pytrends.build_payload([ticker], timeframe='today 3-m')
        data = pytrends.interest_over_time()
        return data
    except Exception as e:
        st.warning(f"Google Trends error: {e}")
        return pd.DataFrame()

# --- Ticker-Daten laden ---

@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, period="max")
    if data.empty:
        raise ValueError(f"No data found for ticker '{symbol}'.")
    return data
# --- Main Analysis ---
if start and ticker:
    try:
        _ = load_data(ticker)
    except Exception as e:
        st.error(f"❌ Fehler beim Laden von Börsendaten: {e}")
        st.stop()

    st.markdown("## 📰 Markt- & Unternehmensnachrichten")
    st.markdown("---")

    # Unternehmensnews – Yahoo Finance
    if use_yahoo_news:
        with st.expander("📈 Yahoo Finance News", expanded=True):
            st.markdown("##### Letzte Artikel von Yahoo Finance")
            yahoo_news_articles = fetch_yahoo_news_rss(ticker)

            for art in yahoo_news_articles[:5]:
                score = analyze_article_sentiment(art)
                label = classify_sentiment(score)
                published_at = art.get("published") or ""
                try:
                    date_str = pd.to_datetime(published_at).strftime("%Y-%m-%d")
                except Exception:
                    date_str = "unbekannt"

                st.markdown(
                    f"- [{art['title']}]({art['link']})  \n"
                    f"🗓️ {date_str} | 🧠 Sentiment: **{label}** ({score:.2f})"
                )

            show_sentiment_chart(yahoo_news_articles)

    # Unternehmensnews – Google News
    if use_google_news:
        with st.expander("🔍 Google News", expanded=False):
            st.markdown("##### Letzte Artikel von Google News")
            google_news_articles = fetch_google_news_rss(ticker, company_name)

            for art in google_news_articles[:5]:
                score = analyze_article_sentiment(art)
                label = classify_sentiment(score)
                published_at = art.get("published") or ""
                try:
                    date_str = pd.to_datetime(published_at).strftime("%Y-%m-%d")
                except Exception:
                    date_str = "unbekannt"

                st.markdown(
                    f"- [{art['title']}]({art['link']})  \n"
                    f"🗓️ {date_str} | 🧠 Sentiment: **{label}** ({score:.2f})"
                )

            show_sentiment_chart(google_news_articles)

    # Makronews – Investing.com
    if use_investing_rss:
        st.markdown("---")
        st.markdown("## 🌍 Makroökonomische Nachrichten")
        with st.expander("📰 Investing.com RSS", expanded=False):
            investing_rss_urls = {
                "📊 Wirtschaft": "https://www.investing.com/rss/news_285.rss",
                "🏦 Zentralbanken": "https://www.investing.com/rss/news_25.rss",
                "📈 Konjunkturindikatoren": "https://www.investing.com/rss/news_95.rss"
            }

            for section_title, url in investing_rss_urls.items():
                st.markdown(f"#### {section_title}")
                invest_articles = fetch_investing_rss(url)

                for art in invest_articles[:5]:
                    score = analyze_article_sentiment(art)
                    label_sent = classify_sentiment(score)
                    published_at = art.get("published") or ""
                    try:
                        date_str = pd.to_datetime(published_at).strftime("%Y-%m-%d")
                    except Exception:
                        date_str = "unbekannt"

                    st.markdown(
                        f"- [{art['title']}]({art['link']})  \n"
                        f"🗓️ {date_str} | 🧠 Sentiment: **{label_sent}** ({score:.2f})"
                    )

                show_sentiment_chart(invest_articles)

    # Google Trends
    if use_google_trends:
        st.markdown("---")
        st.markdown("## 🔎 Google Trends Analyse")
        with st.expander("📈 Google Trends", expanded=False):
            trends_df = get_google_trends(ticker)
            if not trends_df.empty:
                st.line_chart(trends_df[ticker])
            else:
                st.info("ℹ️ Keine Google Trends Daten verfügbar.")

# --- Handelslogik (performance-optimiert) ---

def simulate_trading(signals, prices, trading_capital, allow_long=True, allow_short=True,
                     trailing_stop_loss_pct=0.05, take_profit_pct=0.1,
                     transaction_cost_pct=0.001, use_take_profit=True):

    print(f"[DEBUG] allow_long={allow_long}, allow_short={allow_short}")
    print(f"[DEBUG] Signals columns: {signals.columns.tolist()}")
    print(f"[DEBUG] Signals head:\n{signals.head()}")
    print(f"[DEBUG] Prices head:\n{prices.head()}")

    # Index zu datetime konvertieren
    if not isinstance(signals.index, pd.DatetimeIndex):
        signals.index = pd.to_datetime(signals.index, errors='coerce')
        print("[INFO] Signals-Index zu DatetimeIndex konvertiert")
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors='coerce')
        print("[INFO] Preise-Index zu DatetimeIndex konvertiert")

    available_capital = trading_capital
    trades = []
    open_position = None  # Statt DataFrame: nur ein dict (max. 1 Position zur Zeit)

    for date, signal in signals.iterrows():
        if date not in prices.index:
            print(f"[WARNUNG] Kein Preis für {date}, überspringe.")
            continue

        price = prices.loc[date]
        buy_signal = bool(signal.get('Buy', False))
        sell_signal = bool(signal.get('Sell', False))

        # --- Falls eine Position offen ist ---
        if open_position is not None:
            closed = False

            if open_position['type'] == 'Long':
                open_position['trailing_high'] = max(open_position['trailing_high'], price)

                # Take Profit
                if use_take_profit and price >= open_position['entry_price'] * (1 + take_profit_pct):
                    pnl = (price - open_position['entry_price']) * open_position['capital'] / open_position['entry_price']
                    pnl -= 2 * transaction_cost_pct * open_position['capital']
                    available_capital += pnl
                    open_position.update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'closed_by': 'Take Profit',
                        'actual_stoploss_pct': None,
                        'trailing_low': None
                    })
                    trades.append(open_position.copy())
                    open_position = None
                    closed = True

                # Trailing Stop Loss
                if not closed and price <= open_position['trailing_high'] * (1 - trailing_stop_loss_pct):
                    pnl = (price - open_position['entry_price']) * open_position['capital'] / open_position['entry_price']
                    pnl -= 2 * transaction_cost_pct * open_position['capital']
                    available_capital += pnl
                    open_position.update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'closed_by': 'Trailing Stop Loss',
                        'actual_stoploss_pct': abs(price - open_position['entry_price']) / open_position['entry_price'] * 100,
                        'trailing_low': None
                    })
                    trades.append(open_position.copy())
                    open_position = None
                    closed = True

                # Signalwechsel
                if not closed and sell_signal and allow_short:
                    pnl = (price - open_position['entry_price']) * open_position['capital'] / open_position['entry_price']
                    pnl -= 2 * transaction_cost_pct * open_position['capital']
                    available_capital += pnl
                    open_position.update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'closed_by': 'Signalwechsel (Sell bei Long)',
                        'actual_stoploss_pct': abs(price - open_position['entry_price']) / open_position['entry_price'] * 100,
                        'trailing_low': None
                    })
                    trades.append(open_position.copy())
                    open_position = None
                    closed = True

            elif open_position['type'] == 'Short':
                open_position['trailing_low'] = min(open_position['trailing_low'], price)

                # Take Profit
                if use_take_profit and price <= open_position['entry_price'] * (1 - take_profit_pct):
                    pnl = (open_position['entry_price'] - price) * open_position['capital'] / open_position['entry_price']
                    pnl -= 2 * transaction_cost_pct * open_position['capital']
                    available_capital += pnl
                    open_position.update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'closed_by': 'Take Profit',
                        'actual_stoploss_pct': None,
                        'trailing_high': None
                    })
                    trades.append(open_position.copy())
                    open_position = None
                    closed = True

                # Trailing Stop Loss
                if not closed and price >= open_position['trailing_low'] * (1 + trailing_stop_loss_pct):
                    pnl = (open_position['entry_price'] - price) * open_position['capital'] / open_position['entry_price']
                    pnl -= 2 * transaction_cost_pct * open_position['capital']
                    available_capital += pnl
                    open_position.update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'closed_by': 'Trailing Stop Loss',
                        'actual_stoploss_pct': abs(price - open_position['entry_price']) / open_position['entry_price'] * 100,
                        'trailing_high': None
                    })
                    trades.append(open_position.copy())
                    open_position = None
                    closed = True

                # Signalwechsel
                if not closed and buy_signal and allow_long:
                    pnl = (open_position['entry_price'] - price) * open_position['capital'] / open_position['entry_price']
                    pnl -= 2 * transaction_cost_pct * open_position['capital']
                    available_capital += pnl
                    open_position.update({
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'closed_by': 'Signalwechsel (Buy bei Short)',
                        'actual_stoploss_pct': abs(price - open_position['entry_price']) / open_position['entry_price'] * 100,
                        'trailing_high': None
                    })
                    trades.append(open_position.copy())
                    open_position = None
                    closed = True

        # --- Falls keine Position offen ist, öffne neue ---
        if open_position is None:
            if buy_signal and allow_long:
                print(f"[INFO] Long Entry am {date} bei {price}")
                open_position = {
                    'type': 'Long',
                    'entry_date': date,
                    'entry_price': price,
                    'trailing_high': price,
                    'trailing_low': None,
                    'capital': available_capital
                }
            elif sell_signal and allow_short:
                print(f"[INFO] Short Entry am {date} bei {price}")
                open_position = {
                    'type': 'Short',
                    'entry_date': date,
                    'entry_price': price,
                    'trailing_high': None,
                    'trailing_low': price,
                    'capital': available_capital
                }

    # Falls Position am Ende offen bleibt
    if open_position is not None:
        open_position.update({
            'exit_date': None,
            'exit_price': None,
            'pnl': None,
            'closed_by': 'Still Open',
            'actual_stoploss_pct': None
        })
        trades.append(open_position.copy())

    if not trades:
        print("[INFO] Keine Trades durchgeführt. Rückgabe eines leeren DataFrames mit Spalten.")
        return pd.DataFrame(columns=[
            'type', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
            'pnl', 'closed_by', 'actual_stoploss_pct', 'trailing_high', 'trailing_low', 'capital'
        ])

    df = pd.DataFrame(trades)
    for col in ['entry_date', 'exit_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    print(f"[DEBUG] Anzahl Trades: {len(df)}")
    return df

# --- Technische Indikatoren ---

def SMA(series, period):
    return series.rolling(window=period).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def Bollinger_Bands(series, period=20, std_dev=2):
    sma = SMA(series, period)
    rolling_std = series.rolling(window=period).std()
    upper = sma + (rolling_std * std_dev)
    lower = sma - (rolling_std * std_dev)
    return upper, sma, lower

def MACD(data, fast=12, slow=26, signal=9):
    close_col = next((col for col in data.columns if 'close' in col.lower()), None)
    if close_col is None:
        raise KeyError("Keine Spalte mit 'Close' gefunden")
    
    close = data[close_col]
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def OBV(data):
    close_col = next((col for col in data.columns if 'close' in col.lower()), None)
    volume_col = next((col for col in data.columns if 'volume' in col.lower()), None)
    
    if close_col is None:
        raise KeyError("Keine Spalte mit 'Close' gefunden")
    if volume_col is None:
        raise KeyError("Keine Spalte mit 'Volume' gefunden")
    
    close = data[close_col]
    volume = data[volume_col]

    obv = np.where(close > close.shift(1), volume,
                   np.where(close < close.shift(1), -volume, 0))
    obv = obv.flatten()
    return pd.Series(obv, index=data.index).cumsum()

def get_column(data, keyword):
    col = next((c for c in data.columns if keyword.lower() in c.lower()), None)
    if col is None:
        raise KeyError(f"Keine Spalte mit '{keyword}' gefunden.")
    return col

def Stochastic_Oscillator(data, k_period=14, d_period=3):
    low = data[get_column(data, 'low')]
    high = data[get_column(data, 'high')]
    close = data[get_column(data, 'close')]

    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    k = 100 * ((close - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def ATR(data, period=14):
    high = data[get_column(data, 'high')]
    low = data[get_column(data, 'low')]
    close = data[get_column(data, 'close')]

    high_low = high - low
    high_close_prev = (high - close.shift(1)).abs()
    low_close_prev = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def VWAP(data):
    close = data[get_column(data, 'close')]
    volume = data[get_column(data, 'volume')]

    cum_vol = volume.cumsum()
    cum_vol_price = (close * volume).cumsum()
    return cum_vol_price / cum_vol

def Ichimoku(data, period1=9, period2=26, period3=52):
    high = data[get_column(data, 'high')]
    low = data[get_column(data, 'low')]
    close = data[get_column(data, 'close')]

    high_9 = high.rolling(window=period1).max()
    low_9 = low.rolling(window=period1).min()
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = high.rolling(window=period2).max()
    low_26 = low.rolling(window=period2).min()
    kijun_sen = (high_26 + low_26) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period2)

    high_52 = high.rolling(window=period3).max()
    low_52 = low.rolling(window=period3).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(period2)

    chikou_span = close.shift(-period2)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def ADX(data, period=14):
    high = data[get_column(data, 'high')]
    low = data[get_column(data, 'low')]
    close = data[get_column(data, 'close')]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    tr = ATR(data, period)

    plus_dm_1d = np.ravel(plus_dm)
    minus_dm_1d = np.ravel(minus_dm)

    plus_di = 100 * (pd.Series(plus_dm_1d, index=data.index).rolling(window=period).sum() / tr)
    minus_di = 100 * (pd.Series(minus_dm_1d, index=data.index).rolling(window=period).sum() / tr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di

def CMO(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).sum()
    cmo = 100 * (gain - loss) / (gain + loss)
    return cmo

def Keltner_Channel(data, period=20, atr_period=14, multiplier=2):
    high = data[get_column(data, 'high')]
    low = data[get_column(data, 'low')]
    close = data[get_column(data, 'close')]

    typical_price = (high + low + close) / 3
    ema = typical_price.ewm(span=period, adjust=False).mean()
    atr = ATR(data, atr_period)

    upper = ema + multiplier * atr
    lower = ema - multiplier * atr

    if isinstance(upper, pd.DataFrame):
        upper = upper.iloc[:, 0]
    if isinstance(ema, pd.DataFrame):
        ema = ema.iloc[:, 0]
    if isinstance(lower, pd.DataFrame):
        lower = lower.iloc[:, 0]
    return upper, ema, lower

def add_volume_spike(data, window=20, threshold=2.0, rvol_thresh=2.0, price_col='Close'):
    # Alias für threshold
    z_thresh = threshold

    # Dynamisch Volume-Spalte finden
    volume_cols = [col for col in data.columns if 'volume' in col.lower()]
    if not volume_cols:
        raise ValueError("Keine Volume-Spalte gefunden.")
    volume_col = volume_cols[0]

    # Rolling-Statistiken berechnen
    rolling_mean = data[volume_col].rolling(window=window).mean()
    rolling_std = data[volume_col].rolling(window=window).std()
    rolling_median = data[volume_col].rolling(window=window).median()

    # Z-Score berechnen (wie stark weicht das aktuelle Volumen vom Mittelwert ab)
    z_score = (data[volume_col] - rolling_mean) / rolling_std

    # Relative Volumen Kennzahl (Volumen im Vergleich zum Durchschnitt)
    rvol = data[volume_col] / rolling_mean

    # Preisfilter: z.B. Volumenspike nur, wenn Close über SMA20 (typisch für bullische Bestätigung)
    sma20 = SMA(data[price_col], 20)

    # Volumenanstieg zum Vortag (mindestens 20% höher)
    vol_increase = data[volume_col] > (1.2 * data[volume_col].shift(1))

    # Volumenspike nach oben
    volume_spike_up = (
        (z_score > z_thresh) &
        (rvol > rvol_thresh) &
        (data[price_col] > sma20) &
        vol_increase &
        (data[volume_col] > rolling_median)
    )

    # Volumenspike nach unten
    volume_spike_down = (
        (z_score < -z_thresh) &
        (rvol > rvol_thresh) &
        (data[price_col] < sma20) &
        vol_increase &
        (data[volume_col] > rolling_median)
    )

    # Kombinierte Spike-Spalte (1 wenn up oder down)
    volume_spike = volume_spike_up | volume_spike_down

    # Ergebnis in DataFrame schreiben
    data['Volume_Spike_Up'] = volume_spike_up.astype(int)
    data['Volume_Spike_Down'] = volume_spike_down.astype(int)
    data['Volume_Spike'] = volume_spike.astype(int)
    data['Volume_MA'] = rolling_mean
    data['Volume_STD'] = rolling_std
    data['Volume_Z'] = z_score
    data['RVOL'] = rvol
    data['Volume_Median'] = rolling_median

    return data

def add_steep_movement_flag(data, close_col='Close', horizon=1, std_window=20, multiplier=1.5):
    """
    Kennzeichnet steile Bewegungen, wenn die Rendite größer ist als ein Vielfaches der rolling Volatilität.
    """
    returns = data[close_col].pct_change(horizon)
    rolling_volatility = returns.rolling(std_window).std()

    # Steep, wenn absolute Rendite > multiplier * rolling std
    flags = (returns.abs() > (rolling_volatility * multiplier)).astype(int)
    data['SteepMoveFlag'] = flags

    return data

def add_loss_weights(data, steep_flag_col='SteepMoveFlag', base_weight=1, steep_weight=5):
    data['loss_weight'] = base_weight
    data.loc[data[steep_flag_col] == 1, 'loss_weight'] = steep_weight
    return data

def add_steep_movement_target(data, close_col='Close', horizon=1, std_window=20, multiplier=1.5):
    returns = data[close_col].pct_change(horizon).shift(-horizon)  # zukünftige Rendite
    rolling_volatility = returns.rolling(std_window).std()
    # Steile Bewegung in der Zukunft (1 wenn ja, sonst 0)
    data['target_steep_move'] = (returns.abs() > (rolling_volatility * multiplier)).astype(int)
    return data

def ROC(series, period=12):
    return series.pct_change(period)

def add_steep_movement_flags(data, close_col='Close', horizons=[1,2,3], rolling_windows=[5,10,20], multipliers=[1.0, 1.2, 1.5]):
    """
    Fügt im DataFrame Flags für steile Bewegungen hinzu.
    - horizons: Liste von Tagen für die Rendite-Berechnung (z.B. 1-3 Tage)
    - rolling_windows: Liste der Rolling-Std-Fenster (z.B. 5, 10, 20 Tage)
    - multipliers: Multiplikatoren für die Schwellenwerte

    Ergebnis: Für jede Kombination wird eine Spalte angelegt,
    z.B. 'steep_move_h1_w5_m1.2' mit 1 oder 0.
    """
    for h in horizons:
        returns = data[close_col].pct_change(h)
        for w in rolling_windows:
            rolling_std = returns.rolling(w).std()
            for m in multipliers:
                threshold = rolling_std * m
                flag_col = f'steep_move_h{h}_w{w}_m{m}'
                data[flag_col] = (returns.abs() > threshold).astype(int)
    return data

# --- Marktphase erkennen ---

def detect_market_phase(data):
    """
    Ermittelt die Marktlage basierend auf ADX, Trend und Volatilität.
    Rückgabe: DataFrame mit Marktphasen ("Bullenmarkt", "Bärenmarkt", "Seitwärts", "Volatilität") pro Zeitindex
    """
    adx, plus_di, minus_di = ADX(data)
    vol = data['Close'].pct_change().rolling(window=20).std()

    vol_threshold = vol.quantile(0.75)

    phases = []

    for i in range(len(data)):
        trend_strength = adx.iloc[i]
        trend_dir = plus_di.iloc[i] > minus_di.iloc[i]
        volatility = vol.iloc[i]

        if pd.isna(trend_strength) or pd.isna(volatility):
            phases.append(np.nan)  # Unklare Phase, z.B. Anfangswerte
            continue

        if trend_strength > 25 and trend_dir:
            phases.append("Bullenmarkt")
        elif trend_strength > 25 and not trend_dir:
            phases.append("Bärenmarkt")
        elif volatility > vol_threshold:
            phases.append("Volatilität")
        else:
            phases.append("Seitwärts")

    return pd.Series(phases, index=data.index)

def get_risk_free_rate():
    try:
        ticker = yf.Ticker("^TNX")  # US 10Y Treasury Yield (in Prozent * 100)
        data = ticker.history(period="1d")
        if not data.empty:
            latest_yield = data['Close'][-1] / 100  # Prozent in Dezimal
            return latest_yield
    except Exception as e:
        print(f"Fehler beim Abruf der Risk-Free Rate: {e}")
    return 0.02  # Fallback 2%

def calculate_sharpe_from_equity_curve(trades, trading_capital, risk_free_rate=0.02):
    """
    Berechnet Sharpe Ratio basierend auf täglicher Equity Curve.

    trades: DataFrame mit Trades inklusive 'entry_date', 'exit_date', 'pnl', 'capital'
    trading_capital: Startkapital für Normalisierung der Equity Curve
    risk_free_rate: annualisierte risikofreie Rendite (z.B. 0.02 für 2%)
    """
    if trades.empty:
        return 0

    # Sortiere Trades nach Exit-Date (Ende des Trades)
    trades = trades.sort_values(by='exit_date').copy()
    trades['return'] = trades['pnl'] / trading_capital  # Return relativ zum Startkapital

    # Erstelle DataFrame für Equity Curve mit Tagesindex
    start_date = trades['entry_date'].min()
    end_date = trades['exit_date'].max()
    all_days = pd.date_range(start_date, end_date)

    equity_curve = pd.DataFrame(index=all_days)
    equity_curve['daily_return'] = 0.0

    # Für jeden Trade addiere den Return am Exit-Tag
    for idx, trade in trades.iterrows():
        exit_day = trade['exit_date']
        if exit_day in equity_curve.index:
            equity_curve.at[exit_day, 'daily_return'] += trade['return']

    # Kumulierte Equity berechnen
    equity_curve['cumulative_return'] = (1 + equity_curve['daily_return']).cumprod()

    # Umwandeln in tägliche Renditen
    equity_curve['daily_pct_change'] = equity_curve['cumulative_return'].pct_change().fillna(0)

    # Annualisierung: 252 Börsentage
    excess_returns = equity_curve['daily_pct_change'] - (risk_free_rate / 252)
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()

    if std_excess_return == 0:
        return 0

    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
    return sharpe_ratio

# --- Hybride Signal-Generierung ---

def hybrid_signal_generator(data):
    signals = pd.DataFrame(index=data.index)
    signals['Buy'] = False
    signals['Sell'] = False

    # Phasen-Detection (du musst deine detect_market_phase Funktion haben)
    phases = detect_market_phase(data)  # Beispiel: gibt Serie mit Phasen zurück
    signals['Market_Phase'] = phases

    phase_map = {"Bullenmarkt": 0, "Bärenmarkt": 1, "Seitwärts": 2, "Volatilität": 3}
    signals['Market_Phase_Num'] = signals['Market_Phase'].map(phase_map)

    # Indikatoren vorbereiten (hier als Beispiel)
    rsi = RSI(data['Close']).fillna(method='bfill')
    macd, macd_signal = MACD(data)
    macd = macd.fillna(method='bfill')
    macd_signal = macd_signal.fillna(method='bfill')
    upper_bb, middle_bb, lower_bb = Bollinger_Bands(data['Close'])
    upper_bb = upper_bb.fillna(method='bfill')
    lower_bb = lower_bb.fillna(method='bfill')
    adx, plus_di, minus_di = ADX(data)
    adx = adx.fillna(method='bfill')
    plus_di = plus_di.fillna(method='bfill')
    minus_di = minus_di.fillna(method='bfill')
    k, d = Stochastic_Oscillator(data)
    k = k.fillna(method='bfill')
    d = d.fillna(method='bfill')
    atr = ATR(data).fillna(method='bfill')
    vwap = VWAP(data).fillna(method='bfill')
    cmo = CMO(data['Close']).fillna(method='bfill')
    tenkan, kijun, senkou_a, senkou_b, chikou = Ichimoku(data)
    tenkan = tenkan.fillna(method='bfill')
    kijun = kijun.fillna(method='bfill')
    senkou_a = senkou_a.fillna(method='bfill')
    senkou_b = senkou_b.fillna(method='bfill')
    chikou = chikou.fillna(method='bfill')
    kc_upper, kc_middle, kc_lower = Keltner_Channel(data)
    kc_upper = kc_upper.fillna(method='bfill')
    kc_middle = kc_middle.fillna(method='bfill')
    kc_lower = kc_lower.fillna(method='bfill')

    for i in range(len(data)):
        phase = phases.iloc[i]

        if pd.isna(phase):
            continue

        buy_signal = False
        sell_signal = False

        if phase == "Bullenmarkt":
            buy_signal = (macd.iloc[i] > macd_signal.iloc[i]) and (rsi.iloc[i] < 70) and \
                         (data['Close'].iloc[i] > vwap.iloc[i]) and (plus_di.iloc[i] > minus_di.iloc[i]) and \
                         (adx.iloc[i] > 20) and (data['Volume_Spike'].iloc[i] == 1)

            sell_signal = (macd.iloc[i] < macd_signal.iloc[i]) and (rsi.iloc[i] > 70)

        elif phase == "Bärenmarkt":
            buy_signal = (rsi.iloc[i] < 40) and (k.iloc[i] < d.iloc[i])
            sell_signal = (macd.iloc[i] < macd_signal.iloc[i]) and (data['Close'].iloc[i] < vwap.iloc[i]) and \
                          (plus_di.iloc[i] < minus_di.iloc[i]) and (adx.iloc[i] > 20)

        elif phase == "Seitwärts":
            buy_signal = data['Close'].iloc[i] < lower_bb.iloc[i]
            sell_signal = data['Close'].iloc[i] > upper_bb.iloc[i]

        else:  # Volatilität
            buy_signal = (data['Volume_Spike'].iloc[i] == 1) and \
                         (data['Close'].iloc[i] > data['Open'].iloc[i]) and (data['Close'].iloc[i] > kc_middle.iloc[i])

            sell_signal = (data['Volume_Spike'].iloc[i] == 1) and \
                          (data['Close'].iloc[i] < data['Open'].iloc[i]) and (data['Close'].iloc[i] < kc_middle.iloc[i])

        signals.at[data.index[i], 'Buy'] = buy_signal
        signals.at[data.index[i], 'Sell'] = sell_signal

    return signals

# ----------------------- DATEN LADEN -------------------------

@st.cache_data(ttl=3600)
def load_data(symbol):
    data = yf.download(symbol, period="max")
    data.index = data.index.normalize()
    return data

# ----------------------- FEATURES HINZUFÜGEN -------------------------

def add_features(data):
    # Dynamisch die erste Close-Spalte finden
    close_cols = [col for col in data.columns if col.lower().startswith("close")]
    if not close_cols:
        raise ValueError("Keine Close-Spalte gefunden, die mit 'Close' beginnt.")
    close_col = close_cols[0]

    # Dynamisch die Volume-Spalte finden
    volume_cols = [col for col in data.columns if 'volume' in col.lower()]
    if not volume_cols:
        raise ValueError("Keine Volume-Spalte gefunden.")
    volume_col = volume_cols[0]

    window = 20
    threshold = 2

    # Berechnung technischer Indikatoren
    data['SMA_5'] = SMA(data[close_col], 5)
    data['SMA_20'] = SMA(data[close_col], 20)
    data['SMA_100'] = SMA(data[close_col], 100)
    data['SMA_200'] = SMA(data[close_col], 200)

    data['RSI_14'] = RSI(data[close_col], 14)

    bb_upper, bb_mid, bb_lower = Bollinger_Bands(data[close_col], 20, 2)
    data['BB_upper'] = bb_upper
    data['BB_mid'] = bb_mid
    data['BB_lower'] = bb_lower

    macd, macd_signal = MACD(data)
    data['MACD'] = macd
    data['MACD_signal'] = macd_signal

    data['OBV'] = OBV(data)

    k, d = Stochastic_Oscillator(data)
    data['Stochastic_K'] = k
    data['Stochastic_D'] = d

    data['ATR_14'] = ATR(data)
    data['VWAP'] = VWAP(data)

    tenkan, kijun, senkou_a, senkou_b, chikou = Ichimoku(data)
    data['Ichimoku_Tenkan'] = tenkan
    data['Ichimoku_Kijun'] = kijun
    data['Ichimoku_Senkou_A'] = senkou_a
    data['Ichimoku_Senkou_B'] = senkou_b
    data['Ichimoku_Chikou'] = chikou

    adx, plus_di, minus_di = ADX(data)
    data['ADX'] = adx
    data['Plus_DI'] = plus_di
    data['Minus_DI'] = minus_di

    data['CMO_14'] = CMO(data[close_col], 14)

    kc_upper, kc_middle, kc_lower = Keltner_Channel(data)
    data['KC_upper'] = kc_upper
    data['KC_middle'] = kc_middle
    data['KC_lower'] = kc_lower

    # Volume Spike hinzufügen
    data = add_volume_spike(data, threshold=2, price_col=close_col)

    if 'forecast' in data.columns:
        data['crossover_buy'] = ((data[close_col].shift(1) < data['forecast'].shift(1)) & 
                                 (data[close_col] > data['forecast'])).astype(int)
        data['crossover_sell'] = ((data[close_col].shift(1) > data['forecast'].shift(1)) & 
                                  (data[close_col] < data['forecast'])).astype(int)
    else:
        data['crossover_buy'] = 0
        data['crossover_sell'] = 0

    data = add_steep_movement_flag(data, close_col=close_col)

    data = add_loss_weights(data)

    data = add_steep_movement_target(data, close_col=close_col)

    data['SMA_5_20_cross'] = 0
    data.loc[data['SMA_5'] > data['SMA_20'], 'SMA_5_20_cross'] = 1
    data.loc[data['SMA_5'] < data['SMA_20'], 'SMA_5_20_cross'] = -1

    returns = data[close_col].pct_change()
    data['rolling_std_10'] = returns.rolling(window=10).std()

    data['ROC_12'] = ROC(data[close_col], 12)

    data['Momentum_10'] = data[close_col] - data[close_col].shift(10)

    return data

# ------------------ Funktion: Target für 1 Tag erstellen ------------------

def create_single_target(data, close_col='Close', horizon=1):
    data[f'Target_price'] = data[close_col].shift(-horizon)
    data.dropna(subset=[f'Target_price'], inplace=True)
    return data

# ------------------ Zusatzfeature: Steile Bewegungen erkennen ------------------

def add_steep_movement_flag(data, close_col='Close', horizon=1, thresh=0.02):
    future_return = (data[close_col].shift(-horizon) - data[close_col]) / data[close_col]
    data['SteepMoveFlag'] = 0
    data.loc[future_return > thresh, 'SteepMoveFlag'] = 1
    data.loc[future_return < -thresh, 'SteepMoveFlag'] = -1
    return data

# ------------------ Optuna-Studie (Single Output) ------------------

def optuna_study_singleoutput(data, train_default, test_default_months, features, trading_capital,
                             allow_long, allow_short,
                             transaction_cost_amount,
                             n_trials, use_sentiment_in_optuna=False):

    def objective(trial):
        test_offset = pd.DateOffset(months=test_default_months)
        test_start_date = data.index.max() - test_offset
        train_start_date = test_start_date - pd.DateOffset(years=train_default)

        data_train = data[(data.index >= train_start_date) & (data.index < test_start_date)]
        data_test = data[data.index >= test_start_date]

        if len(data_train) < 50 or len(data_test) < 10:
            raise optuna.exceptions.TrialPruned()

        optional_features = features.copy()
        if not use_sentiment_in_optuna:
            optional_features = [f for f in optional_features if "sentiment" not in f.lower()]

        selected_optional = [f for f in optional_features if trial.suggest_categorical(f"_use_{f}", [True, False])]
        if len(selected_optional) == 0:
            raise optuna.exceptions.TrialPruned()

        params = {
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
            'n_estimators': trial.suggest_int("n_estimators", 50, 500),
            'max_depth': trial.suggest_int("max_depth", 3, 15),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }

        X_train = data_train[selected_optional]
        y_train = data_train['Target_price']
        X_test = data_test[selected_optional]
        y_test = data_test['Target_price']

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        def custom_loss(preds, dtrain):
            y_true = dtrain.get_label()
            weight = np.abs(np.diff(np.insert(y_true, 0, y_true[0])))  # erste Differenz
            grad = 2 * weight * (preds - y_true)
            hess = 2 * weight
            return grad, hess

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            obj=custom_loss,
            verbose_eval=False
        )

        preds = model.predict(dtest)

        data_test_copy = data_test.copy()
        data_test_copy['forecast'] = preds

        signals = hybrid_signal_generator(data_test_copy)

        trades = simulate_trading(
            signals=signals,
            prices=data_test_copy['Close'],
            trading_capital=trading_capital,
            allow_long=allow_long,
            allow_short=allow_short,
            trailing_stop_loss_pct=trial.suggest_float("trailing_stop_loss_pct", 0.01, 0.1),
            take_profit_pct=trial.suggest_float("take_profit_pct", 0.01, 0.3),
            transaction_cost_pct=transaction_cost_amount / trading_capital,
            use_take_profit=trial.suggest_categorical("use_take_profit", [True, False])
        )

        sharpe = calculate_sharpe_from_equity_curve(trades, trading_capital)

        if np.isnan(sharpe) or np.isinf(sharpe):
            raise optuna.exceptions.TrialPruned()

        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study

# ------------------ MAIN BLOCK ------------------

if start:
    data = load_data(ticker)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in data.columns]

    data = add_features(data)

    close_col = next((col for col in data.columns if 'close' in col.lower()), None)
    if close_col is None:
        raise KeyError("Keine Spalte mit 'close' im Namen gefunden.")
    data.rename(columns={close_col: 'Close'}, inplace=True)

    volume_col = next((col for col in data.columns if 'volume' in col.lower()), None)
    if volume_col is None:
        raise KeyError("Keine Spalte mit 'volume' im Namen gefunden.")
    data.rename(columns={volume_col: 'Volume'}, inplace=True)

    # Target für 1 Tag berechnen
    data = create_single_target(data, close_col='Close', horizon=1)

    # Steile Bewegungen als Feature hinzufügen
    data = add_steep_movement_flag(data, close_col='Close', horizon=1, thresh=0.02)

    # Feature-Auswahl
    features = [col for col in data.columns if col not in ['Target_price', 'Date', 'timestamp']]
    sentiment_features = [f for f in features if "sentiment" in f.lower()]
    non_sentiment_features = [f for f in features if "sentiment" not in f.lower()]
    features_for_optuna = features if use_sentiment_in_optuna else non_sentiment_features

    # Optuna
    study = optuna_study_singleoutput(
        data,
        train_default=train_default,
        test_default_months=test_default_months,
        features=features_for_optuna,
        trading_capital=trading_capital,
        allow_long=allow_long,
        allow_short=allow_short,
        transaction_cost_amount=transaction_cost_amount,
        n_trials=n_trials,
        use_sentiment_in_optuna=use_sentiment_in_optuna
    )

    # Beste Parameter und neues Training
    best_params = study.best_params
    test_offset = pd.DateOffset(months=test_default_months)
    test_start_date = data.index.max() - test_offset
    train_start_date = test_start_date - pd.DateOffset(years=train_default)

    data_train = data[(data.index >= train_start_date) & (data.index < test_start_date)]
    data_test = data[data.index >= test_start_date]

    selected_features = [f for f in features_for_optuna if best_params.get(f"_use_{f}", False)]

    model = XGBRegressor(
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(data_train[selected_features], data_train['Target_price'])
    preds = model.predict(data_test[selected_features])

    # Ergebnisse
    mse = ((preds - data_test['Target_price']) ** 2).mean()
    pct_deviation = 100 * np.abs(preds.mean() - data_test['Target_price'].mean()) / data_test['Target_price'].mean()

    data_test = data_test.copy()
    data_test['forecast'] = preds

    # --- Funktion für Signalgenerierung mit KI-Filter ---
    def hybrid_signal_generator_with_ki_filter(data):
        signals = hybrid_signal_generator(data)  # klassische Signale

        # Prognose-Return: Forecast vs. Close
        forecast_return = (data['forecast'] - data['Close']) / data['Close']

        # Nur kaufen, wenn klassisches Kaufsignal und KI sagt Preis steigt
        signals['Buy'] = signals['Buy'] & (forecast_return > 0)

        # Nur verkaufen, wenn klassisches Verkaufssignal und KI sagt Preis fällt
        signals['Sell'] = signals['Sell'] & (forecast_return < 0)

        return signals

    # 1. Steile Bewegungen als True-Labels
    steep_true = data_test['SteepMoveFlag']

    # 2. Optional: Steile Bewegungen aus den Vorhersagen ableiten
    pred_returns = data_test['forecast'].pct_change().abs()
    steep_pred = (pred_returns > 0.02).astype(int)

    # 3. Evaluation der steilen Bewegungen
    precision = precision_score(steep_true, steep_pred, average='weighted')
    recall = recall_score(steep_true, steep_pred, average='weighted')
    f1 = f1_score(steep_true, steep_pred, average='weighted')

    # 4. Ausgabe der Metriken
    print(f"SteepMove Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    st.sidebar.write(f"SteepMove Precision: {precision:.4f}")
    st.sidebar.write(f"SteepMove Recall: {recall:.4f}")
    st.sidebar.write(f"SteepMove F1-Score: {f1:.4f}")

    # Nächster Tag Vorhersage
    future_date = data_test.index[-1] + pd.Timedelta(days=1)
    future_pred = preds[-1]
    future_df = pd.DataFrame({'Pred_Close': [future_pred]}, index=[future_date])

    signals = hybrid_signal_generator_with_ki_filter(data_test)
    transaction_cost_pct = transaction_cost_amount / trading_capital

    trades_df = simulate_trading(
        signals=signals,
        prices=data_test['Close'],
        trading_capital=trading_capital,
        allow_long=allow_long,
        allow_short=allow_short,
        trailing_stop_loss_pct=best_params['trailing_stop_loss_pct'],
        take_profit_pct=best_params['take_profit_pct'],
        transaction_cost_pct=transaction_cost_pct,
        use_take_profit=best_params['use_take_profit']
    )

    # --------------------- Ergebnisse ---------------------

    st.sidebar.subheader("📊 Ergebnis der Analyse")
    st.sidebar.write(f"MSE: {mse:.4f}")
    st.sidebar.write(f"Abweichung vom Mittelwert: {pct_deviation:.2f} %")

    preds_series = pd.Series(preds, index=data_test.index)
    forecast_series = future_df['Pred_Close']
    combined_preds = pd.concat([preds_series, forecast_series])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data_test.index, data_test['Target_price'], label="Tatsächlicher Kurs")
    ax.plot(combined_preds.index, combined_preds, label="Vorhersage", color="orange")
    
    if 'type' in trades_df.columns:
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['entry_price'] = pd.to_numeric(trades_df['entry_price'], errors='coerce')
        valid_trades = trades_df.dropna(subset=['entry_date', 'entry_price'])

        ax.plot(valid_trades.loc[valid_trades['type'] == 'Long', 'entry_date'],
                valid_trades.loc[valid_trades['type'] == 'Long', 'entry_price'], '^', color='green', markersize=10, label='Long')
        ax.plot(valid_trades.loc[valid_trades['type'] == 'Short', 'entry_date'],
                valid_trades.loc[valid_trades['type'] == 'Short', 'entry_price'], 'v', color='red', markersize=10, label='Short')

    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
# ------------------ Performance ------------------

if 'trades_df' in globals() and not trades_df.empty:
    st.subheader("💼 Trading-Performance")
    
    # Dauer des Trades berechnen
    trades_df['duration'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days
    
    # Prozentuale Rendite berechnen
    trades_df['return_pct'] = trades_df['pnl'] / trades_df['capital'] * 100
    
    # Stoploss als formatierten String
    trades_df['actual_stoploss_pct'] = trades_df['actual_stoploss_pct'].map(lambda x: f"{x:.2f} %" if pd.notnull(x) else "")
    
    # Übersicht der Trades anzeigen
    st.dataframe(trades_df[['type', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'duration', 'pnl', 'return_pct', 'actual_stoploss_pct']], use_container_width=True)

    # ------------------ Letzte 3 Handelssignale ------------------
    st.subheader("🔔 Letzte 3 Handelssignale")

    last_trades = trades_df.tail(3).copy()
    last_trades['Signal'] = last_trades['type'].map({'Long': 'Kauf', 'Short': 'Verkauf'})
    last_trades['Datum'] = last_trades['entry_date']

    st.table(last_trades[['Datum', 'Signal', 'entry_price', 'exit_price']])

    # ------------------ Performance-Metriken ------------------
    gross_total = trades_df['pnl'].sum()
    tax_total = gross_total * tax_rate_pct if gross_total > 0 else 0
    net_total = gross_total - tax_total

    st.metric("Bruttoergebnis", f"{gross_total:.2f} €")
    st.metric("Steuerlast", f"{tax_total:.2f} €")
    st.metric("Nettoergebnis", f"{net_total:.2f} €")

    gross_return_pct = (gross_total / trading_capital) * 100
    net_return_pct = (net_total / trading_capital) * 100
    st.metric("Bruttorendite", f"{gross_return_pct:.2f} %")
    st.metric("Nettorendite", f"{net_return_pct:.2f} %")
    st.metric("Gewinn-Trades", f"{(trades_df['pnl'] > 0).sum()} / {len(trades_df)}")

    avg_win = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean()
    avg_loss = abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean())
    crv = avg_win / avg_loss if avg_loss else float('inf')
    sharpe_ratio = calculate_sharpe_from_equity_curve(trades_df, trading_capital)

    st.metric("CRV", f"{crv:.2f}")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

else:
    st.info("Keine Trades durchgeführt.")
