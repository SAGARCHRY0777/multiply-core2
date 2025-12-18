"""Technical indicator condition checks.

Unified conditions for both live trading and backtesting.
- Live: Uses default index=-1 (last row)
- Backtesting: Passes specific index
"""


def check_most_recent_macd_crossing(df):
    """Check most recent MACD/signal line crossing."""
    for i in range(len(df) - 1, 0, -1):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        if curr['macd'] is None or curr['macd_signal'] is None:
            continue
        if prev['macd'] is None or prev['macd_signal'] is None:
            continue
        
        if curr['macd'] > curr['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            return "macd_over_signal"
        elif curr['macd'] < curr['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            return "signal_over_macd"
    
    return None


def directional_indicator_check(df, index=-1):
    """Check +DI vs -DI relationship."""
    row = df.iloc[index]
    if row['mdi'] < row['pdi']:
        return "plus_di_above_minus_di"
    elif row['mdi'] > row['pdi']:
        return "minus_di_above_plus_di"
    return None


def check_adx_greater_than_14(df, index=-1):
    """Check if ADX > 14."""
    if df.iloc[index]['adx'] > 14:
        return "adx_greater_than_14"
    return None


def check_adx_greater_than_25(df, index=-1):
    """Check if ADX > 25."""
    if df.iloc[index]['adx'] > 25:
        return "adx_greater_than_25"
    return None


def check_rsi(df, index=-1):
    """Check RSI level."""
    rsi = int(df.iloc[index]['rsi'])
    if rsi > 60:
        return "rsi_above_60"
    elif rsi < 40:
        return "rsi_below_40"
    return None


def rsi_trend_check(df):
    """Check RSI trend over last 3 candles."""
    last = df.iloc[-1]['rsi']
    prev = df.iloc[-2]['rsi']
    before = df.iloc[-3]['rsi']
    
    if last > prev > before:
        return "rsi_uptrend"
    elif last < prev < before:
        return "rsi_downtrend"
    return None


def price_50ema(df, ltp, index=-1):
    """Check price vs 50 EMA."""
    ema = df.iloc[index]['ema_50']
    if ltp > ema:
        return "price_above_50ema"
    elif ltp < ema:
        return "price_below_50ema"
    return None


def check_limit_price_break(crossing_index, df):
    """Check if price has broken crossing candle's high/low."""
    try:
        crossing = df.loc[crossing_index]
        last_close = df.iloc[-1]['close']
        
        if last_close > crossing['high']:
            return "price_above_limit_high"
        elif last_close < crossing['low']:
            return "price_below_limit_low"
    except:
        pass
    return None
