"""Divergence detection.

Unified for both live trading and backtesting.
"""
import pandas as pd


def get_divergence(results):
    """Get last non-None divergence from results list."""
    if results:
        for date, divergence in reversed(results):
            if divergence != 'None':
                return date, divergence
    return None, None


def detect_divergence(df, timeframe):
    """
    Detect bullish/bearish divergences in price and RSI.
    Returns list of (date, divergence_type) tuples.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Determine check period based on timeframe
    if timeframe.endswith('M'):
        periods_per_day = 26 if timeframe == '15M' else (6 if timeframe == '1H' else 78)
        total_days = 4
    elif timeframe.endswith('H'):
        periods_per_day = 6
        total_days = 8
    elif timeframe.endswith('D'):
        periods_per_day = 1
        total_days = 14
    elif timeframe.endswith('W'):
        periods_per_day = 0.2
        total_days = 24
    elif timeframe.endswith('MO'):
        periods_per_day = 0.033
        total_days = 120
    else:
        return None, None
    
    total_candles = int(periods_per_day * total_days)
    
    if len(df) < total_candles:
        return None, None
    
    results = []
    check_df = df.iloc[-total_candles:]
    
    for i in range(1, len(check_df)):
        current_idx = check_df.index[i]
        current = check_df.iloc[i]
        current_date = current['datetime']
        current_price = current['low']
        current_rsi = current['rsi']
        
        # Look back for comparison
        lookback = min(i, 30)
        prev_df = check_df.iloc[max(0, i-lookback):i]
        
        if len(prev_df) < 2:
            results.append((current_date, 'None'))
            continue
        
        # Find previous low for bullish divergence
        prev_low_idx = prev_df['low'].idxmin()
        prev_low = prev_df.loc[prev_low_idx, 'low']
        prev_low_rsi = prev_df.loc[prev_low_idx, 'rsi']
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if current_price <= prev_low and current_rsi > prev_low_rsi:
            results.append((current_date, 'bullish divergence'))
            continue
        
        # Find previous high for bearish divergence
        current_high = current['high']
        prev_high_idx = prev_df['high'].idxmax()
        prev_high = prev_df.loc[prev_high_idx, 'high']
        prev_high_rsi = prev_df.loc[prev_high_idx, 'rsi']
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if current_high >= prev_high and current_rsi < prev_high_rsi:
            results.append((current_date, 'bearish divergence'))
            continue
        
        results.append((current_date, 'None'))
    
    return results, total_days
