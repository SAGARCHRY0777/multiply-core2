import pandas as pd
import numpy as np


def timeframe_to_periods_per_day(timeframe):
    """Convert timeframe string to periods per day."""
    periods = {
        '1M': 375, '3M': 125, '5M': 75, '15M': 26, '30M': 13,
        '1H': 6, '2H': 3, '4H': 2, '1D': 1, '1W': 0.2, '1MO': 0.033
    }
    return periods.get(timeframe, 26)


def get_last_ema_crossing(crossings_list):
    """Get the last EMA crossing from list of (index, value) tuples."""
    for index, value in reversed(crossings_list):
        if value == 1.0:
            return "ema_crossing_above", index
        elif value == -1.0:
            return "ema_crossing_below", index
    return None, None


def calculate_ema_signal(stock_df, short_ema, long_ema, timeframe):
    """Calculate EMA signal crossings."""
    periods_per_day = timeframe_to_periods_per_day(timeframe)
    
    short_col = f'ema_{short_ema}'
    long_col = f'ema_{long_ema}'
    
    signal_df = pd.DataFrame()
    signal_df['datetime'] = stock_df['datetime']
    signal_df['Signal'] = np.where(stock_df[short_col] > stock_df[long_col], 1.0, 0.0)
    signal_df['Position'] = signal_df['Signal'].diff()
    
    return signal_df, periods_per_day


def check_ema_5_crossing_13(current_date, stock_df, timeframe, days):
    """Check for EMA 5 crossing EMA 13 after current_date."""
    signal_df, periods_per_day = calculate_ema_signal(stock_df, 5, 13, timeframe)
    x = int(periods_per_day * days)
    
    # Filter entries after current_date
    signal_df_filtered = signal_df[signal_df['datetime'] > current_date]
    
    # Check the last x entries for any crossing
    recent_crossings = signal_df_filtered['Position'].iloc[-x:]
    recent_crossings_list = [(index, value) for index, value in recent_crossings.items()]
    
    last_crossing_result, last_crossing_index = get_last_ema_crossing(recent_crossings_list)
    
    if last_crossing_result == "ema_crossing_above":
        return "ema_crossing_above_13", last_crossing_index
    elif last_crossing_result == "ema_crossing_below":
        return "ema_crossing_below_13", last_crossing_index
    return None, None


def check_ema_5_crossing_50(current_date, stock_df, timeframe, days):
    """Check for EMA 5 crossing EMA 50 after current_date."""
    signal_df, periods_per_day = calculate_ema_signal(stock_df, 5, 50, timeframe)
    x = int(periods_per_day * days)
    
    # Filter entries after current_date
    signal_df_filtered = signal_df[signal_df['datetime'] > current_date]
    
    # Check the last x entries for any crossing
    recent_crossings = signal_df_filtered['Position'].iloc[-x:]
    recent_crossings_list = [(index, value) for index, value in recent_crossings.items()]
    
    last_crossing_result, last_crossing_index = get_last_ema_crossing(recent_crossings_list)
    
    if last_crossing_result == "ema_crossing_above":
        return "ema_crossing_above_50", last_crossing_index
    elif last_crossing_result == "ema_crossing_below":
        return "ema_crossing_below_50", last_crossing_index
    return None, None
