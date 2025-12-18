import pandas as pd
import math

# =============================================================================
# LEGACY LOGIC HELPERS
# =============================================================================

def identify_target_sell_point_bullish(buy_index, data, profit_threshold, y):
    """
    Legacy: Finds the index of the MAXIMUM close price in the window if it exceeds target.
    Optimistic: Exits at the peak.
    """
    buy_price = data.iloc[buy_index]['close']
    min_index = min(buy_index + y, len(data))
    if min_index <= buy_index:
        return None
        
    window = data.iloc[buy_index:min_index]
    if window.empty:
        return None

    max_price = window['close'].max()
    highest_point_index = window['close'].idxmax()
    
    profit = buy_price * profit_threshold
    
    if max_price >= buy_price + profit:
        return highest_point_index
    return None


def identify_stop_loss_sell_point_bullish(buy_index, data, threshold_price, y):
    """
    Legacy: Checks for CONSISTENT downward movement below threshold.
    Optimistic: Ignores stop loss if price recovers within window y.
    """
    max_idx = min(buy_index + y, len(data))
    
    for i in range(buy_index + 1, max_idx):
        close_price = data.iloc[i]['close']
        if close_price < threshold_price:
            consistent_downward = True
            # Look ahead y candles from THIS point (or until end of data)
            lookahead_end = min(i + y, len(data))
            
            for j in range(i, lookahead_end):
                if data.iloc[j]['close'] >= threshold_price:
                    consistent_downward = False
                    break
            
            if consistent_downward:
                # If consistent, backtrack to find local SMA peak? (Legacy logic)
                for j in range(i, buy_index, -1):
                    if j > 0 and 'sma' in data.columns:
                        # Legacy logic used SMA for precise exit point in downward trend
                        # If SMA isn't available (should be now), this might fail or skipping check
                        if data.iloc[j]['sma'] > data.iloc[j - 1]['sma']:
                            return j + 1
                # Fallback if loop finishes or SMA check passes/fails weirdly
                # The legacy code returns inside the loop. If it doesn't return, it continues searching.
                # If we are here, consistent_downward was True, but we didn't return.
                # Legacy code continues outer loop. 
    return None


def identify_target_sell_point_bearish(buy_index, data, profit_threshold, y):
    """
    Legacy: Finds the index of the MINIMUM close price in the window if it exceeds target.
    Optimistic: Exits at the bottom.
    """
    buy_price = data.iloc[buy_index]['close']
    min_index = min(buy_index + y, len(data))
    if min_index <= buy_index:
        return None

    window = data.iloc[buy_index:min_index]
    if window.empty:
        return None
        
    min_price = window['close'].min()
    lowest_point_index = window['close'].idxmin()
    
    profit = buy_price * profit_threshold
    
    if min_price <= buy_price - profit:
        return lowest_point_index
    return None


def identify_stop_loss_sell_point_bearish(buy_index, data, threshold_price, y):
    """
    Legacy: Checks for CONSISTENT upward movement above threshold.
    Optimistic: Ignores stop loss if price recovers within window y.
    """
    max_idx = min(buy_index + y, len(data))
    
    for i in range(buy_index + 1, max_idx):
        close_price = data.iloc[i]['close']
        if close_price > threshold_price:
            consistent_upward = True
            lookahead_end = min(i + y, len(data))
            
            for j in range(i, lookahead_end):
                if data.iloc[j]['close'] <= threshold_price:
                    consistent_upward = False
                    break
            
            if consistent_upward:
                for j in range(i, buy_index, -1):
                    if j > 0 and 'sma' in data.columns:
                        if data.iloc[j]['sma'] < data.iloc[j - 1]['sma']:
                            return j + 1
    return None


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_buy_index(args):
    """
    Process a buy signal and calculate trade outcome using LEGACY OPTIMISTIC LOGIC.
    Returns trade dict with from/to, profit, duration, etc.
    """
    # Unpack args (handling variable length for backward compatibility)
    if len(args) == 9:
        symbol, stock_df, interval, timeframe, buy_index, setup, threshold_price, profit_threshold_arg, total_candles_arg = args
    else:
        symbol, stock_df, interval, timeframe, buy_index, setup, threshold_price = args
        profit_threshold_arg = None
        total_candles_arg = None

    try:
        buy_row = stock_df.iloc[buy_index]
        buy_datetime = buy_row['datetime']
        buy_price = buy_row['close']

        # 1. Determine Parameters
        
        # Total Candles (y)
        if total_candles_arg is not None:
            total_candles = int(total_candles_arg)
        else:
            # Fallback defaults matches legacy
            total_candles = 104
            if timeframe in ['1M', '5M']:
                total_candles = 80
            elif timeframe == '15M':
                total_candles = 104
            elif timeframe == '1H':
                total_candles = 42
            elif timeframe == '1D':
                total_candles = 14
            elif timeframe == '1MO': # Added missing fallback
                total_candles = 150

        # Profit Threshold
        if profit_threshold_arg is not None:
            profit_threshold = float(profit_threshold_arg)
        else:
            profit_threshold = 0.02
            if 'indices' in setup.lower() or 'commodities' in setup.lower():
                profit_threshold = 0.002

        sell_index = None
        sell_point_type = None

        # 2. Execute Strategy
        if 'bullish' in setup:
            # Check Stop Loss first (Legacy priority?)
            # Legacy code checks Stop Loss first, then Target.
            # identify_stop_loss_sell_point_bullish return index if stopped out
            
            stop_loss_sell_point = identify_stop_loss_sell_point_bullish(
                buy_index, stock_df, threshold_price, total_candles
            )
            
            target_sell_point = identify_target_sell_point_bullish(
                buy_index, stock_df, profit_threshold, total_candles
            )

            # Decision Logic:
            # Legacy code:
            # if target_sell_point is not None: sell_index = target; sell_point_type = 'target'
            # elif stop_loss_sell_point is not None: sell_index = stop_loss; sell_point_type = 'stoploss'
            # else: timeout
            
            if target_sell_point is not None:
                sell_index = target_sell_point
                sell_point_type = 'target'
            elif stop_loss_sell_point is not None:
                sell_index = stop_loss_sell_point
                sell_point_type = 'stoploss'
            else:
                sell_index = min(buy_index + total_candles, len(stock_df) - 1)
                sell_point_type = 'stoploss' # Legacy treats timeout as stoploss usually

        else: # Bearish
            stop_loss_sell_point = identify_stop_loss_sell_point_bearish(
                buy_index, stock_df, threshold_price, total_candles
            )
            
            target_sell_point = identify_target_sell_point_bearish(
                buy_index, stock_df, profit_threshold, total_candles
            )

            if target_sell_point is not None:
                sell_index = target_sell_point
                sell_point_type = 'target'
            elif stop_loss_sell_point is not None:
                sell_index = stop_loss_sell_point
                sell_point_type = 'stoploss'
            else:
                sell_index = min(buy_index + total_candles, len(stock_df) - 1)
                sell_point_type = 'stoploss'

        # 3. Calculate Results
        sell_row = stock_df.iloc[sell_index]
        sell_datetime = sell_row['datetime']
        sell_price = sell_row['close'] # Always use close price in legacy

        # Calculate profit
        if 'bullish' in setup:
            profit_pct = ((sell_price - buy_price) / buy_price) * 100
        else:
            profit_pct = ((buy_price - sell_price) / buy_price) * 100

        # Calculate duration
        try:
            duration = (sell_datetime - buy_datetime).total_seconds()
        except:
            duration = 0

        return {
            'from': str(buy_datetime),
            'to': str(sell_datetime),
            'buy_index': int(buy_index),
            'sell_index': int(sell_index),
            'buy_price': float(buy_price),
            'sell_price': float(sell_price),
            'profit_percentage': float(profit_pct),
            'duration': float(duration),
            'setup': setup,
            'sell_point_type': sell_point_type,
            'symbol': symbol,
            'interval': interval
        }
    except Exception as e:
        print(f"Error in process_buy_index: {e}")
        return None
