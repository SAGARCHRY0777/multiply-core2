"""Volume condition checks.

Unified for both live trading and backtesting.
"""


def candle_with_good_volume(crossing_index, df):
    """Check if candle at crossing_index has good volume."""
    try:
        pos = df.index.get_loc(crossing_index)
        if pos < 20:
            return None, None
        
        candle = df.loc[crossing_index]
        prev_20 = df.iloc[pos-20:pos]
        avg_volume = prev_20['volume'].mean()
        
        is_bullish = candle['close'] > candle['open']
        is_bearish = candle['close'] < candle['open']
        good_volume = candle['volume'] > avg_volume
        
        if is_bullish and good_volume:
            return "bullish_candle_with_good_volume", crossing_index
        elif is_bearish and good_volume:
            return "bearish_candle_with_good_volume", crossing_index
    except:
        pass
    
    return None, None
