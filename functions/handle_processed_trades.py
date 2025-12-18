import os
import json
from datetime import datetime, timedelta

PROCESSED_LOG_FILE = 'processed_trades.json'

# Interval to timedelta mapping for candle duration
INTERVAL_DURATION = {
    'ONE_MINUTE': timedelta(minutes=1),
    'FIVE_MINUTE': timedelta(minutes=5),
    'FIFTEEN_MINUTE': timedelta(minutes=15),
    'ONE_HOUR': timedelta(hours=1),
    'ONE_DAY': timedelta(days=1),
    'ONE_WEEK': timedelta(weeks=1),
    'ONE_MONTH': timedelta(days=30)
}

# Number of candles that must pass before re-processing same symbol/interval/setup
CANDLE_COOLDOWN = 10

# Price difference (in points) above stored price to allow re-entry
# PRICE_DIFF_THRESHOLD = 20  # Deprecated in favor of 0.5% check


def load_processed_trades():
    """Load processed trades from the log file."""
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}


def save_processed_trades(processed_trades):
    """Save processed trades to the log file."""
    with open(PROCESSED_LOG_FILE, 'w') as file:
        json.dump(processed_trades, file, indent=2)


def _get_candles_passed(trade_datetime, current_datetime, interval):
    """
    Calculate how many candles have passed between two datetimes.

    Args:
        trade_datetime: datetime when trade was stored
        current_datetime: current datetime
        interval: interval string (e.g., 'FIVE_MINUTE')

    Returns:
        int: number of candles passed
    """
    duration = INTERVAL_DURATION.get(interval, timedelta(hours=1))
    
    # Normalize both datetimes to naive (remove timezone info)
    if hasattr(trade_datetime, 'tzinfo') and trade_datetime.tzinfo is not None:
        trade_datetime = trade_datetime.replace(tzinfo=None)
    if hasattr(current_datetime, 'tzinfo') and current_datetime.tzinfo is not None:
        current_datetime = current_datetime.replace(tzinfo=None)
    
    time_diff = current_datetime - trade_datetime

    # Calculate candles passed (floor division)
    candles = int(time_diff.total_seconds() / duration.total_seconds())
    return max(0, candles)


def is_trade_processed(symbol, interval, indicator, processed_trades, current_close_price=None, current_datetime=None):
    """
    Check if a trade should be skipped based on processed trades cache.

    A trade is considered "processed" (should skip) if:
    - Same symbol/interval/indicator exists in cache, AND
    - Less than 10 candles have passed since stored trade, AND
    - Current close price is NOT 20+ points above stored close price

    Args:
        symbol: The symbol name
        interval: The interval (e.g., 'FIFTEEN_MINUTE')
        indicator: The indicator type (e.g., 'bullish', 'bearish')
        processed_trades: Dictionary of processed trades
        current_close_price: Current analysis last close price (optional)
        current_datetime: Current datetime for comparison (optional, defaults to now)

    Returns:
        bool: True if trade should be skipped, False if trade should be processed
    """
    key = f"{symbol}_{interval}_{indicator}"

    if key not in processed_trades:
        return False  # Not in cache, allow processing

    trade_info = processed_trades[key]
    trade_datetime = datetime.strptime(trade_info['datetime'], '%Y-%m-%d %H:%M:%S')

    if current_datetime is None:
        current_datetime = datetime.now()

    # Check 1: Have 10 candles passed AND minimum 30 minutes elapsed?
    candles_passed = _get_candles_passed(trade_datetime, current_datetime, interval)
    time_elapsed = current_datetime - trade_datetime
    min_time_elapsed = time_elapsed >= timedelta(minutes=30)
    
    if candles_passed >= CANDLE_COOLDOWN and min_time_elapsed:
        return False  # Enough candles passed AND 30 mins elapsed, allow processing

    # Check 2: Is current price 0.5% above stored price?
    stored_price = trade_info.get('last_close_price')
    if current_close_price is not None and stored_price is not None:
        stored_price = float(stored_price)
        price_diff_percentage = (current_close_price - stored_price) / stored_price * 100
        if price_diff_percentage >= 0.5:
             return False  # Price moved 0.5%+, allow processing

    # Neither condition met, skip this trade
    return True


def mark_trade_as_processed(symbol, interval, indicator, processed_trades, last_close_price=None):
    """
    Mark a trade as processed with current datetime and close price.

    Args:
        symbol: The symbol name
        interval: The interval (e.g., 'FIFTEEN_MINUTE')
        indicator: The indicator type (e.g., 'bullish', 'bearish')
        processed_trades: Dictionary of processed trades
        last_close_price: The analysis last close price (optional but recommended)
    """
    key = f"{symbol}_{interval}_{indicator}"
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    trade_entry = {
        'symbol': symbol,
        'interval': interval,
        'indicator': indicator,
        'datetime': current_time
    }

    if last_close_price is not None:
        trade_entry['last_close_price'] = float(last_close_price)

    processed_trades[key] = trade_entry
    save_processed_trades(processed_trades)


def filter_expired_trades(processed_trades, current_time=None):
    """
    Filter out trades where 10+ candles have passed.

    Args:
        processed_trades: Dictionary of processed trades
        current_time: Optional datetime to use as reference (defaults to now)

    Returns:
        dict: Filtered dictionary with only non-expired trades
    """
    if current_time is None:
        current_time = datetime.now()

    filtered_trades = {}

    for key, trade_info in processed_trades.items():
        trade_datetime = datetime.strptime(trade_info['datetime'], '%Y-%m-%d %H:%M:%S')
        interval = trade_info.get('interval', 'ONE_HOUR')

        # Check if less than 10 candles have passed
        candles_passed = _get_candles_passed(trade_datetime, current_time, interval)

        if candles_passed < CANDLE_COOLDOWN:
            filtered_trades[key] = trade_info

    return filtered_trades


def clear_processed_trades():
    """Clear all processed trades."""
    with open(PROCESSED_LOG_FILE, 'w') as file:
        json.dump({}, file)


def should_process_trade(symbol, interval, indicator, processed_trades, current_close_price=None,
                         current_datetime=None):
    """
    Convenience function: returns True if trade SHOULD be processed, False if should skip.

    This is the inverse of is_trade_processed for clearer semantics.

    Args:
        symbol: The symbol name
        interval: The interval (e.g., 'FIFTEEN_MINUTE')
        indicator: The indicator type (e.g., 'bullish', 'bearish')
        processed_trades: Dictionary of processed trades
        current_close_price: Current analysis last close price
        current_datetime: Current datetime for comparison

    Returns:
        bool: True if trade should be processed, False if should be skipped
    """
    return not is_trade_processed(
        symbol, interval, indicator, processed_trades,
        current_close_price, current_datetime
    )


def check_and_register_trade(symbol, interval, setup_type, processed_trades, last_close_price=None,
                             current_datetime=None):
    """
    Check if trade should be processed, and if so, immediately register it.

    This combines checking and registration to prevent duplicate processing.
    Use 'bullish' or 'bearish' as indicator based on setup_type.

    Args:
        symbol: The symbol name
        interval: The interval (e.g., 'FIFTEEN_MINUTE')
        setup_type: The setup type (e.g., 'bullish_momentum_setup', 'bearish_momentum_setup')
        processed_trades: Dictionary of processed trades (will be modified)
        last_close_price: The analysis last close price for future comparison
        current_datetime: Current datetime for comparison

    Returns:
        bool: True if trade was registered (should process), False if already exists (should skip)
    """
    # Determine indicator type from setup_type
    if 'bullish' in setup_type.lower():
        indicator = 'bullish'
    elif 'bearish' in setup_type.lower():
        indicator = 'bearish'
    else:
        indicator = setup_type  # fallback

    # Check if trade already exists and is still valid
    if is_trade_processed(symbol, interval, indicator, processed_trades, last_close_price, current_datetime):
        return False  # Already processed, should skip

    # Not processed yet - register immediately to prevent duplicates
    mark_trade_as_processed(symbol, interval, indicator, processed_trades, last_close_price)
    return True  # Registered successfully, should process


def remove_trade_from_processed(symbol, interval, indicator, processed_trades):
    """
    Remove a specific trade from processed trades.

    Useful if trade processing fails and you want to allow retry.

    Args:
        symbol: The symbol name
        interval: The interval
        indicator: The indicator type ('bullish' or 'bearish')
        processed_trades: Dictionary of processed trades
    """
    key = f"{symbol}_{interval}_{indicator}"
    if key in processed_trades:
        del processed_trades[key]
        save_processed_trades(processed_trades)
