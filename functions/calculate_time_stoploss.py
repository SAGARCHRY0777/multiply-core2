from datetime import timedelta
import logging
from typing import Dict, Any


def calculate_time_stoploss(
    symbol: str,
    interval: str,
    buy_time,
    commodities_list: list = None,
    indices_list: list = None
) -> Dict[str, Any]:

    # Determine symbol type
    is_commodity_or_index = symbol in commodities_list or symbol in indices_list
    
    # Time stoploss rules
    if is_commodity_or_index:
        # Commodities & Indices rules
        duration_map = {
            'ONE_MINUTE': timedelta(hours=3),
            'FIVE_MINUTE': timedelta(hours=3),
            'FIFTEEN_MINUTE': timedelta(hours=6),
            'ONE_HOUR': timedelta(days=3),
            'ONE_DAY': timedelta(days=21),
            'ONE_WEEK': timedelta(days=90),
            'ONE_MONTH': timedelta(days=150)
        }
        symbol_type = 'commodity_index'
    else:
        # Equity rules
        duration_map = {
            'ONE_MINUTE': timedelta(hours=6),
            'FIVE_MINUTE': timedelta(hours=6),
            'FIFTEEN_MINUTE': timedelta(days=4),
            'ONE_HOUR': timedelta(days=7),
            'ONE_DAY': timedelta(days=21),
            'ONE_WEEK': timedelta(days=90),
            'ONE_MONTH': timedelta(days=150)
        }
        symbol_type = 'equity'
    
    # Get duration for this interval
    duration = duration_map.get(interval)
    
    if duration is None:
        # Fallback for unknown intervals
        logging.warning(f"Unknown interval '{interval}' for time stoploss calculation. Using default 6 hours.")
        duration = timedelta(hours=6)
    
    # Calculate time stoploss
    time_stoploss = buy_time + duration
    
    # Calculate duration in hours and days for convenience
    total_hours = duration.total_seconds() / 3600
    total_days = duration.days + (duration.seconds / 86400)
    
    result = {
        'time_stoploss': time_stoploss,
        'duration_hours': total_hours,
        'duration_days': total_days,
        'symbol_type': symbol_type,
        'interval': interval,
        'buy_time': buy_time,
        'symbol': symbol
    }
    
    logging.info(
        f"Time stoploss for {symbol} ({symbol_type}) at {interval}: "
        f"{time_stoploss.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({total_hours:.1f} hours / {total_days:.1f} days from entry)"
    )
    
    return result
