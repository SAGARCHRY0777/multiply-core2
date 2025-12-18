import logging
import time as time_lib
from tradingview_websocket import TradingViewWebSocket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import random
import pandas as pd
import threading
from functions.data_utils import adjust_from_date, convert_record_02, create_quotes, to_ist_naive

INTERVAL_TIMEFRAME_MAP = {
    "ONE_MINUTE": "1",
    "FIVE_MINUTE": "5",
    "FIFTEEN_MINUTE": "15",
    "ONE_HOUR": "60",
    "ONE_DAY": "1D",
    "ONE_WEEK": "1W",
    "ONE_MONTH": "1M"
}
CONNECTION_SEMAPHORE = threading.Semaphore(5)


def fetch_candle_data_tradingview(symbol_name, interval_tf, lookback=360,
                                  current_time_ist=None, retry_count=0, max_retries=3):
    with CONNECTION_SEMAPHORE:
        try:
            if symbol_name == 'SENSEX':
                tv_symbol = f"BSE:{symbol_name}"
            else:
                tv_symbol = f"NSE:{symbol_name}"
            logging.debug(f"Fetching {lookback} candles for {tv_symbol} at {interval_tf} interval")
            ws = TradingViewWebSocket(tv_symbol, interval_tf, lookback)
            ws.connect()
            ws.run()
            raw_data = ws.result_data

            if not raw_data or not isinstance(raw_data, list) or len(raw_data) == 0:
                logging.warning(f"No valid data for {tv_symbol}, treating as unavailable.")
                return None

            candles = []
            for item in raw_data:
                try:
                    v = item.get("v")
                    if not v or len(v) < 6: continue
                    ts, o, h, l, c, vol = v[0], v[1], v[2], v[3], v[4], v[5]
                    dt_ist = datetime.fromtimestamp(ts, tz=ZoneInfo("Asia/Kolkata"))
                    candles.append([dt_ist.isoformat(), o, h, l, c, vol])
                except (ValueError, TypeError, IndexError, KeyError) as e:
                    logging.warning(f"Failed to parse candle for {symbol_name}: {e}")
                    continue

            if not candles: return None

            if current_time_ist and candles:
                cutoff_time = current_time_ist.replace(hour=15, minute=45, second=0, microsecond=0)
                if current_time_ist < cutoff_time:
                    candles.pop()
            return candles

        except Exception as e:
            # Handle retries for rate limiting or empty responses
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg or "Expecting value" in error_msg:
                if retry_count < max_retries:
                    delay = (2 ** retry_count) + random.uniform(0, 1)
                    logging.warning(f"Rate limit/JSON error for {symbol_name}. Retrying in {delay:.2f}s")
                    time_lib.sleep(delay)
                    return fetch_candle_data_tradingview(symbol_name, interval_tf, lookback, current_time_ist,
                                                         retry_count + 1)
                else:
                    logging.error(f"Max retries reached for {symbol_name} due to errors.")
                    return None
            else:
                logging.error(f"Generic error for {symbol_name}: {e}")
                return None


def process_symbol_pair_tradingview(args):
    print("process_symbol_pair_tradingview")
    symbol, interval, last_processed_dates, current_time_ist = args
    date_info = last_processed_dates.get(symbol)
    if not date_info: return None
    last_date = date_info["last_dt"]
    interval_tf = INTERVAL_TIMEFRAME_MAP.get(interval)
    if not interval_tf: return False
    candle_data = fetch_candle_data_tradingview(symbol, interval_tf, lookback=360, current_time_ist=current_time_ist)
    if not candle_data or len(candle_data) < 2: return None
    to_date = current_time_ist - timedelta(minutes={"ONE_MINUTE": 1, "FIVE_MINUTE": 5}.get(interval, 0))
    from_date = adjust_from_date(last_date, interval)

    candle_df = pd.DataFrame(candle_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    candle_df['date'] = pd.to_datetime(candle_df['date']).dt.tz_localize(None)
    quotes = create_quotes(pd.DataFrame([convert_record_02(c) for c in candle_df.values],
                                        columns=['datetime', 'open', 'high', 'low', 'close', 'volume']))
    indicator_data = (quotes, from_date, to_date, interval)

    to_date_naive = to_date.replace(tzinfo=None)
    if to_date_naive <= from_date: return None

    new_candles = [c for c in candle_data if
                   from_date <= datetime.fromisoformat(c[0]).replace(tzinfo=None) <= to_date_naive]
    bulk_data = [{"symbol": symbol, "interval": interval, "datetime": to_ist_naive(c[0]), "open": c[1], "high": c[2],
                  "low": c[3], "close": c[4], "volume": c[5]} for c in new_candles]

    return {'symbol': symbol, 'indicator_data': indicator_data, 'bulk_data': bulk_data}
