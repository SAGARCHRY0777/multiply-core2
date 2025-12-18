from functions.data_utils import adjust_from_date, get_current_datetime_ist, convert_record_02, to_ist_naive, create_quotes
from functions.user_login_cache import record_login_failure, record_login_success
from functions.SO_angel_v2.angel_one_functions import ohlc_history
from datetime import timedelta, datetime
import pandas as pd
import time

def get_ohlc_data_with_retry(symbol_dict, interval, f_date, to_date, initial_login_details,
                             login_details_list, max_retries=5, delay=2.5):
    """
    Retrieve OHLC data with retry mechanism in case of failures.
    """
    base_delay = float(delay)
    print(symbol_dict, interval, f_date, to_date, initial_login_details, login_details_list)

    for attempt in range(1, max_retries + 1):
        delay_seconds = float(attempt) * base_delay
        user_id = None
        try:
            if attempt == 1:
                login_details = initial_login_details
            else:
                if login_details_list:
                    login_details = login_details_list[attempt % len(login_details_list)]  # Rotate login details
                    print(attempt)
                    print(login_details)
                else:
                    login_details = initial_login_details

            if isinstance(login_details, dict):
                user_id = login_details.get('user_id')
            print(user_id)
            print(symbol_dict)
            data_dict = ohlc_history(
                symbol_dict['exch_seg'], symbol_dict["name"],
                symbol_dict["token"], interval, f_date, to_date, login_details
            )
            # Check if data is valid and return it
            if data_dict and data_dict.get('data'):
                record_login_success(user_id)
                return data_dict.get('data')
            elif data_dict and (data_dict.get('status') is True or data_dict.get('message') == 'SUCCESS'):
                record_login_success(user_id)
                return []
            else:
                record_login_failure(user_id)
        except Exception as e:
            print(f"Error retrieving data for {symbol_dict['name']} (Attempt {attempt}): {e}")
            record_login_failure(user_id)
        # Delay before the next retry
        if attempt < max_retries:
            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
    print(f"Failed to retrieve data after {max_retries} attempts for {symbol_dict['name']}.")
    return None

def process_symbol_pair_angelone(args):
    symbol_dict, user_dict, interval, last_processed_dates, send_interval, login_details_list = args
    symbol = symbol_dict["name"]
    date_info = last_processed_dates.get(symbol)
    if not date_info: return None
    last_date = date_info["last_dt"]
    start_date_from_db = date_info["start_date"]  # 360 rows before last date

    from_date = adjust_from_date(last_date, interval) if last_date else (
                get_current_datetime_ist() - timedelta(days=14)).replace(tzinfo=None)
    if from_date.tzinfo is not None:
        from_date = from_date.replace(tzinfo=None)
    to_date = get_current_datetime_ist() - timedelta(minutes={"ONE_MINUTE": 1, "FIVE_MINUTE": 5}.get(interval, 0))

    if to_date.replace(tzinfo=None) <= from_date: return None

    # For broker API, start_date is lookback, end_date is current time
    fetch_start_date = start_date_from_db if start_date_from_db else from_date
    candle_data = get_ohlc_data_with_retry(symbol_dict, send_interval, fetch_start_date, to_date, user_dict,
                                           login_details_list)
    if not candle_data: return False

    candle_df = pd.DataFrame(candle_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    candle_df['date'] = pd.to_datetime(candle_df['date']).dt.tz_localize(None)
    quotes = create_quotes(pd.DataFrame([convert_record_02(c) for c in candle_df.values],
                                        columns=['datetime', 'open', 'high', 'low', 'close', 'volume']))
    indicator_data = (quotes, from_date, to_date, interval)

    to_date_naive = to_date.replace(tzinfo=None)
    new_candles = [c for c in candle_data if
                   from_date <= datetime.fromisoformat(c[0]).replace(tzinfo=None) <= to_date_naive]
    bulk_data = [{"symbol": symbol, "interval": interval, "datetime": to_ist_naive(c[0]), "open": c[1], "high": c[2],
                  "low": c[3], "close": c[4], "volume": c[5]} for c in new_candles]

    return {'symbol': symbol, 'indicator_data': indicator_data, 'bulk_data': bulk_data}
