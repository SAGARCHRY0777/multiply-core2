import math
import json
import os
import logging
from contextlib import contextmanager
import requests
from dotenv import load_dotenv
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, Column, String, DateTime, func, select, text
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Dict, Any, Optional
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, declarative_base
import pytz
from pytz import timezone
import regex as re
import pandas as pd
import gc
from torch.fx.experimental.symbolic_shapes import lru_cache
from datetime import datetime
from stock_indicators import Quote

load_dotenv()

def get_db_url():
    """Get database connection URL from environment variables."""
    return os.getenv('DB_URL')

db_url = get_db_url()
engine = create_engine(db_url, connect_args={'sslmode': 'require'})
Base = declarative_base()

class CandleData(Base):
    __tablename__ = 'candle_data'
    symbol = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    datetime = Column(DateTime, primary_key=True)

def fetch_config(filename="config.json"):
    with open(os.path.join(os.path.dirname(__file__), f"../{filename}")) as f:
        data = json.load(f)
    return data

def get_five_minute_symbols(candidate_symbols=None):
    """
    Returns symbols that have FIVE_MINUTE data.
    If 'candidate_symbols' is provided, checks ONLY for those symbols (efficient).
    """
    try:
        if candidate_symbols:
            # Optimized: Check presence only for specific symbols
            query = text("SELECT DISTINCT symbol FROM profitdict WHERE interval = 'FIVE_MINUTE' AND symbol IN :symbols")
            params = {"symbols": tuple(candidate_symbols)}
            with engine.connect() as conn:
                return conn.execute(query, params).scalars().all()
        else:
            # Optimized: Fetch all unique symbols
            query = text("SELECT DISTINCT symbol FROM profitdict WHERE interval = 'FIVE_MINUTE'")
            with engine.connect() as conn:
                return conn.execute(query).scalars().all()

    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def get_filtered_nifty50_symbols():
    """
    Fetch NIFTY 50 from NSE and return only those present in DB (5-min).
    Optimized to Query DB only for the relevant 50 symbols.
    """
    try:
        # 1. Fetch NIFTY 50 List efficiently
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com/"
        }
        # Handshake & Fetch
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        response = session.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
            headers=headers,
            timeout=5
        )

        data = response.json()
        nifty_symbols = [
            x["symbol"] for x in data.get("data", [])
            if x["symbol"] != "NIFTY 50"
        ]

        # 2. Filter using DB (Push down predicate)
        if not nifty_symbols:
            return []

        return get_five_minute_symbols(candidate_symbols=nifty_symbols)

    except Exception as e:
        print(f"Error in Nifty 50 filter: {e}")
        return []

def is_within_time_range(start_time, end_time, current_time):
    # Handle case where current_time is a datetime object instead of a time object
    if isinstance(current_time, datetime):
        current_time = current_time.time()
    return start_time <= current_time <= end_time

def get_last_and_start_dates_candle_data_for_symbols(
    symbols: Iterable[str],
    interval: str,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each symbol in `symbols`, return:
      - last_dt: most recent datetime
      - start_date: datetime that is 360 rows before the last one (or the earliest available if <360 rows)

    Queries each symbol in parallel (ThreadPoolExecutor). Returns a dict:
        { symbol: {"start_date": datetime_or_none, "last_dt": datetime_or_none} }
    """
    # create engine/session factory once
    db_url = get_db_url()
    engine: Engine = create_engine(db_url, connect_args={"sslmode": "require"})
    Session = sessionmaker(bind=engine, future=True)

    def worker(symbol: str) -> (str, Dict[str, Any]):
        """Query a single symbol and return (symbol, result_dict)."""
        try:
            with Session() as session:
                # 1) most recent datetime
                stmt_last = select(func.max(CandleData.datetime)).where(
                    CandleData.interval == interval,
                    CandleData.symbol == symbol,
                )
                last_dt = session.execute(stmt_last).scalar_one_or_none()

                # 2) 360th most recent datetime (i.e. offset 359) — if missing, fallback to earliest datetime
                stmt_360th = (
                    select(CandleData.datetime)
                    .where(CandleData.interval == interval, CandleData.symbol == symbol)
                    .order_by(CandleData.datetime.desc())
                    .offset(359)
                    .limit(1)
                )
                start_date = session.execute(stmt_360th).scalar_one_or_none()

                if start_date is None:
                    # fewer than 360 rows — use earliest available row (min datetime)
                    stmt_min = select(func.min(CandleData.datetime)).where(
                        CandleData.interval == interval,
                        CandleData.symbol == symbol,
                    )
                    start_date = session.execute(stmt_min).scalar_one_or_none()

                return symbol, {"start_date": start_date, "last_dt": last_dt}

        except Exception as e:
            # return None values on error but include message in a debug-friendly way
            return symbol, {"start_date": None, "last_dt": None, "error": str(e)}

    results: Dict[str, Dict[str, Any]] = {}
    try:
        # choose a sane worker count for IO-bound DB calls
        if max_workers is None:
            # min(32, len(symbols)) avoids creating huge thread pools for large symbol lists
            max_workers = min(32, max(1, len(list(symbols))))
        # symbols may be an iterator; make a list to iterate multiple times
        symbol_list = list(symbols)

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(worker, sym): sym for sym in symbol_list}
            for fut in as_completed(futures):
                sym, res = fut.result()
                results[sym] = res

        return results
    finally:
        engine.dispose()

def adjust_from_date(last_available_date, interval):
    if interval == "ONE_MINUTE":
        return last_available_date + timedelta(minutes=1)
    elif interval == 'FIVE_MINUTE':
        return last_available_date + timedelta(minutes=5)
    elif interval == 'FIFTEEN_MINUTE':
        return last_available_date + timedelta(minutes=15)
    elif interval == 'ONE_HOUR':
        return last_available_date + timedelta(hours=1)
    elif interval == 'ONE_DAY':
        return last_available_date + timedelta(days=1)
    elif interval == 'ONE_WEEK':
        return last_available_date + timedelta(weeks=1)
    elif interval == 'ONE_MONTH':
        return last_available_date + relativedelta(months=1)
    else:
        raise ValueError(f"Unsupported interval: {interval}")

def get_nearest_expiry_commodities(all_symbol_data, commodity_names):
    """
    Finds the nearest expiry future contracts for a given list of commodities.

    Args:
        all_symbol_data (List[dict]): A list of all available symbol dictionaries.
        commodity_names (List[str]): A list of commodity names (e.g., ['GOLD', 'SILVER']).

    Returns:
        List[dict]: A list of symbol dictionaries for the nearest expiry contracts.
    """
    latest_commodities = []
    today = datetime.now().date()
    
    for name in commodity_names:
        # Filter for FUTCOM instruments in the MCX exchange for the given commodity name
        filtered_symbols = [
            s for s in all_symbol_data
            if s.get('name') == name and
               s.get('instrumenttype') == 'FUTCOM' and
               s.get('exch_seg') == 'MCX'
        ]

        if not filtered_symbols:
            logging.warning(f"No FUTCOM MCX symbol found for {name}")
            continue

        # Find the symbol with the NEAREST (min) expiry date that hasn't expired yet
        try:
            # Filter out expired contracts and find the nearest one
            valid_symbols = []
            for s in filtered_symbols:
                try:
                    expiry_date = datetime.strptime(s.get('expiry', '1970-01-01'), '%d%b%Y').date()
                    if expiry_date >= today:  # Only include non-expired contracts
                        valid_symbols.append((s, expiry_date))
                except ValueError:
                    continue
            
            if valid_symbols:
                # Use min() to get nearest expiry, not max()!
                nearest_symbol = min(valid_symbols, key=lambda x: x[1])[0]
                latest_commodities.append(nearest_symbol)
                logging.info(f"Selected {name}: {nearest_symbol.get('symbol')} expiry {nearest_symbol.get('expiry')}")
            else:
                logging.warning(f"No valid (non-expired) contracts found for {name}")
                
        except ValueError as e:
            logging.error(f"Could not parse expiry date for {name}. Error: {e}")

    return latest_commodities

def get_current_datetime_ist():
    ist = timezone('Asia/Kolkata')
    return datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(ist)

def load_user_db():
    user_list = []
    try:
        query = text("SELECT * FROM user_db")
        with engine.connect() as connection:
            result = connection.execute(query)
            # Use result.mappings() to get each row as a dictionary
            user_list = [dict(row) for row in result.mappings()]
    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            query = text("SELECT * FROM user_db")
            with engine.connect() as connection:
                result = connection.execute(query)
                # Use result.mappings() to get each row as a dictionary
                user_list = [dict(row) for row in result.mappings()]
        except Exception as e:
            print(f"An error occurred: {e}")

    return user_list

def create_symbol_dict():

    url = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'

    d = requests.get(url).json()

    return d

def get_all_symbol_data():
    all_symbol_dicts = create_symbol_dict()
    filtered_all_symbol_data = []
    accepted_exchanges_list = ['NSE', 'NFO', 'MCX', 'BSE']
    for data in all_symbol_dicts:
        if data['exch_seg'] in accepted_exchanges_list:
            filtered_all_symbol_data.append(data)
        else:
            continue

    return filtered_all_symbol_data

def filter_symbol_data(filtered_all_symbol_data):
    unique_analysis_symbol_dicts_list = []
    seen_names = set()

    # Extract names of symbols with FUTSTK or OPTSTK instrument types
    fut_opt_names = {
        symbol_dict["name"] for symbol_dict in filtered_all_symbol_data
        if symbol_dict.get("instrumenttype") in ("FUTSTK", "OPTSTK")
    }

    # Iterate through the filtered symbol data
    for symbol_dict in filtered_all_symbol_data:
        symbol_name = symbol_dict.get("symbol", "")
        symbol_name_value = symbol_dict.get("name", "")

        # Apply filters: no digits in symbol, unique name, in FUT/OPT names, in profit_dict_symbols
        if (
                not re.search(r'\d', symbol_name) and
                symbol_name_value not in seen_names and
                symbol_name_value in fut_opt_names
        ):
            unique_analysis_symbol_dicts_list.append(symbol_dict)
            seen_names.add(symbol_name_value)

    return unique_analysis_symbol_dicts_list

def get_start_and_end_dates(interval: str) -> tuple:
    """
    Calculate start and end dates based on the given interval.

    Args:
        interval (str): The interval for which to calculate dates.
                        Supported values: 'FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY', 'ONE_WEEK', 'ONE_MONTH'.

    Returns:
        tuple: Start and end dates (datetime objects).

    Raises:
        ValueError: If the interval is unsupported.
    """

    end_date = get_current_datetime_ist()

    # Determine start date based on the interval
    try:
        if interval == "ONE_MINUTE":
            start_date = end_date - timedelta(days=30)
        elif interval == "FIVE_MINUTE":
            start_date = end_date - timedelta(days=30)
        elif interval == "FIFTEEN_MINUTE":
            start_date = end_date - timedelta(days=200)
        elif interval == "ONE_HOUR":
            start_date = end_date - timedelta(days=200)
        elif interval == "ONE_DAY":
            start_date = end_date - timedelta(days=200)
        elif interval == "ONE_WEEK":
            start_date = end_date - relativedelta(years=2)
        elif interval == "ONE_MONTH":
            start_date = end_date - relativedelta(years=5)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    except Exception as e:
        logging.error(f"Error calculating dates for interval {interval}: {e}", exc_info=True)
        raise

    return start_date, end_date

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def is_price_diff_more_than_60_percent(hi_stock_df):
    """
    Check if the current close is at least 60% below the highest high
    in the entire DataFrame (all rows).
    """
    highest_price = hi_stock_df['high'].max()
    current_price = hi_stock_df.iloc[-1]['close']
    price_diff_percent = ((highest_price - current_price) / highest_price) * 100
    return price_diff_percent > 60

# NOTE: Removed @lru_cache - can't cache when 'symbols' is a list (unhashable)
def get_interval_specific_candle_data(
    interval: str,
    start_date,
    end_date,
    symbols: Optional[List[str]] = None
) -> Dict:
    """
    Fetch candle data for a specific interval, date range, and optional list of symbols.

    :param interval: The interval to filter by (e.g., '1H', '1D').
    :param start_date: The start date for filtering.
    :param end_date: The end date for filtering.
    :param symbols: An optional list of symbols to filter by.
    :return: A dictionary of symbols and their corresponding records.
    """
    base_query = """
        SELECT symbol, datetime,
               first_value(interval) OVER (PARTITION BY symbol) AS interval,
               open, high, low, close, volume
        FROM candle_data
        WHERE interval = :interval
          AND datetime BETWEEN :start_date AND :end_date
    """

    params = {
        'interval': interval,
        'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
        'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S')
    }

    if symbols:
        base_query += " AND symbol = ANY(:symbols)"
        params['symbols'] = list(symbols)  # Ensure it's a list for ANY()

    base_query += " ORDER BY datetime DESC"

    from sqlalchemy import bindparam
    from sqlalchemy.dialects.postgresql import ARRAY
    from sqlalchemy.types import String
    
    if symbols:
        query = text(base_query).bindparams(
            bindparam('interval', interval),
            bindparam('start_date', start_date.strftime('%Y-%m-%d %H:%M:%S')),
            bindparam('end_date', end_date.strftime('%Y-%m-%d %H:%M:%S')),
            bindparam('symbols', list(symbols), type_=ARRAY(String))
        )
    else:
        query = text(base_query).bindparams(
            bindparam('interval', interval),
            bindparam('start_date', start_date.strftime('%Y-%m-%d %H:%M:%S')),
            bindparam('end_date', end_date.strftime('%Y-%m-%d %H:%M:%S'))
        )

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            candle_data_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if candle_data_df.empty:
            return {}

        # Optimize memory usage
        float_cols = ['open', 'high', 'low', 'close']
        for col in float_cols:
            if col in candle_data_df.columns:
                candle_data_df[col] = candle_data_df[col].astype(np.float32)
        if 'volume' in candle_data_df.columns:
            candle_data_df['volume'] = candle_data_df['volume'].astype(np.int32)

        candle_data_dict = {
            symbol: group[['datetime', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            for symbol, group in candle_data_df.groupby('symbol')
        }

        return candle_data_dict

    except Exception as e:
        logging.error(f"Error fetching candle data: {str(e)}")
        raise


@contextmanager
def manage_interval_data(interval, start_date, end_date, indicators_list, symbols_list=None):
    """
    Context manager for handling interval-specific data.
    Ensures proper cleanup of memory-intensive data resources.
    """
    candle_data, indicator_data = None, None
    try:
        # Fetch interval-specific data
        candle_data = get_interval_specific_candle_data(interval, start_date, end_date, symbols=symbols_list)
        indicator_data = get_interval_specific_indicator_data(interval, start_date, end_date,
                                                              indicators_list, symbols=symbols_list)
        yield candle_data, indicator_data  # Yield data for use in the context
    except Exception as e:
        import traceback
        logging.error(f"Error managing data for interval {interval}: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        raise
    finally:
        # Cleanup
        for resource in [candle_data, indicator_data]:
            if resource is not None:
                del resource
        gc.collect()


def get_interval_specific_indicator_data(
        interval: str,
        start_date,
        end_date,
        indicators_list: List[str],
        symbols: Optional[List[str]] = None
):
    """
    Fetches interval-specific indicator data for a given date range, indicator list,
    and optional list of symbols.

    :param interval: The interval for filtering (e.g., '1H', '1D').
    :param start_date: Start date for filtering.
    :param end_date: End date for filtering.
    :param indicators_list: List of indicators to include in the result.
    :param symbols: An optional list of symbols to filter by.
    :return: A nested dictionary {symbol: {indicator: [(datetime, value), ...], ...}, ...}.
    """
    table_name = 'all_indicators_data'

    # Lowercase and sanitize the indicators list
    indicators = [ind.lower() for ind in indicators_list if isinstance(ind, str)]
    columns = ['symbol', 'datetime'] + indicators

    columns_str = ', '.join(f'"{col}"' for col in columns)

    base_query = f"""
        SELECT {columns_str} 
        FROM {table_name}
        WHERE "interval" = :interval
          AND "datetime" BETWEEN :start_date AND :end_date
    """

    from sqlalchemy import bindparam
    from sqlalchemy.dialects.postgresql import ARRAY
    from sqlalchemy.types import String
    
    if symbols:
        # Add symbols filter to query
        base_query += ' AND "symbol" = ANY(:symbols)'
        query = text(base_query).bindparams(
            bindparam('interval', interval),
            bindparam('start_date', start_date.strftime('%Y-%m-%d %H:%M:%S')),
            bindparam('end_date', end_date.strftime('%Y-%m-%d %H:%M:%S')),
            bindparam('symbols', list(symbols), type_=ARRAY(String))
        )
    else:
        query = text(base_query).bindparams(
            bindparam('interval', interval),
            bindparam('start_date', start_date.strftime('%Y-%m-%d %H:%M:%S')),
            bindparam('end_date', end_date.strftime('%Y-%m-%d %H:%M:%S'))
        )

    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            indicator_data_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if indicator_data_df.empty:
            return {}

        result_dict = {}
        for symbol, group in indicator_data_df.groupby('symbol'):
            result_dict[symbol] = {}
            for indicator in indicators:
                if indicator in group.columns:
                    # Convert to list of tuples for consistency
                    indicator_data = [tuple(x) for x in group[['datetime', indicator]].dropna().to_numpy()]
                    result_dict[symbol][indicator] = indicator_data

        return result_dict

    except Exception as e:
        raise RuntimeError(f"Failed to fetch filtered indicator data: {e}")



def preprocess_data(intervals, unique_symbol_dicts_list):
    """
    Preprocess data for all symbols and intervals.
    Fetch data once per interval and filter per symbol.
    Skip "beat-up" stocks if price is down more than 60% from the max high in the last 365 candles.
    """
    preprocessed_data = {}
    indicators_list = ['adx', 'pdi', 'mdi', 'rsi', 'macd', 'macd_signal', 'ema5', 'ema13', 'ema50']
    symbol_names = [symbol_dict['name'] for symbol_dict in unique_symbol_dicts_list]

    for interval in intervals:
        start_date, end_date = get_start_and_end_dates(interval)

        try:
            # Fetch interval data once
            with (manage_interval_data(interval, start_date, end_date, indicators_list, symbols_list=symbol_names) as
                  (all_candle_data, all_indicator_data)):
                interval_data = {}
                for symbol_dict in unique_symbol_dicts_list:
                    print("symbol_dict:", symbol_dict)
                    analysis_symbol = symbol_dict['name']

                    try:
                        # Filter data for the symbol
                        candle_data = all_candle_data.get(analysis_symbol, [])
                        indicator_data = all_indicator_data.get(analysis_symbol, {})

                        if not candle_data or not indicator_data:
                            logging.info(f'candle or indicator data not available for: {analysis_symbol} | {interval}')
                            continue

                        # Convert candle data to a DataFrame
                        candle_df = pd.DataFrame(
                            candle_data,
                            columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
                        )
                        candle_df['datetime'] = pd.to_datetime(candle_df['datetime']).dt.tz_localize(None)
                        candle_df = optimize_dataframe(candle_df)

                        # Process indicator data into a single DataFrame
                        indicator_frames = []
                        for indicator_name, data in indicator_data.items():
                            try:
                                df = pd.DataFrame(data, columns=['datetime', indicator_name])
                                df['datetime'] = pd.to_datetime(df['datetime'])
                                indicator_frames.append(
                                    df.drop_duplicates('datetime').set_index('datetime')
                                )
                            except Exception as ind_err:
                                logging.warning(f"Failed to process indicator {indicator_name}: {ind_err}")

                        if indicator_frames:
                            merged_indicators = pd.concat(indicator_frames, axis=1, join='outer')
                            merged_indicators.reset_index(inplace=True)
                            stock_df = pd.merge(candle_df, merged_indicators, on='datetime', how='outer')
                        else:
                            stock_df = candle_df

                        # Rename EMA columns if present
                        ema_column_mapping = {
                            'ema5': 'ema_5',
                            'ema13': 'ema_13',
                            'ema50': 'ema_50'
                        }
                        stock_df = stock_df.rename(columns=ema_column_mapping)

                        # Optimize and clean merged data
                        stock_df = optimize_dataframe(stock_df)
                        stock_df = stock_df.fillna(method='ffill').fillna(method='bfill')
                        stock_df = stock_df.sort_values("datetime").reset_index(drop=True)
                        macd_df = stock_df[['datetime', 'macd', 'macd_signal']].copy()
                        candle_df = candle_df.sort_values("datetime").reset_index(drop=True)
                        # -----------------------------
                        # Beat-up stock check
                        # -----------------------------
                        if is_price_diff_more_than_60_percent(stock_df):
                            logging.info(
                                f"Skipping beat-up stock: {analysis_symbol} on interval {interval}"
                            )
                            continue

                        # If it's not a beat-up stock, store preprocessed data
                        interval_data[analysis_symbol] = {
                            'stock_df': stock_df,
                            'macd_df': macd_df,
                            'candle_df': candle_df
                        }

                    except Exception as e:
                        logging.error(
                            f"Error preprocessing data for symbol {analysis_symbol} in interval {interval}: {e}"
                        )

                preprocessed_data[interval] = interval_data

        except Exception as e:
            logging.error(f"Error retrieving data for interval {interval}: {e}")

    return preprocessed_data

def get_symbol_dict(all_symbol_data, symbol):
    """
    Get symbol dictionary for a given symbol name.
    Handles both equity symbols (symbol-EQ) and indices/commodities (by name).
    """
    result_symbol_dict = {}
    for symbol_dict in all_symbol_data:
        # First try matching with '-EQ' suffix for equities
        if f"{symbol}-EQ" == symbol_dict['symbol']:
            result_symbol_dict = symbol_dict
            break
        # Also try direct name match for indices and commodities
        elif symbol == symbol_dict['name']:
            result_symbol_dict = symbol_dict
            break
    return result_symbol_dict

def filter_profit_data(profit_dict, analysis_symbol, interval, indices_list, commodities_list):

    try:
        # Extract interval-specific data for the symbol
        interval_data = profit_dict.get(analysis_symbol, {}).get(interval, None)
        if not interval_data:
            raise KeyError(f"No data found for symbol: {analysis_symbol}, interval: {interval}")

        # Convert string data to dictionary if needed
        if isinstance(interval_data, str):
            interval_data = json.loads(interval_data)

        # Extract the required fields
        bullish_average_sell_point = interval_data['batch_results'][-1][
            'bullish_momentum_setup_average_sell_point_type']
        bearish_average_sell_point = interval_data['batch_results'][-1][
            'bearish_momentum_setup_average_sell_point_type']
        bullish_average_profit_percentage = interval_data['batch_results'][-1][
            'bullish_momentum_setup_average_profit_percentage']
        bearish_average_profit_percentage = interval_data['batch_results'][-1][
            'bearish_momentum_setup_average_profit_percentage']

        # Return the filtered data
        return {
            'bullish_average_sell_point': bullish_average_sell_point,
            'bearish_average_sell_point': bearish_average_sell_point,
            'bullish_average_profit_percentage': bullish_average_profit_percentage,
            'bearish_average_profit_percentage': bearish_average_profit_percentage
        }

    except KeyError as e:
        logging.error(f"KeyError while filtering profit data: {e}")
        if analysis_symbol in indices_list or commodities_list:
            return {
                'bullish_average_sell_point': 'target',
                'bearish_average_sell_point': 'target',
                'bullish_average_profit_percentage': 0.2,
                'bearish_average_profit_percentage': -0.2
            }
        else:
            return {
                'bullish_average_sell_point': 'target',
                'bearish_average_sell_point': 'target',
                'bullish_average_profit_percentage': 0.8,
                'bearish_average_profit_percentage': -0.8
            }

    except Exception as e:
        logging.error(f"Unexpected error while filtering profit data: {e}")
        if analysis_symbol in indices_list or commodities_list:
            return {
                'bullish_average_sell_point': 'target',
                'bearish_average_sell_point': 'target',
                'bullish_average_profit_percentage': 0.2,
                'bearish_average_profit_percentage': -0.2
            }
        else:
            return {
                'bullish_average_sell_point': 'target',
                'bearish_average_sell_point': 'target',
                'bullish_average_profit_percentage': 0.8,
                'bearish_average_profit_percentage': -0.8
            }

def clear_memory(*args):
    """Force garbage collection and delete specified variables."""
    for arg in args:
        del arg
    gc.collect()
    logging.debug("Memory cleared")

def add_percentage(price, percent):
    return price * (1 + percent/100)

def subtract_percentage(price, percent):
    return price * (1 - percent/100)

def multiply_stoploss_calculator(
    symbol: str,
    interval: str,
    entry_timestamp: Any,
    direction: str,
):
    if not symbol:
        raise ValueError("symbol is required")
    if not interval:
        raise ValueError("interval is required")
    if not direction:
        raise ValueError("direction is required")

# Global IST timezone object
ist = pytz.timezone("Asia/Kolkata")

def safe_get_symbol_data(analysis_symbol, interval, profit_dict):
    """Safely extract and parse symbol data from profit dictionary."""
    try:
        if not analysis_symbol or not interval:
            logging.warning("Missing analysis_symbol or interval")
            return None

        interval_data = profit_dict.get(analysis_symbol, {}).get(interval, {})
        if not interval_data:
            logging.warning(f"No data found for symbol {analysis_symbol} and interval {interval}")
            return None

        if isinstance(interval_data, str):
            interval_data = json.loads(interval_data)

        return interval_data
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        logging.error(f"Error processing data for symbol {analysis_symbol}: {e}")
        return None


def extract_login_details(user_db, user_id):
    logging.debug(f"Extracting login details for user_id: {user_id}")
    for user_dict in user_db:
        if user_dict['user_id'] == user_id:
            logging.debug(f"Found login details for user_id: {user_id}")
            return user_dict
    logging.warning(f"No login details found for user_id: {user_id}")
    return None

def calculate_quantities(funds_info, withdraw_amount, lot_size, symbol_ltp, maximum_buy_percentage,
                         open_interest=None, limit_ratio=0.005):
    """Calculate total order quantities with safety checks."""
    try:
        # Validate inputs
        if not funds_info or not isinstance(funds_info, dict):
            logging.error("Invalid funds info")
            return 0, 0, 0

        lot_size = float(lot_size or 0)
        symbol_ltp = float(symbol_ltp or 0)

        if lot_size <= 0 or symbol_ltp <= 0:
            logging.error(f"Invalid lot size ({lot_size}) or symbol price ({symbol_ltp})")
            return 0, 0, 0

        per_lot_amount = symbol_ltp * lot_size
        if per_lot_amount <= 0:
            logging.error("Invalid per lot amount")
            return 0, 0, 0

        try:
            available_funds = max(0, float(funds_info['data']['net']) - withdraw_amount)
        except (KeyError, TypeError, ValueError) as e:
            logging.error(f"Error calculating available funds: {e}")
            return 0, 0, 0

        # Calculate max amount based on percentage
        if maximum_buy_percentage > 0:
            max_amount = available_funds * (maximum_buy_percentage / 100)
        else:
            max_amount = available_funds

        # Calculate the number of lots
        quantity_by_funds = available_funds / per_lot_amount if per_lot_amount > 0 else 0

        if maximum_buy_percentage > 0:
            quantity_by_max = max_amount / per_lot_amount if per_lot_amount > 0 else 0
            number_of_lots = math.floor(min(quantity_by_funds, quantity_by_max))
        else:
            quantity_by_max = quantity_by_funds
            number_of_lots = math.floor(quantity_by_max)

        if open_interest is not None:
            try:
                oi_value = float(open_interest)
            except (TypeError, ValueError):
                oi_value = None
            if oi_value is not None and oi_value > 0 and limit_ratio > 0:
                max_lots_by_oi = int(oi_value * limit_ratio)
                if max_lots_by_oi <= 0:
                    return 0, 0, per_lot_amount
                number_of_lots = min(number_of_lots, max_lots_by_oi)

        # Calculate the total quantity in terms of units
        total_quantity = number_of_lots * int(lot_size)
        disclosed_quantity = max(1, round(total_quantity / 10))  # Disclose 10% or at least 1 unit

        return total_quantity, disclosed_quantity, per_lot_amount
    except Exception as e:
        logging.error(f"Error in calculate_quantities: {e}")
        return 0, 0, 0


def prep_order_params(symbol, symbol_token, quantity, exchange, segment):
    print("prep_order_params")
    order_type = 'MARKET'
    if segment == 'Equity':
        product_type = 'DELIVERY'
    else:
        product_type = 'CARRYFORWARD'
    validity = 'DAY'
    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": symbol_token,
        "transactiontype": "BUY",
        "quantity": quantity,
        "ordertype": order_type,
        "producttype": product_type,
        "exchange": exchange,
        "duration": validity
    }
    # Remove None values from order_params
    order_params = {k: v for k, v in order_params.items() if v is not None}

    return order_params

def to_ist_naive(datetime_str):
    """
    Convert an ISO 8601 datetime string to naive datetime in IST.
    """
    # Convert string to datetime (assuming UTC)
    dt = datetime.fromisoformat(datetime_str)
    # Convert to IST timezone
    dt_ist = dt.astimezone(ist)
    # Return as naive datetime (strip timezone info)
    return dt_ist.replace(tzinfo=None)

def convert_record_02(candle):
    try:
        # Extract values from candle
        timestamp, open_price, high_price, low_price, close_price, volume = candle

        # Handle pandas Timestamp or datetime objects
        if isinstance(timestamp, (pd.Timestamp, datetime)):
            parsed_datetime = timestamp
        else:
            # Convert string timestamp to datetime if needed
            # Clean the timestamp string of any potential encoding issues
            if isinstance(timestamp, str):
                timestamp = timestamp.encode('ascii', 'ignore').decode('ascii')
            else:
                timestamp = str(timestamp)

            # Define the possible datetime formats
            formats = [
                '%Y-%m-%d %H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f'
            ]

            # Try to parse the datetime string
            parsed_datetime = None
            for fmt in formats:
                try:
                    parsed_datetime = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue

            if parsed_datetime is None:
                raise ValueError(f"Could not parse timestamp: {timestamp}")

        # Format the datetime consistently
        formatted_datetime = parsed_datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Convert numeric values with validation and handle potential string encoding issues
        try:
            # Clean any string values before conversion
            def clean_value(val):
                if isinstance(val, str):
                    try:
                        # First try to encode as UTF-8 and decode back
                        return val.encode('utf-8', 'ignore').decode('utf-8')
                    except UnicodeError:
                        # If that fails, try ASCII as fallback
                        return val.encode('ascii', 'ignore').decode('ascii')
                return val

            open_price = float(clean_value(open_price))
            high_price = float(clean_value(high_price))
            low_price = float(clean_value(low_price))
            close_price = float(clean_value(close_price))
            volume = float(clean_value(volume))

            output = (
                formatted_datetime,
                open_price,
                high_price,
                low_price,
                close_price,
                int(volume)
            )
            return output
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting numeric values: {e}")

    except Exception as e:
        raise ValueError(f"Error in convert_record_02: {e}")

# Create Quote objects from DataFrame
def create_quotes(df):
    if df.empty:
        return []

    quotes = []

    # Ensure date column is datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['datetime'])

    if df.empty:
        return []

    # Convert numeric columns to float, replacing NaN with 0
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by date to ensure proper order
    df = df.sort_values('datetime')

    for _, row in df.iterrows():
        try:
            quote = Quote(
                row['datetime'].to_pydatetime(),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            )
            quotes.append(quote)
        except Exception as e:
            print(f"Error creating quote: {e}")
            continue

    return quotes


def get_profit_dict():
    from sqlalchemy import text, create_engine
    import os
    import json

    db_url = os.environ.get('DB_URL')
    if not db_url:
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.environ.get('DB_URL')

    if not db_url:
        logging.warning("DB_URL not set, returning empty profit dict")
        return {}

    engine = create_engine(db_url, connect_args={'sslmode': 'require'})

    profit_dict = {}
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT *
                FROM profitdict
            """)
            result = conn.execute(query)

            for row in result.mappings():
                symbol = row['symbol']
                interval = row['interval']
                batch_results = row['batch_results']

                if symbol not in profit_dict:
                    profit_dict[symbol] = {}

                # Parse JSON if it's a string
                if isinstance(batch_results, str):
                    batch_results = json.loads(batch_results)

                profit_dict[symbol][interval] = {'batch_results': batch_results}

    except Exception as e:
        logging.error(f"Error fetching profit dict: {e}")
    finally:
        engine.dispose()

    return profit_dict

def get_azure_storage_connection_string():
    """Get Azure Storage connection string from environment variables."""
    return os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def get_symbol_specific_candle_data(symbol, interval, start_date, end_date):
    """Fetch candle data for symbol/interval/date range."""
    query = text("""
        SELECT datetime, open, high, low, close, volume FROM candle_data
        WHERE interval = :interval AND symbol = :symbol
        AND datetime >= :start_date AND datetime <= :end_date
        ORDER BY datetime ASC;
    """)
    params = {
        'interval': interval, 'symbol': symbol,
        'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
        'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
            return df.values.tolist()
    except Exception as e:
        print(f"Error fetching candle data: {e}")
        return []

def get_symbol_specific_indicator_data(symbol, interval, indicator_type, start_date, end_date):
    """Fetch indicator data for symbol/interval/date range."""
    table = 'all_indicators_data' if indicator_type == 'all_indicators' else 'current_indicators_data'
    query = f"""
        SELECT * FROM {table}
        WHERE interval = '{interval}' AND symbol = '{symbol}'
        AND datetime >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
        AND datetime <= '{end_date.strftime('%Y-%m-%d %H:%M:%S')}'
        ORDER BY datetime ASC;
    """
    try:
        df = pd.read_sql(query, engine)
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Error fetching indicator data: {e}")
        return []
