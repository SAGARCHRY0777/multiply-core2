import pandas as pd
import os
import pickle
import gc
import traceback
from datetime import datetime, timedelta
import pytz
import time
from typing import Dict, Tuple, Optional, List
from multiprocessing import Pool, cpu_count
from functions.data_utils import (
    get_symbol_specific_candle_data,
    get_symbol_specific_indicator_data,
    get_all_symbol_data,
    filter_symbol_data,
    get_filtered_nifty50_symbols
)
from functions.profit_dict_storage import (
    save_profit_dict_row,
    merge_trade_histories,
    get_existing_symbol_history,
)
from functions.condition_utils import (
    extract_and_format_averages
)
from conditions import (
    detect_divergence,
    get_divergence,
    check_ema_5_crossing_13,
    check_ema_5_crossing_50,
    candle_with_good_volume,
    check_most_recent_macd_crossing,
    directional_indicator_check,
    check_adx_greater_than_14,
    check_rsi,
    rsi_trend_check,
    price_50ema,
    check_limit_price_break
)
from functions.trade_processing import process_buy_index

TRADE_CONFIG = {
    "equity": {
        "FIFTEEN_MINUTE": {"total_candles": 104, "profit_threshold": 0.02},
        "ONE_HOUR": {"total_candles": 42, "profit_threshold": 0.03},
        "ONE_DAY": {"total_candles": 14, "profit_threshold": 0.04},
        "ONE_WEEK": {"total_candles": 4, "profit_threshold": 0.06},
        "ONE_MONTH": {"total_candles": 5, "profit_threshold": 0.035}
    },
    "indices": {
        "ONE_MINUTE": {"total_candles": 80, "profit_threshold": 0.002},
        "FIVE_MINUTE": {"total_candles": 80, "profit_threshold": 0.002},
        "FIFTEEN_MINUTE": {"total_candles": 16, "profit_threshold": 0.002},
        "ONE_HOUR": {"total_candles": 42, "profit_threshold": 0.003},
        "ONE_DAY": {"total_candles": 14, "profit_threshold": 0.004}
    },
    "commodities": {
        "ONE_MINUTE": {"total_candles": 80, "profit_threshold": 0.002},
        "FIVE_MINUTE": {"total_candles": 80, "profit_threshold": 0.002},
        "FIFTEEN_MINUTE": {"total_candles": 16, "profit_threshold": 0.002},
        "ONE_HOUR": {"total_candles": 42, "profit_threshold": 0.003},
        "ONE_DAY": {"total_candles": 14, "profit_threshold": 0.004}
    },
    "symbol_types": {
        "indices": ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"],
        "commodities": ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]
    }
}

SPECIAL_SYMBOLS = ['NIFTY', 'BANKNIFTY', 'SENSEX', 'GOLD', 'SILVER', 'CRUDEOIL', 'NATURALGAS']
INTERVAL_TO_TIMEFRAME = {
    'ONE_MINUTE': '1M', 'FIVE_MINUTE': '5M', 'FIFTEEN_MINUTE': '15M',
    'ONE_HOUR': '1H', 'ONE_DAY': '1D', 'ONE_WEEK': '1W', 'ONE_MONTH': '1MO'
}

CHECKPOINT_FILE = 'update_profit_dict_checkpoint.pkl'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_symbol_type(symbol: str) -> str:
    """Determine symbol type."""
    symbol_upper = symbol.upper()
    if symbol_upper in TRADE_CONFIG['symbol_types'].get('indices', []):
        return 'indices'
    elif symbol_upper in TRADE_CONFIG['symbol_types'].get('commodities', []):
        return 'commodities'
    return 'equity'


def get_trade_config(symbol: str, interval: str) -> Dict:
    """Get trade config for symbol/interval."""
    symbol_type = get_symbol_type(symbol)
    return TRADE_CONFIG.get(symbol_type, {}).get(interval, {
        'total_candles': 104, 'profit_threshold': 0.02
    })


def get_intervals_for_symbol(symbol: str, nifty50: List[str]) -> List[str]:
    """Get intervals for symbol type."""
    default = ['FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY', 'ONE_WEEK', 'ONE_MONTH']
    if symbol in SPECIAL_SYMBOLS:
        return ['ONE_MINUTE', 'FIVE_MINUTE'] + default
    elif symbol in nifty50:
        return ['FIVE_MINUTE'] + default
    return default


def parse_datetime_string(date_str: str) -> datetime:
    """Parse datetime string."""
    date_str = date_str.split('+')[0].strip().replace(" UTC", "")
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').replace(
        tzinfo=pytz.timezone('Asia/Kolkata'))


def get_last_processed_date(symbol: str, interval: str) -> Optional[datetime]:
    """Get last processed date from database."""
    try:
        history = get_existing_symbol_history(symbol, interval)
        if history:
            latest = max(history, key=lambda x: x['to'])
            return parse_datetime_string(latest['to'])
    except Exception:
        pass
    return None

# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(symbol: str, interval: str):
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'symbol': symbol, 'interval': interval, 'time': datetime.now()}, f)


def load_checkpoint() -> Optional[Dict]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# =============================================================================
# VECTORIZED PRE-FILTER
# =============================================================================

def vectorized_prefilter(df: pd.DataFrame) -> pd.Series:
    """Fast vectorized pre-filter to reduce candidates."""
    bullish = (
        (df['close'] > df['ema_50']) & (df['pdi'] > df['mdi']) &
        (df['adx'] > 14) & (df['rsi'] > 60) & (df['macd'] > df['macd_signal'])
    )
    bearish = (
        (df['close'] < df['ema_50']) & (df['mdi'] > df['pdi']) &
        (df['adx'] > 14) & (df['rsi'] < 40) & (df['macd'] < df['macd_signal'])
    )
    return bullish | bearish


def meet_buy_conditions_with_divergence(interval: str, stock_df: pd.DataFrame,
                                        filtered_df: pd.DataFrame, index: int) -> Tuple[
    Dict, Optional[float], Optional[datetime]]:
    """Exact meet_buy_conditions with divergence date extraction."""
    result = {}
    last_close = filtered_df.iloc[-1]['close']
    timeframe = INTERVAL_TO_TIMEFRAME.get(interval, '15M')

    divergences, total_check_days = detect_divergence(filtered_df, timeframe)
    if divergences is None:
        return result, None, None

    divergence_date, divergence_result = get_divergence(divergences)

    if divergence_result == "bullish divergence":
        ema_13_result, ema_13_idx = check_ema_5_crossing_13(divergence_date, filtered_df, timeframe, total_check_days)
        if ema_13_result == "ema_crossing_above_13":
            ema_50_result, ema_50_idx = check_ema_5_crossing_50(divergence_date, filtered_df, timeframe,
                                                                total_check_days)
            if ema_50_result == "ema_crossing_above_50":
                for check_volume in [True, False]:
                    if check_volume:
                        vol_check, _ = candle_with_good_volume(ema_50_idx, filtered_df)
                        if vol_check != "bullish_candle_with_good_volume":
                            continue

                    if price_50ema(stock_df, last_close, index) != "price_above_50ema":
                        continue
                    if check_most_recent_macd_crossing(filtered_df) != "macd_over_signal":
                        continue
                    if directional_indicator_check(stock_df, index) != "plus_di_above_minus_di":
                        continue
                    if check_adx_greater_than_14(stock_df, index) != "adx_greater_than_14":
                        continue
                    if check_rsi(stock_df, index) != "rsi_above_60":
                        continue
                    if check_limit_price_break(ema_50_idx, filtered_df) != "price_above_limit_high":
                        continue
                    if rsi_trend_check(filtered_df) != "rsi_uptrend":
                        continue

                    result['bullish_momentum_setup_label'] = 'True'
                    return result, filtered_df.at[ema_50_idx, 'close'], divergence_date

    elif divergence_result == "bearish divergence":
        ema_13_result, ema_13_idx = check_ema_5_crossing_13(divergence_date, filtered_df, timeframe, total_check_days)
        if ema_13_result == "ema_crossing_below_13":
            ema_50_result, ema_50_idx = check_ema_5_crossing_50(divergence_date, filtered_df, timeframe,
                                                                total_check_days)
            if ema_50_result == "ema_crossing_below_50":
                for check_volume in [True, False]:
                    if check_volume:
                        vol_check, _ = candle_with_good_volume(ema_50_idx, filtered_df)
                        if vol_check != "bearish_candle_with_good_volume":
                            continue

                    if price_50ema(stock_df, last_close, index) != "price_below_50ema":
                        continue
                    if check_most_recent_macd_crossing(filtered_df) != "signal_over_macd":
                        continue
                    if directional_indicator_check(stock_df, index) != "minus_di_above_plus_di":
                        continue
                    if check_adx_greater_than_14(stock_df, index) != "adx_greater_than_14":
                        continue
                    if check_rsi(stock_df, index) != "rsi_below_40":
                        continue
                    if check_limit_price_break(ema_50_idx, filtered_df) != "price_below_limit_low":
                        continue
                    if rsi_trend_check(filtered_df) != "rsi_downtrend":
                        continue

                    result['bearish_momentum_setup_label'] = 'True'
                    return result, filtered_df.at[ema_50_idx, 'close'], divergence_date

    return result, None, None


# =============================================================================
# MULTIPROCESSING
# =============================================================================

_GLOBAL_DF = None
_GLOBAL_INTERVAL = None
_GLOBAL_CONFIG = None


def _init_worker(df_dict, interval, config):
    global _GLOBAL_DF, _GLOBAL_INTERVAL, _GLOBAL_CONFIG
    _GLOBAL_DF = pd.DataFrame(df_dict)
    _GLOBAL_INTERVAL = interval
    _GLOBAL_CONFIG = config


def _worker(idx):
    global _GLOBAL_DF, _GLOBAL_INTERVAL, _GLOBAL_CONFIG
    return process_candidate(idx, _GLOBAL_DF, _GLOBAL_INTERVAL, _GLOBAL_CONFIG)


def process_candidate(idx: int, stock_df: pd.DataFrame, interval: str, config: Dict = None) -> Optional[Tuple]:
    filtered_df = stock_df[stock_df['datetime'] <= stock_df.iloc[idx]['datetime']].copy()
    if len(filtered_df) < 50:
        return None

    result, threshold, div_date = meet_buy_conditions_with_divergence(interval, stock_df, filtered_df, idx)

    # Use config if available
    profit_threshold = config.get('profit_threshold', 0.02) if config else 0.02
    total_candles = config.get('total_candles', 104) if config else 104

    if result.get('bullish_momentum_setup_label'):
        return (idx, 'bullish_momentum_setup', threshold, div_date, profit_threshold, total_candles)
    elif result.get('bearish_momentum_setup_label'):
        return (idx, 'bearish_momentum_setup', threshold, div_date, profit_threshold, total_candles)
    return None


def process_candidates_parallel(candidates: List[int], df: pd.DataFrame,
                                interval: str, config: Dict, n_workers: int = None) -> List[Tuple]:
    """Process candidates in parallel using multiprocessing."""
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if len(candidates) < 100:
        return [r for r in (process_candidate(i, df, interval, config) for i in candidates) if r]

    try:
        with Pool(n_workers, initializer=_init_worker, initargs=(df.to_dict('list'), interval, config)) as pool:
            results = pool.map(_worker, candidates)
        return [r for r in results if r]
    except Exception:
        return [r for r in (process_candidate(i, df, interval, config) for i in candidates) if r]


# =============================================================================
# CORE DETECTION ENGINE
# =============================================================================

def detect_trades_fast(symbol: str, interval: str, start_date: datetime,
                       end_date: datetime, n_workers: int = None, verbose: bool = True) -> Dict:
    """
    Fast trade detection for single symbol/interval.

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        interval: Time interval (e.g., 'FIFTEEN_MINUTE')
        start_date: Start datetime
        end_date: End datetime
        n_workers: Number of worker processes
        verbose: Print progress

    Returns:
        Dict with symbol_history (trades) and metadata
    """
    timeframe = INTERVAL_TO_TIMEFRAME.get(interval, '15M')
    config = get_trade_config(symbol, interval)
    t0 = time.time()

    # Fetch data
    candle_data = get_symbol_specific_candle_data(symbol, interval, start_date, end_date)
    indicator_data = get_symbol_specific_indicator_data(symbol, interval, 'all_indicators', start_date, end_date)

    if not candle_data or not indicator_data:
        return {'error': 'No data', 'symbol': symbol, 'interval': interval, 'symbol_history': []}

    # Build DataFrame
    candle_df = pd.DataFrame(candle_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    candle_df['datetime'] = pd.to_datetime(candle_df['datetime']).dt.tz_localize(None)

    indicator_df = pd.DataFrame(indicator_data)
    indicator_df['datetime'] = pd.to_datetime(indicator_df['datetime']).dt.tz_localize(None)
    indicator_df = indicator_df.rename(columns={'ema5': 'ema_5', 'ema13': 'ema_13', 'ema50': 'ema_50'})

    stock_df = pd.merge(candle_df, indicator_df, on='datetime', how='inner')
    stock_df = stock_df.sort_values('datetime').reset_index(drop=True).ffill().bfill()

    if verbose:
        print(f"    Data: {len(stock_df)} rows")

    # Pre-filter
    mask = vectorized_prefilter(stock_df)
    candidates = [i for i in stock_df.index[mask].tolist() if i >= 50]

    if verbose:
        print(f"    Candidates: {len(candidates)} ({len(candidates) / len(stock_df) * 100:.1f}%)")

    # Process candidates
    signals = process_candidates_parallel(candidates, stock_df, interval, config, n_workers)

    # Dedupe by divergence_date
    seen = set()
    deduped = []
    for s in signals:
        key = (s[3], s[1])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    # Process to trades
    trades = []
    
    # Unpack updated tuple which now includes profit_threshold and total_candles
    for idx, setup, thresh, div_date, profit_thresh, total_candles in deduped:
        try:
            trade = process_buy_index((symbol, stock_df, interval, timeframe, idx, setup, thresh, profit_thresh, total_candles))
            if trade:
                trades.append(trade)
        except:
            pass

    # Dedupe by sell_index
    by_sell = {}
    for t in trades:
        sell_idx = t['sell_index']
        if sell_idx not in by_sell or t['buy_index'] < by_sell[sell_idx]['buy_index']:
            by_sell[sell_idx] = t

    final = sorted(by_sell.values(), key=lambda x: x['buy_index'])

    if verbose:
        print(f"    Trades: {len(final)} ({time.time() - t0:.2f}s)")

    return {
        'symbol': symbol, 'interval': interval,
        'from_date': str(stock_df.iloc[0]['datetime']) if len(stock_df) > 0 else None,
        'to_date': str(stock_df.iloc[-1]['datetime']) if len(stock_df) > 0 else None,
        'total_trades': len(final), 'processing_time': time.time() - t0,
        'config': config, 'symbol_history': final
    }


# =============================================================================
# MAIN UPDATE FUNCTION
# =============================================================================

def fetch_historical_trades(symbols: List[str] = None, intervals: List[str] = None,
                            incremental: bool = True, save_to_db: bool = True,
                            n_workers: int = None, verbose: bool = True) -> bool:
    try:
        gc.enable()
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)

        if verbose:
            print("=" * 60)
            print("UPDATE PROFIT DICT (FAST)")
            print(f"Started: {now}")
            print("=" * 60)

        try:
            nifty50 = get_filtered_nifty50_symbols()
        except:
            nifty50 = []

        if symbols:
            symbol_list = symbols
        else:
            all_data = get_all_symbol_data()
            filtered = filter_symbol_data(all_data)
            symbol_list = list(set([s['name'] for s in filtered] + SPECIAL_SYMBOLS))

        # Prioritize special symbols
        special = [s for s in symbol_list if s in SPECIAL_SYMBOLS]
        others = [s for s in symbol_list if s not in SPECIAL_SYMBOLS]
        symbol_list = special + others

        if verbose:
            print(f"\nSymbols: {len(symbol_list)}")

        # Resume from checkpoint
        checkpoint = load_checkpoint()
        start_idx = 0
        start_interval = None
        if checkpoint:
            try:
                start_idx = symbol_list.index(checkpoint['symbol'])
                start_interval = checkpoint['interval']
                if verbose:
                    print(f"Resuming: {checkpoint['symbol']} ({checkpoint['interval']})")
            except ValueError:
                pass

        for i, symbol in enumerate(symbol_list[start_idx:], start=start_idx):
            if verbose:
                print(f"\n[{i + 1}/{len(symbol_list)}] {symbol}")

            sym_intervals = intervals or get_intervals_for_symbol(symbol, nifty50)

            if i == start_idx and start_interval:
                try:
                    idx = sym_intervals.index(start_interval)
                    sym_intervals = sym_intervals[idx:]
                except ValueError:
                    pass

            for interval in sym_intervals:
                save_checkpoint(symbol, interval)

                if verbose:
                    print(f"  {interval}:", end=" ")

                if incremental:
                    last = get_last_processed_date(symbol, interval)
                    start = last if last else now - timedelta(days=365 * 9)
                else:
                    start = now - timedelta(days=365 * 9)

                if start.tzinfo is None:
                    start = ist.localize(start)

                result = detect_trades_fast(symbol, interval, start, now, n_workers, verbose=False)

                if 'error' in result:
                    if verbose:
                        print("Skipped")
                    continue

                new_trades = result.get('symbol_history', [])
                if verbose:
                    print(f"{len(new_trades)} new", end="")

                if not new_trades:
                    if verbose:
                        print()
                    continue

                existing = get_existing_symbol_history(symbol, interval) or []
                merged = merge_trade_histories(existing, new_trades)

                if verbose:
                    print(f" -> {len(merged)} total")

                if save_to_db:
                    bull = extract_and_format_averages(merged, 'bullish_momentum_setup')
                    bear = extract_and_format_averages(merged, 'bearish_momentum_setup')

                    from_date = min(t['from'] for t in merged) if merged else result['from_date']
                    to_date = max(t['to'] for t in merged) if merged else result['to_date']

                    # Build batch_result in original format
                    batch_result = {
                        **bull, **bear,
                        'from_date': from_date,
                        'to_date': to_date,
                        'total_trades': bull.get('bullish_momentum_setup_total_trades', 0) + bear.get(
                            'bearish_momentum_setup_total_trades', 0),
                        'overall_from_date': from_date,
                        'overall_to_date': to_date,
                        'overall_bullish_average_profit_percentage': bull.get(
                            'bullish_momentum_setup_average_profit_percentage', 0),
                        'overall_bearish_average_profit_percentage': bear.get(
                            'bearish_momentum_setup_average_profit_percentage', 0),
                        'overall_bullish_average_sell_point_type': bull.get(
                            'bullish_momentum_setup_average_sell_point_type'),
                        'overall_bearish_average_sell_point_type': bear.get(
                            'bearish_momentum_setup_average_sell_point_type'),
                        'overall_bullish_average_duration_seconds': bull.get(
                            'bullish_momentum_setup_average_duration_seconds', 0),
                        'overall_bearish_average_duration_seconds': bear.get(
                            'bearish_momentum_setup_average_duration_seconds', 0),
                        'overall_bullish_total_trades': bull.get('bullish_momentum_setup_total_trades', 0),
                        'overall_bearish_total_trades': bear.get('bearish_momentum_setup_total_trades', 0),
                        'overall_total_trades': len(merged)
                    }

                    # Match original storage structure exactly
                    data = {
                        'from_date': from_date,
                        'to_date': to_date,
                        'batch_results': [batch_result],
                        'total_trades': len(merged),
                        'symbol_history': merged
                    }
                    save_profit_dict_row(symbol, interval, data)

                gc.collect()

        clear_checkpoint()

        if verbose:
            print("\n" + "=" * 60)
            print("COMPLETE!")
            print("=" * 60)

        return True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
