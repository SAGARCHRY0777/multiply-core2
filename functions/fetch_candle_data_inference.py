from functions.data_utils import (get_last_and_start_dates_candle_data_for_symbols,
                                  get_nearest_expiry_commodities)
from functions.tradingview_fetcher import process_symbol_pair_tradingview
from functions.angelone_fetcher import process_symbol_pair_angelone
from functions.process_indicators_data_live import process_indicators_data_live
from functions.save_candle_data import save_candle_data
from functions.save_all_indicator_data_to_timescaledb import save_all_indicator_data_by_interval
from concurrent.futures import ThreadPoolExecutor
import logging
import gc

def fetch_candle_data_angelone(symbols_list, interval, login_details_list, all_symbols_data):
    print("fetch_candle_data_angelone")
    max_workers = len(symbols_list)
    logging.info(f"Processing interval: {interval}")
    commodities_symbols = get_nearest_expiry_commodities(all_symbols_data, symbols_list)

    last_processed_dates = get_last_and_start_dates_candle_data_for_symbols([s for s in symbols_list], interval)
    if not last_processed_dates:
        return True
    results_cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        if symbols_list:
            logging.info(f"Processing {len(commodities_symbols)} commodities via Broker API.")
            num_users = len(login_details_list)
            broker_args = [
                (s, login_details_list[i % num_users], interval, last_processed_dates, interval, login_details_list)
                for i, s in enumerate(commodities_symbols)
            ]
            futures.extend([executor.submit(process_symbol_pair_angelone, arg) for arg in broker_args])

            # Process results as they complete
        for future in futures:
            result = future.result()
            if result and result.get('symbol'):
                results_cache[result['symbol']] = result

    # Aggregate and save data from both sources
    bulk_candle_data = [item for res in results_cache.values() for item in res['bulk_data']]

    indicator_processing_data = [res['indicator_data'] for res in results_cache.values()]
    symbol_list = list(results_cache.keys())
    if bulk_candle_data:
        save_candle_data(bulk_candle_data)

    if indicator_processing_data:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indicator_results = list(executor.map(process_indicators_data_live, indicator_processing_data))
        symbol_interval_data = list(zip(symbol_list, [interval] * len(symbol_list), indicator_results))
        save_all_indicator_data_by_interval(symbol_interval_data)

    gc.collect()

    return True

def fetch_candle_data_tradingview(symbols_list, interval, comm_time):
    print("fetch_candle_data_tradingview")
    max_workers = len(symbols_list)
    logging.info(f"Processing interval: {interval}")
    last_processed_dates = get_last_and_start_dates_candle_data_for_symbols([s for s in symbols_list], interval)
    if not last_processed_dates:
        return True
    results_cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        if symbols_list:
            logging.info(f"Processing {len(symbols_list)} indices via TradingView.")
            tv_args = [(s, interval, last_processed_dates, comm_time) for s in symbols_list   ]
            futures.extend([executor.submit(process_symbol_pair_tradingview, arg) for arg in tv_args])

        for future in futures:
            result = future.result()
            if result and result.get('symbol'):
                results_cache[result['symbol']] = result
    bulk_candle_data = [item for res in results_cache.values() for item in res['bulk_data']]
    indicator_processing_data = [res['indicator_data'] for res in results_cache.values()]
    symbol_list = list(results_cache.keys())
    if bulk_candle_data:
        save_candle_data(bulk_candle_data)

    if indicator_processing_data:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indicator_results = list(executor.map(process_indicators_data_live, indicator_processing_data))
        symbol_interval_data = list(zip(symbol_list, [interval] * len(symbol_list), indicator_results))
        save_all_indicator_data_by_interval(symbol_interval_data)

    gc.collect()

    return True

def fetch_candle_data_inference(symbols_list, base_interval, comm_time, inference_update_intervals,
                                commodities_list, commodities_label, login_details_list,
                                all_symbols_data):
    print("fetch_candle_data_inference")
    update_intervals = inference_update_intervals[base_interval]
    if not update_intervals:
        return True

    if commodities_label is True and any(sym in commodities_list for sym in symbols_list) is True:
        selected_commodity_symbols = []
        selected_non_commodity_symbols = []

        for sym in symbols_list:
            if sym in commodities_list:
                selected_commodity_symbols.append(sym)
            else:
                selected_non_commodity_symbols.append(sym)

        if selected_commodity_symbols:
            fetch_candle_data_angelone(selected_commodity_symbols, base_interval, login_details_list, all_symbols_data)
        fetch_candle_data_tradingview(selected_non_commodity_symbols, base_interval, comm_time)
    else:
        fetch_candle_data_tradingview(symbols_list, base_interval, comm_time)

    return True
