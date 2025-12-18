from functions.data_utils import (get_last_and_start_dates_candle_data_for_symbols,
                                  get_nearest_expiry_commodities, filter_symbol_data, is_within_time_range)
from functions.tradingview_fetcher import process_symbol_pair_tradingview
from functions.angelone_fetcher import process_symbol_pair_angelone
from functions.process_indicators_data import process_indicators_data
from functions.save_candle_data import save_candle_data
from functions.save_all_indicator_data_to_timescaledb import save_all_indicator_data_by_interval
from concurrent.futures import ThreadPoolExecutor
from datetime import time
import time as time_lib
import logging
import gc

def fetch_candle_data_indices(comm_time, symbols_list, intervals, allowed_intervals):
    print("fetch_candle_data_indices")
    max_workers = len(symbols_list)
    for interval in intervals:
        logging.info(f"Processing interval: {interval}")
        if interval not in allowed_intervals:
            logging.info(f"Interval: {interval} not in allowed intervals")
            continue
        last_processed_dates = get_last_and_start_dates_candle_data_for_symbols([s for s in symbols_list], interval)
        if not last_processed_dates:
            continue
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
                indicator_results = list(executor.map(process_indicators_data, indicator_processing_data))
            symbol_interval_data = list(zip(symbol_list, [interval] * len(symbol_list), indicator_results))
            save_all_indicator_data_by_interval(symbol_interval_data)

        gc.collect()

    return True

def fetch_candle_data_commodities(symbols_list, intervals, allowed_intervals, all_symbols_data, login_details_list):
    print("fetch_candle_data_commodities")
    commodities_symbols = get_nearest_expiry_commodities(all_symbols_data, symbols_list)
    max_workers = len(symbols_list)
    for interval in intervals:
        logging.info(f"Processing interval: {interval}")
        if interval not in allowed_intervals:
            logging.info(f"Interval: {interval} not in allowed intervals")
            continue

        last_processed_dates = get_last_and_start_dates_candle_data_for_symbols([s for s in symbols_list], interval)
        if not last_processed_dates:
            continue
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
                indicator_results = list(executor.map(process_indicators_data, indicator_processing_data))
            symbol_interval_data = list(zip(symbol_list, [interval] * len(symbol_list), indicator_results))
            save_all_indicator_data_by_interval(symbol_interval_data)

        gc.collect()

    return True

def process_lower_interval_equities(
        wave_number, interval, wave_size, last_processed_date_candle_data_dict, comm_time, wave_cooldown,
        symbol_processing_start_time, max_workers, symbols_list
):
    results_cache = {}
    retry_counts = {}

    while symbols_list:
        wave_number += 1
        logging.info(f"Processing {len(symbols_list)} symbols for interval {interval}")

        # Process symbols in waves
        all_wave_results = []
        for wave_start in range(0, len(symbols_list), wave_size):
            wave_end = min(wave_start + wave_size, len(symbols_list))
            wave_symbols = symbols_list[wave_start:wave_end]

            logging.info(
                f"Wave {wave_number}: Processing symbols {wave_start + 1}-{wave_end} of {len(symbols_list)}")

            # Prepare args for this wave
            wave_args = []
            for symbol in wave_symbols:
                args = (symbol, interval, last_processed_date_candle_data_dict, comm_time)
                wave_args.append(args)

            # Process this wave in parallel (5 workers max for the wave)
            with ThreadPoolExecutor(max_workers=min(wave_size, len(wave_args))) as executor:
                futures = [executor.submit(process_symbol_pair_tradingview, args) for args in wave_args]
                wave_results = [future.result() for future in futures]

            all_wave_results.extend(wave_results)

            # Cooldown between waves (except for the last wave)
            if wave_end < len(symbols_list):
                logging.info(f"Wave cooldown: waiting {wave_cooldown}s before next wave...")
                time_lib.sleep(wave_cooldown)

        # Process results from all waves
        failed_symbols = []
        for idx, result in enumerate(all_wave_results):
            symbol_name = symbols_list[idx]

            if result is None:
                continue
            if result is False:
                retry_counts[symbol_name] = retry_counts.get(symbol_name, 0) + 1
                if retry_counts[symbol_name] < 3:
                    logging.error(
                        f"Error processing symbol {symbol_name}. Retry count: {retry_counts[symbol_name]}")
                    failed_symbols.append(symbol_name)
                else:
                    logging.error(
                        f"Symbol {symbol_name} failed after {retry_counts[symbol_name]} retries. Skipping.")
            else:
                results_cache[symbol_name] = result

        if failed_symbols:
            logging.info(f"Retrying for {len(failed_symbols)} failed symbols after a short sleep.")
            time_lib.sleep(2)
        symbols_list = failed_symbols

    symbol_processing_end_time = time_lib.time()
    logging.info(f"Symbol processing took: {symbol_processing_end_time - symbol_processing_start_time:.2f} seconds")
    # --- End Symbol Processing ---

    # Collect results
    bulk_candle_data = []
    indicator_processing_data = []
    symbol_list = []
    for symbol, result in results_cache.items():
        indicator_data = result['indicator_data']
        bulk_candle_data.extend(result['bulk_data'])
        indicator_processing_data.append(indicator_data)
        symbol_list.append(symbol)
    # --- Candle Data Saving ---
    if bulk_candle_data:
        logging.info(f"Saving {len(bulk_candle_data)} candle records")
        save_candle_data(bulk_candle_data)
        logging.info("Candle data saved")
        del bulk_candle_data
    # --- End Candle Data Saving ---

    # --- Indicator Processing and Saving ---
    if indicator_processing_data:
        logging.info("Processing indicator data")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indicator_results = list(executor.map(process_indicators_data, indicator_processing_data))
        del indicator_processing_data
        symbol_interval_data = list(zip(symbol_list, [interval] * len(symbol_list), indicator_results))
        del indicator_results
        logging.info("Saving all indicator data by interval")
        save_all_indicator_data_by_interval(symbol_interval_data)
        logging.info("Indicator data saved")
        del symbol_interval_data
    # --- End Indicator Processing and Saving ---

    gc.collect()
    del results_cache


def process_higher_interval_equities(all_symbols_data, max_workers, symbols_list, login_details_list,
                                     interval, last_processed_date_candle_data_dict):
    equity_symbol_dicts = filter_symbol_data(all_symbols_data)
    results_cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        if symbols_list:
            logging.info(f"Processing {len(equity_symbol_dicts)} commodities via Broker API.")
            num_users = len(login_details_list)
            broker_args = [
                (s, login_details_list[i % num_users], interval, last_processed_date_candle_data_dict,
                 interval, login_details_list)
                for i, s in enumerate(equity_symbol_dicts)
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
            indicator_results = list(executor.map(process_indicators_data, indicator_processing_data))
        symbol_interval_data = list(zip(symbol_list, [interval] * len(symbol_list), indicator_results))
        save_all_indicator_data_by_interval(symbol_interval_data)

    gc.collect()


def fetch_candle_data_equities(comm_time, symbols_list, intervals, allowed_intervals,
                               all_symbols_data, login_details_list):
    print("fetch_candle_data_equities")
    if not symbols_list:
        logging.warning("No equity symbols provided, skipping equity data fetch.")
        return True
    max_workers = 8
    wave_size = 8
    wave_cooldown = 1.5
    wave_number = 0
    symbol_processing_start_time = time_lib.time()

    for interval in intervals:
        logging.info(f"Processing interval: {interval}")
        if interval not in allowed_intervals:
            logging.info(f"Interval: {interval} not in allowed intervals")
            continue
        last_processed_date_candle_data_dict = get_last_and_start_dates_candle_data_for_symbols(
            [s for s in symbols_list], interval)
        if not last_processed_date_candle_data_dict:
            continue
        if interval == 'ONE_MINUTE' or 'FIVE_MINUTE':
            process_lower_interval_equities(
                wave_number, interval, wave_size, last_processed_date_candle_data_dict, comm_time, wave_cooldown,
                symbol_processing_start_time, max_workers, symbols_list
            )
        else:
            process_higher_interval_equities(all_symbols_data, max_workers, symbols_list, login_details_list,
                                             interval, last_processed_date_candle_data_dict)
    return True

def fetch_candle_data_post_market(config, comm_time, intervals, all_symbols_data, login_details_list,
                                  equity_symbols=None):
    print("fetch_candle_data_post_market")
    is_equity_market_hours = is_within_time_range(time(16, 0), time(18, 0),
                                                  comm_time)
    is_commodity_market_hours = is_within_time_range(time(00, 0), time(2, 0),
                                                     comm_time)

    if is_equity_market_hours:
        indices_list = config['indices_list']
        allowed_intervals = config['allowed_indices_intervals']
        fetch_candle_data_indices(comm_time, indices_list, intervals, allowed_intervals)
        equity_list = equity_symbols
        allowed_intervals = config['allowed_equity_intervals']
        fetch_candle_data_equities(comm_time, equity_list, intervals, allowed_intervals,
                                   all_symbols_data, login_details_list)

    if is_commodity_market_hours:
        commodities_list = config['commodities_list']
        allowed_intervals = config['allowed_commodities_intervals']
        fetch_candle_data_commodities(commodities_list, intervals, allowed_intervals, all_symbols_data,
                                      login_details_list)

    return True
