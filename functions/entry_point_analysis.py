
from functions.data_utils import (is_within_time_range, get_nearest_expiry_commodities,
                                  filter_symbol_data, preprocess_data, get_symbol_dict,
                                  filter_profit_data, clear_memory)
from functions.rule_based_trade_filter_v1 import rule_based_trade_filter
from functions.handle_processed_trades import (load_processed_trades, filter_expired_trades, clear_processed_trades,
                                               save_processed_trades, check_and_register_trade)
from functions.email_utils import send_email_with_error_report, send_email
from functions.multiply_stoploss_calculator import multiply_stoploss_calculator
from functions.automatic_buy_order_handling import automatic_buy_order_handling
from datetime import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

def analysis_symbol_selection(config, is_equity_market_hours, is_commodity_market_hours, all_symbols_data, intervals, equity_symbols=None):
    analysis_symbol_data_list = []

    if config['indices'] is True and is_equity_market_hours:
        indices_tokens_list = config['indices_tokens_list']
        indices_symbol_dicts = [
            s for s in all_symbols_data
            if s['token'] in indices_tokens_list
        ]
        analysis_symbol_data_list.extend(indices_symbol_dicts)

    if config['commodities'] is True and is_commodity_market_hours:
        commodities_list = config['commodities_list']
        commodities_symbol_dicts = get_nearest_expiry_commodities(all_symbols_data, commodities_list)
        analysis_symbol_data_list.extend(commodities_symbol_dicts)
        logging.info(f"Added {len(commodities_symbol_dicts)} commodity symbol dicts for processing")

    if config['equity'] is True and is_equity_market_hours:
        equity_list = equity_symbols
        equity_symbol_dicts = filter_symbol_data(all_symbols_data)
        if intervals == ['FIVE_MINUTE']:
            nifty_fifty_symbol_dicts = [
                sd for sd in equity_symbol_dicts
                if sd['name'] in equity_list
            ]
            analysis_symbol_data_list.extend(nifty_fifty_symbol_dicts)
        else:
            analysis_symbol_data_list.extend(equity_symbol_dicts)

    seen_names = set()
    deduplicated_symbol_dicts = []
    for sd in analysis_symbol_data_list:
        if sd['name'] not in seen_names:
            deduplicated_symbol_dicts.append(sd)
            seen_names.add(sd['name'])
    analysis_symbol_data_list = deduplicated_symbol_dicts

    logging.info(f"Total unique symbols to process: {len(analysis_symbol_data_list)}")

    return analysis_symbol_data_list

def run_symbol_analysis(args):
    interval, symbol_dict, stock_df, macd_df, additional_inputs, config = args
    analysis_symbol = symbol_dict['name']
    analysis_token = symbol_dict['token']
    try:
        bullish_average_sell_point = additional_inputs['bullish_average_sell_point']
        bearish_average_sell_point = additional_inputs['bearish_average_sell_point']
        commodities_list = config['commodities_list']
        indices_list = config['indices_list']
        if analysis_symbol in commodities_list or indices_list:
            target_profit_percentage = config['commodities_indices_profit_percentage'][interval]
        else:
            target_profit_percentage = config['equity_profit_percentage'][interval]
        result = rule_based_trade_filter(stock_df, macd_df, analysis_symbol, analysis_token, interval,
                                         bullish_average_sell_point, bearish_average_sell_point,
                                         target_profit_percentage)
        return result
    except Exception as e:
        logging.error(f"Error analyzing symbol {analysis_symbol} in interval {interval}: {e}")
        return None

def entry_point_analysis(config, comm_time, intervals, all_symbols_data, profit_dict, user_db,
                         equity_symbols=None):
    print("entry_point_analysis")
    try:
        is_equity_market_hours = is_within_time_range(time(9, 15), time(14, 30), comm_time)
        is_commodity_market_hours = is_within_time_range(time(9, 0), time(22, 30), comm_time)
        # analysis symbol selection
        indices_list = config['indices_list']
        commodities_list = config['commodities_list']
        analysis_symbol_data_list = analysis_symbol_selection(config, is_equity_market_hours, is_commodity_market_hours,
                                                              all_symbols_data, intervals, equity_symbols)
        symbol_count = len(analysis_symbol_data_list)
        # --- Interval-based Processed Trades Cleanup ---
        if comm_time.minute in (0, 15, 30, 45):
            logging.info("Running 15-minute cleanup of processed trades list.")
            processed_trades = load_processed_trades()
            if processed_trades:
                # Filter expired trades based on interval-specific expiration
                filtered_trades = filter_expired_trades(processed_trades, comm_time)
                clear_processed_trades()  # Clear the old file
                save_processed_trades(filtered_trades)  # Save the fresh list
                logging.info(f"Cleaned processed trades list. {len(filtered_trades)} trades remain.")
        processed_trades = load_processed_trades()

        # data fetching
        preprocessed_data = preprocess_data(intervals, analysis_symbol_data_list)
        # trade analysis
        for interval, interval_data in preprocessed_data.items():
            args_list = []
            for symbol, data in interval_data.items():
                logging.info(f"SYMBOL: {symbol}")
                symbol_dict = get_symbol_dict(all_symbols_data, symbol)
                symbol_specific_profit_data = filter_profit_data(profit_dict, symbol, interval,
                                                                 indices_list,
                                                                 commodities_list)
                additional_inputs = {
                    'bullish_average_sell_point': symbol_specific_profit_data.get('bullish_average_sell_point', 'target'),
                    'bearish_average_sell_point': symbol_specific_profit_data.get('bearish_average_sell_point', 'target'),
                    'bullish_average_profit_percentage': symbol_specific_profit_data.get('bullish_average_profit_percentage', 0),
                    'bearish_average_profit_percentage': symbol_specific_profit_data.get('bearish_average_profit_percentage', 0)
                }

                # Prepare arguments for multiprocessing
                args_list.append((
                    interval,
                    symbol_dict,
                    data['stock_df'],
                    data['macd_df'],
                    additional_inputs,
                    config,
                ))

            # Analyze symbols in parallel using ThreadPoolExecutor (more compatible with systemd)
            results = []
            n_procs = min(8, symbol_count) if symbol_count > 0 else 1
            with ThreadPoolExecutor(max_workers=n_procs) as executor:
                futures = [executor.submit(run_symbol_analysis, args) for args in args_list]
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Error in symbol analysis: {e}")
            selected_results = [result for result in results if result.get('result') is True]
            
            # Deduplicate by divergence_date + setup_used (same divergence = same trade)
            seen_divergences = set()
            deduped_results = []
            for result in selected_results:
                analysis_data = result.get('analysis_data', {})
                div_date = analysis_data.get('divergence_date')
                setup = result.get('setup_used')
                key = (str(div_date), setup)
                if key not in seen_divergences:
                    seen_divergences.add(key)
                    deduped_results.append(result)
                else:
                    logging.info(f"Skipping duplicate divergence signal: {result['symbol']}/{setup} from {div_date}")
            selected_results = deduped_results
            logging.info(f"After divergence deduplication: {len(selected_results)} unique signals")
            
            analysis_candle_datetime = None
            if selected_results:
                bullish_symbols = []
                bearish_symbols = []
                signals_available = []
                for result in selected_results:
                    if result.get('result') is True:
                        symbol = result['symbol']
                        setup_used = result.get('setup_used')

                        # Get last close price for the trade
                        analysis_data = result.get('analysis_data', {})
                        analysis_candle_datetime = analysis_data["date"]
                        last_close_price = analysis_data.get('close') if analysis_data else None
                        # Check if trade should be processed and register it
                        if check_and_register_trade(symbol, interval, setup_used, processed_trades, last_close_price):
                            if setup_used == 'bullish_momentum_setup':
                                bullish_symbols.append(symbol)
                                stoploss_result = multiply_stoploss_calculator(
                                    symbol=analysis_data['analysis_symbol'],
                                    interval=interval,
                                    entry_timestamp=analysis_data['date'],
                                    direction='bullish',
                                )
                                # Check if stoploss calculation returned valid result
                                if stoploss_result is None or stoploss_result.get('stoploss') is None:
                                    logging.warning(f"Stoploss calculation returned None for {symbol}, skipping trade")
                                    continue
                                stoploss = float(stoploss_result['stoploss'])
                                signal_result = {
                                    "analysis_symbol": analysis_data['analysis_symbol'],
                                    "analysis_token": analysis_data['analysis_token'],
                                    "interval": interval,
                                    "buy_indicator": analysis_data['buy_indicator'],
                                    "analysis_last_close_price": analysis_data['analysis_last_close_price'],
                                    "date": analysis_data['date'],
                                    "volume": analysis_data['volume'],
                                    "higher_interval_divergence": analysis_data['higher_interval_divergence'],
                                    "target": analysis_data['target'],
                                    "analysis_profit_percentage": analysis_data.get('analysis_profit_percentage', 0),
                                    "stoploss": stoploss
                                }
                                signals_available.append(signal_result)
                                logging.info(f"Registered bullish trade for {symbol}")

                            elif setup_used == 'bearish_momentum_setup':
                                bearish_symbols.append(symbol)
                                stoploss_result = multiply_stoploss_calculator(
                                    symbol=analysis_data['analysis_symbol'],
                                    interval=interval,
                                    entry_timestamp=analysis_data['date'],
                                    direction='bearish',
                                )
                                # Check if stoploss calculation returned valid result
                                if stoploss_result is None or stoploss_result.get('stoploss') is None:
                                    logging.warning(f"Stoploss calculation returned None for {symbol}, skipping trade")
                                    continue
                                stoploss = float(stoploss_result['stoploss'])

                                signal_result = {
                                    "analysis_symbol": analysis_data['analysis_symbol'],
                                    "analysis_token": analysis_data['analysis_token'],
                                    "interval": interval,
                                    "buy_indicator": analysis_data['buy_indicator'],
                                    "analysis_last_close_price": analysis_data['analysis_last_close_price'],
                                    "date": analysis_data['date'],
                                    "volume": analysis_data['volume'],
                                    "higher_interval_divergence": analysis_data['higher_interval_divergence'],
                                    "target": analysis_data['target'],
                                    "analysis_profit_percentage": analysis_data['analysis_profit_percentage'],
                                    "stoploss": stoploss
                                }
                                signals_available.append(signal_result)
                                logging.info(f"Registered bearish trade for {symbol}")
                        else:
                            logging.info(f"Skipping already processed trade: {symbol}/{interval}/{setup_used}")

                logging.info(f"Selected Bullish Symbols: {bullish_symbols}")
                logging.info(f"Selected Bearish Symbols: {bearish_symbols}")
                if bullish_symbols:
                    message = f"Bullish Selected Symbol: {bullish_symbols}, Interval: {interval}, Date: {analysis_candle_datetime}, Created At: {comm_time}"
                    send_email(message, True)
                if bearish_symbols:
                    message = f"Bearish Selected Symbol: {bullish_symbols}, Interval: {interval}, Date: {analysis_candle_datetime}, Created At: {comm_time}"
                    send_email(message, True)
                automatic_buy_order_handling(user_db, interval, signals_available, all_symbols_data, config)
    except Exception as e:
        logging.error(f"Error during symbol analysis: {e}", exc_info=True)
        send_email_with_error_report(traceback.format_exc())

    finally:
        # Final cleanup
        clear_memory()
        logging.info("Completed analysis")
