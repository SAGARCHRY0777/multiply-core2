import logging
import concurrent.futures
from functions.data_utils import (extract_login_details, calculate_quantities,
                                  prep_order_params)
from functions.SO_angel_v2.angel_one_functions import (symbol_ltp_data, get_funds_and_margins_info,
                                                       get_order_book, get_trade_book)
from functions.get_itm_symbol_buy_info import get_itm_symbol_buy_info, safe_quote_data
from functions.upload_trade_dict_after_three_thirty_dict import upload_tradedict_afterthreethirty, is_successful_trade_in_trade_data
from functions.order_placement_utils import buy_equity, buy_futures, buy_options_atm, sell_futures
from functions.upload_analysis_data import upload_analysis_data
from functions.calculate_time_stoploss import calculate_time_stoploss
from functions.upload_i_dict import upload_i_dict
from functions.fetch_template_details import fetch_template_details
from datetime import datetime
from pytz import timezone

ist = timezone('Asia/Kolkata')

def process_single_user(args):
    user_id, context, symbol_buy_info, ltp = args
    # Check funds
    user_db = context['user_db']
    login_details = extract_login_details(user_db, user_id)
    config = context['config']
    
    # Get selected_template_id from login_details
    selected_template_id = login_details.get('selected_template_id')
    
    # Fetch user-specific template settings using selected_template_id
    user_template = fetch_template_details(user_id, template_id=selected_template_id)
    if user_template:
        # Use per-user settings if available
        segment = user_template.get('segment') or config.get('segment', 'Options')
        withdraw_amount = float(user_template.get('withdraw_amount') or config.get('withdraw_amount', 0))
        maximum_buy_percentage = user_template.get('maximum_buy_percentage') or config.get('maximum_buy_percentage')
        entry_mod = user_template.get('entry_mod') or config.get('entry_mod')
        exit_mod = user_template.get('exit_mod') or config.get('exit_mod')
        logging.info(f"Using per-user template for {user_id}: segment={segment}, entry_mod={entry_mod}, exit_mod={exit_mod}")
    else:
        # Fallback to global config
        segment = context.get('segment', config.get('segment', 'Options'))
        withdraw_amount = float(config.get('withdraw_amount', 0))
        maximum_buy_percentage = config.get('maximum_buy_percentage')
        entry_mod = config.get('entry_mod')
        exit_mod = config.get('exit_mod')
        logging.debug(f"No template found for {user_id}, using global config")


    try:
        funds_info = get_funds_and_margins_info(login_details)
        available_funds = float(funds_info['data']['net'])
        if available_funds < withdraw_amount:
            logging.warning(f"Insufficient funds for user {user_id}. Available: {available_funds}, Withdraw: {withdraw_amount}")
            return user_id, None
    except Exception as e:
        logging.error(f"Error checking funds for user {user_id}: {e}")
        return user_id, None

    if symbol_buy_info is None:
        logging.warning(f"Could not find ITM symbol for {context['analysis_symbol']}")
        return user_id, None

    # Lot size and quantity calculation
    lot_size = float(symbol_buy_info.get('lotsize', 1))
    
    if str(maximum_buy_percentage).lower() == "none":
        if float(available_funds) < float(500000):
            maximum_buy_percentage = float(100)
        else:
            maximum_buy_percentage = float(50)
    else:
        maximum_buy_percentage = float(maximum_buy_percentage)
        
    quantity, disclosed_quantity, per_lot_buy_amount = calculate_quantities(
        funds_info, withdraw_amount, lot_size, ltp,
        maximum_buy_percentage)
        
    if quantity == 0:
        return user_id, None
        
    order_params = prep_order_params(symbol_buy_info['symbol'], symbol_buy_info['token'], quantity,
                                     symbol_buy_info.get('exch_seg', context['exchange_segment']), segment)

    buy_indicator = context['buy_indicator']
    exchange_segment = context['exchange_segment']

    if entry_mod == 'auto':
        order_id = None
        # Place the appropriate order based on segment and buy indicator
        if segment == "Options" and buy_indicator == "bullish_momentum_setup":
            order_id, order_params = buy_options_atm(
                login_details, symbol_buy_info['symbol'],
                symbol_buy_info['token'], quantity, symbol_buy_info.get('exch_seg', exchange_segment))
        elif segment == "Options" and buy_indicator == "bearish_momentum_setup":
            order_id, order_params = buy_options_atm(
                login_details, symbol_buy_info['symbol'],
                symbol_buy_info['token'], quantity, symbol_buy_info.get('exch_seg',
                                                                                exchange_segment))
        elif segment == "Futures" and buy_indicator == "bullish_momentum_setup":
            order_id, order_params = buy_futures(
                login_details, symbol_buy_info['symbol'],
                symbol_buy_info['token'], quantity, symbol_buy_info.get('exch_seg',
                                                                                exchange_segment))
        elif segment == "Futures" and buy_indicator == "bearish_momentum_setup":
            order_id, order_params = sell_futures(
                login_details, symbol_buy_info['symbol'],
                symbol_buy_info['token'], quantity, symbol_buy_info.get('exch_seg',
                                                                                exchange_segment))
        elif segment == 'Equity' and buy_indicator == "bullish_momentum_setup":
            order_id, order_params = buy_equity(
                login_details, symbol_buy_info['symbol'],
                symbol_buy_info['token'], quantity, symbol_buy_info.get('exch_seg',
                                                                                exchange_segment),
                disclosed_quantity)
        else:
            logging.warning(f"Unsupported segment or buy indicator for user {user_id}")
            return user_id, None
            
        if order_id is None:
            return user_id, None

        # Verify order status
        order_book = get_order_book(login_details)
        if order_book and order_book.get('data'):
            for order in order_book['data']:
                if order['orderid'] == order_id:
                    if order['orderstatus'] != 'rejected':
                        print(f"ORDER STATUS: {order['orderstatus']}")
                    else:
                        return user_id, None
                else:
                    continue
        
        # Get trade details
        trade_book = get_trade_book(login_details)
        order_details = {}
        if trade_book and trade_book.get('data'):
            for order in trade_book['data']:
                if order['orderid'] == order_id:
                    order_details = order
                    break

        buy_price = order_details.get('fillprice', 0)

        trade_dict = {
            'user_id': user_id,
            'analysis_symbol': context['analysis_symbol'],
            'interval': context['interval'],
            'order_id': order_id,
            'order_variety': order_params['variety'],
            'tradingsymbol': order_params['tradingsymbol'],
            'symbol_token': order_params['symboltoken'],
            'transaction_type': order_params['transactiontype'],
            'start_datetime': datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S'),
            'exchange': order_params['exchange'],
            'product_type': order_params['producttype'],
            'duration': order_params['duration'],
            'quantity': order_params['quantity'],
            'trade_mode': "Auto",
            'buy_price': buy_price,
            'order_status': 'BOUGHT'
        }

        after_three_thirty_dict = {
            'user_id': user_id,
            'analysis_symbol': context['analysis_symbol'],
            'interval': context['interval'],
            'order_id': order_id,
            'tradingsymbol': order_params['tradingsymbol'],
            'product_type': order_params['producttype'],
            'quantity': order_params['quantity'],
            'exchange': order_params['exchange'],
            'symbol_token': order_params['symboltoken'],
            'transaction_type': order_params['transactiontype'],
            'start_datetime': datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_profit_percentage': context["analysis_profit_percentage"],
            'buy_indicator': buy_indicator,
            'trade_mode': "Auto",
            'buy_price': buy_price,
            'order_status': 'BOUGHT',
            'sector': "None"
        }
        upload_tradedict_afterthreethirty(trade_dict, "trade_data")
        upload_tradedict_afterthreethirty(after_three_thirty_dict,
                                          "after_three_thirty_data")

        return user_id, order_params

    return user_id, order_params


def automatic_buy_order_handling(user_db, interval, signals_available,
                                 all_symbols_data, config):
    print("automatic_buy_order_handling")
    logging.info("=" * 50)
    logging.info("Starting Buy Code Execution")
    logging.info("=" * 50)
    segment = config['segment']
    active_users = [u for u in user_db if u['status'] == 'active']
    
    for selected_symbol_dict in signals_available:
        buy_indicator = selected_symbol_dict["buy_indicator"]
        analysis_symbol = selected_symbol_dict["analysis_symbol"]

        logging.info(f"Analyzing symbol: {analysis_symbol}, Buy Indicator: {buy_indicator}")
        
        if analysis_symbol in config['commodities_list']:
            analysis_exchange = "MCX"
        else:
            analysis_exchange = "NSE"
            
        login_details = extract_login_details(user_db, user_db[0]['user_id'])
        
        # 1. Fetch LTP for Analysis Symbol
        ltp_data = symbol_ltp_data(analysis_exchange, analysis_symbol, selected_symbol_dict["analysis_token"],
                                   initial_login_details=login_details, login_details_list=user_db)
        logging.info(ltp_data)
        if ltp_data is None:
            logging.warning(f"Failed to retrieve LTP data for {analysis_symbol}")
            continue

        ltp = ltp_data['data']['ltp']
        if ltp is None:
            logging.warning(f"LTP data not available for {analysis_symbol}")
            continue

        if segment == 'Equity':
            exchange_segment = 'NSE'
        elif segment == 'Futures':
            exchange_segment = 'NFO'
        else:
            exchange_segment = 'NFO'

        # Reference Percentage Diff Check
        last_close_price = selected_symbol_dict['analysis_last_close_price']
        ltp = float(ltp)
        last_close_price = float(last_close_price)
        percentage_difference = ((ltp - last_close_price) / last_close_price) * 100
        logging.info(f"LTP: {ltp}, Last Close Price: {last_close_price}, Percentage Difference: "
                    f"{percentage_difference}")
        percentage_difference = float(percentage_difference)

        # Determine symbol type and get appropriate threshold from config
        thresholds = config.get('percentage_difference_thresholds', {})
        if analysis_symbol in config.get('commodities_list', []):
            threshold = thresholds.get('commodities', 0.04)
            symbol_type = 'commodities'
        elif analysis_symbol in config.get('indices_list', []):
            threshold = thresholds.get('indices', 0.03)
            symbol_type = 'indices'
        else:
            threshold = thresholds.get('equity', 0.5)
            symbol_type = 'equity'

        if abs(percentage_difference) > threshold:
            logging.info(f"Percentage difference for {analysis_symbol} ({symbol_type}) exceeds {threshold}%. Skipping.")
            continue

        context = {
            'buy_indicator': buy_indicator,
            'user_db': user_db,
            'analysis_symbol': analysis_symbol,
            'all_symbols_data': all_symbols_data,
            'segment': segment,
            'selected_symbol_dict': selected_symbol_dict,
            'exchange_segment': exchange_segment,
            'interval': interval,
            'config': config,
            'analysis_last_close_price': selected_symbol_dict['analysis_last_close_price'],
            'analysis_profit_percentage': selected_symbol_dict['analysis_profit_percentage']
        }
        
        # 2. Fetch ITM Symbol & Quote Data ONCE
        logging.info(f"Fetching ITM Symbol info for {analysis_symbol}...")
        symbol_buy_info = get_itm_symbol_buy_info(context['analysis_symbol'], context['buy_indicator'],
                                                  all_symbols_data, segment,
                                                  context['analysis_last_close_price'], user_db)

        if symbol_buy_info is None:
            logging.warning(f"Could not find ITM symbol for {context['analysis_symbol']}")
            continue

        tokens = [symbol_buy_info['token']]
        exch = symbol_buy_info.get('exch_seg', context['exchange_segment'])
        
        quote_request = {
            "mode": "FULL",
            "exchangeTokens": {
                exch: tokens
            }
        }
        
        if not isinstance(user_db, (list, tuple)) or len(user_db) < 2:
            raise ValueError("Invalid user_db format")
            
        logging.info(f"Fetching Quote data for {symbol_buy_info['symbol']}...")
        quote_response = safe_quote_data(quote_request['mode'], quote_request['exchangeTokens'], user_db)
        
        if quote_response is None:
            logging.warning(f"Failed to retrieve quote data for {symbol_buy_info['symbol']}")
            continue

        trade_instrument_ltp = 0
        if quote_response.get('status') and quote_response.get('data'):
            option_list = quote_response['data'].get('fetched', [])
            if option_list:
                option = option_list[0]
                depth = option.get('depth', {})
                depth_sell = depth.get('sell', [])
                if not depth_sell:
                    logging.warning(f"No sell depth data for {symbol_buy_info['symbol']}")
                    continue
                trade_instrument_ltp = depth_sell[0]['price']
                print(f"Trade Instrument LTP: {trade_instrument_ltp}")
            else:
                logging.warning("No fetched data in quote response")
                continue
        else:
            logging.warning(f"Quote response invalid status")
            continue
            
        # 3. Parallel Execution
        order_params_list = []
        success_user_id_list = []
        
        logging.info(f"Starting parallel order placement for {len(active_users)} users")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_users)) as executor:
            future_to_user = {}
            for user in active_users:
                user_id = user['user_id']
                args = (user_id, context, symbol_buy_info, trade_instrument_ltp)
                future = executor.submit(process_single_user, args)
                future_to_user[future] = user_id
                
            for future in concurrent.futures.as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    uid, params = future.result()
                    if params:
                        order_params_list.append({uid: params})
                    
                    if is_successful_trade_in_trade_data(user_id, analysis_symbol) is True:
                        success_user_id_list.append(user_id)
                        
                except Exception as exc:
                    logging.error(f"User {user_id} generated an exception: {exc}")

        # 4. Upload Analysis Data / I_dict
        if order_params_list:
            analysis_dict = {
                "date": selected_symbol_dict['date'],
                "interval": selected_symbol_dict["interval"],
                "buy_indicator": selected_symbol_dict["buy_indicator"],
                "analysis_symbol": selected_symbol_dict["analysis_symbol"],
                "analysis_last_close_price": selected_symbol_dict["analysis_last_close_price"],
                "analysis_token": selected_symbol_dict["analysis_token"],
                "type_one": True,
                "nifty": False,
                "higher_interval_divergence": selected_symbol_dict["higher_interval_divergence"],
                "stoploss": selected_symbol_dict["stoploss"],
                "volume": selected_symbol_dict["volume"],
                "target": selected_symbol_dict["target"],
                "order_params": order_params_list
            }
            upload_analysis_data(interval, analysis_dict)

        if success_user_id_list:
            buy_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            commodities_list = config['commodities_list']
            indices_list = config['indices_list']
            time_stoploss = calculate_time_stoploss(analysis_symbol, interval, buy_time, commodities_list, indices_list)

            i_dict = {
                'analysis_symbol': analysis_symbol,
                'interval': interval,
                'target_exit_price': selected_symbol_dict['target'],
                'time_stoploss': time_stoploss,
                'buy_datetime': buy_time,  # For time decay calculation
                'buy_indicator': buy_indicator,
                'analysis_buy_price': selected_symbol_dict['analysis_last_close_price'],
                'analysis_profit_percentage': selected_symbol_dict['analysis_profit_percentage'],
                'user_id_list': success_user_id_list,
                'analysis_symbol_token': selected_symbol_dict['analysis_token'],
                'stoploss_exit_price': selected_symbol_dict['stoploss']
            }
            upload_i_dict(i_dict)
            logging.info(f"Uploaded {analysis_symbol} for {interval} in i_dict and deleted from analysis_data")

        del context
    return True
