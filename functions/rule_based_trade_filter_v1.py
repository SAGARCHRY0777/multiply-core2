import logging
from conditions import (
    detect_divergence,
    check_ema_5_crossing_13,
    check_ema_5_crossing_50,
    candle_with_good_volume,
    check_most_recent_macd_crossing,
    directional_indicator_check,
    check_adx_greater_than_14,
    price_50ema,
    rsi_trend_check,
    check_rsi,
    check_limit_price_break,
    higher_interval_divergence_check
)
from functions.condition_utils import get_divergence
from functions.data_utils import add_percentage, subtract_percentage

def rule_based_trade_filter(stock_df, macd_df, analysis_symbol, analysis_token, interval,
                            bullish_average_sell_point, bearish_average_sell_point,
                            target_profit_percentage):

    print("rule_based_trade_filter")
    logging.info(f'Analyzing {analysis_symbol}')
    logging.info(f'stock df last row: {stock_df.iloc[-1]}')
    logging.info(f"Interval: {interval}")
    try:
        last_close_price = stock_df.iloc[-1]['close']
        last_volume = stock_df.iloc[-1]['volume']
        last_datetime = stock_df.iloc[-1]['datetime']
        result = {
            'symbol': analysis_symbol,
            'result': False,
            'setup_used': None,
            'analysis_data': None
        }
        interval_to_timeframe_map = {
            'ONE_MINUTE': '1M',
            'FIVE_MINUTE': '5M',
            'FIFTEEN_MINUTE': '15M',
            'ONE_HOUR': '1H',
            'ONE_DAY': '1D',
            'ONE_WEEK': '1W',
            'ONE_MONTH': '1MO'
        }
        timeframe = interval_to_timeframe_map[interval]
        hid_results, _ = higher_interval_divergence_check(stock_df, timeframe)
        last_hid_result_type = "None"
        if hid_results:
            last_hid_result_type = hid_results[-1][1]

        divergences, total_check_days = detect_divergence(stock_df, timeframe)
        divergence_date, last_divergence_result = get_divergence(divergences)
        logging.info(f"Divergence result: {last_divergence_result}, {divergence_date}")

        if last_divergence_result == "bullish divergence":
            ema_5_13_crossing_result, ema_5_13_crossing_index = check_ema_5_crossing_13(
                divergence_date, stock_df, timeframe, total_check_days)
            logging.info(f"EMA 5 crossing 13 result: {ema_5_13_crossing_result}")
            if ema_5_13_crossing_result == "ema_crossing_above_13":
                ema_5_50_crossing_result, ema_5_50_crossing_index = check_ema_5_crossing_50(
                    divergence_date, stock_df, timeframe, total_check_days)
                logging.info(f"EMA 5 crossing 50 result: {ema_5_50_crossing_result}")
                if ema_5_50_crossing_result == "ema_crossing_above_50":
                    ema_5_50_candle_check, _ = candle_with_good_volume(ema_5_50_crossing_index, stock_df)
                    logging.info(f"Candle with good volume result: {ema_5_50_candle_check}")
                    if ema_5_50_candle_check == "bullish_candle_with_good_volume":
                        check_price_above_50_ema = price_50ema(stock_df, last_close_price)
                        logging.info(f"Price above 50 EMA result: {check_price_above_50_ema}")
                        if check_price_above_50_ema == "price_above_50ema":
                            filtered_macd_df = macd_df[
                                (macd_df['datetime'] >= divergence_date) & (macd_df['datetime'] <= last_datetime)
                            ]

                            check_macd_over_signal_line = check_most_recent_macd_crossing(filtered_macd_df)
                            logging.info(f"MACD over signal line result: {check_macd_over_signal_line}")
                            if check_macd_over_signal_line == "macd_over_signal":
                                plus_di_minus_di = directional_indicator_check(stock_df)
                                logging.info(f"Directional indicator result: {plus_di_minus_di}")
                                if plus_di_minus_di == "plus_di_above_minus_di":
                                    adx_above_14 = check_adx_greater_than_14(stock_df)
                                    logging.info(f"ADX greater than 14 result: {adx_above_14}")
                                    if adx_above_14 == "adx_greater_than_14":
                                        rsi_60 = check_rsi(stock_df)
                                        logging.info(f"RSI above 60 result: {rsi_60}")
                                        if rsi_60 == "rsi_above_60":
                                            break_limit_price = check_limit_price_break(ema_5_50_crossing_index,
                                                                                        stock_df)
                                            logging.info(f"Limit price break result: {break_limit_price}")
                                            if break_limit_price == "price_above_limit_high":
                                                current_rsi_trend = rsi_trend_check(stock_df)
                                                logging.info(f"RSI trend result: {current_rsi_trend}")
                                                if current_rsi_trend == "rsi_uptrend":
                                                    result['result'] = True
                                                    result['setup_used'] = 'bullish_momentum_setup'
                                                    result['analysis_data'] = {
                                                        'date': last_datetime,
                                                        'buy_indicator': 'bullish_momentum_setup',
                                                        'analysis_symbol': analysis_symbol,
                                                        'analysis_last_close_price': float(last_close_price),
                                                        'analysis_token': analysis_token,
                                                        'higher_interval_divergence': last_hid_result_type == 'bullish divergence',
                                                        'volume': last_volume,
                                                        'analysis_profit_percentage': target_profit_percentage,
                                                        'target': add_percentage(price=float(last_close_price), percent=float(target_profit_percentage)),
                                                        'divergence_date': divergence_date
                                                    }
                                                    return result
                                                else:
                                                    return result
                                            else:
                                                return result
                                        else:
                                            return result
                                    else:
                                        return result
                                else:
                                    return result
                            else:
                                return result
                        else:
                            return result
                    else:
                        check_price_above_50_ema = price_50ema(stock_df, last_close_price)
                        logging.info(f"Price above 50 EMA result: {check_price_above_50_ema}")
                        if check_price_above_50_ema == "price_above_50ema":
                            filtered_macd_df = macd_df[
                                (macd_df['datetime'] >= divergence_date) & (macd_df['datetime'] <= last_datetime)
                            ]

                            check_macd_over_signal_line = check_most_recent_macd_crossing(filtered_macd_df)
                            logging.info(f"MACD over signal line result: {check_macd_over_signal_line}")
                            if check_macd_over_signal_line == "macd_over_signal":
                                plus_di_minus_di = directional_indicator_check(stock_df)
                                logging.info(f"Directional indicator result: {plus_di_minus_di}")
                                if plus_di_minus_di == "plus_di_above_minus_di":
                                    adx_above_14 = check_adx_greater_than_14(stock_df)
                                    logging.info(f"ADX greater than 14 result: {adx_above_14}")
                                    if adx_above_14 == "adx_greater_than_14":
                                        rsi_60 = check_rsi(stock_df)
                                        logging.info(f"RSI above 60 result: {rsi_60}")
                                        if rsi_60 == "rsi_above_60":
                                            break_limit_price = check_limit_price_break(ema_5_50_crossing_index,
                                                                                        stock_df)
                                            logging.info(f"Limit price break result: {break_limit_price}")
                                            if break_limit_price == "price_above_limit_high":
                                                current_rsi_trend = rsi_trend_check(stock_df)
                                                logging.info(f"RSI trend result: {current_rsi_trend}")
                                                if current_rsi_trend == "rsi_uptrend":
                                                    result['result'] = True
                                                    result['setup_used'] = 'bullish_momentum_setup'
                                                    result['analysis_data'] = {
                                                        'date': last_datetime,
                                                        'buy_indicator': 'bullish_momentum_setup',
                                                        'analysis_symbol': analysis_symbol,
                                                        'analysis_last_close_price': float(last_close_price),
                                                        'analysis_token': analysis_token,
                                                        'higher_interval_divergence': last_hid_result_type == 'bullish divergence',
                                                        'volume': last_volume,
                                                        'analysis_profit_percentage': target_profit_percentage,
                                                        'target': add_percentage(price=float(last_close_price),
                                                                                 percent=float(target_profit_percentage)),
                                                        'divergence_date': divergence_date
                                                    }
                                                    return result
                                                else:
                                                    return result
                                            else:
                                                return result
                                        else:
                                            return result
                                    else:
                                        return result
                                else:
                                    return result
                            else:
                                return result
                        else:
                            return result
                else:
                    return result
            else:
                return result

        elif last_divergence_result == "bearish divergence":
            ema_5_13_crossing_result, ema_5_13_crossing_index = check_ema_5_crossing_13(
                divergence_date, stock_df, timeframe, total_check_days)
            logging.info(f"EMA 5 crossing 13 result: {ema_5_13_crossing_result}")
            if ema_5_13_crossing_result == "ema_crossing_below_13":
                ema_5_50_crossing_result, ema_5_50_crossing_index = check_ema_5_crossing_50(
                    divergence_date, stock_df, timeframe, total_check_days)
                logging.info(f"EMA 5 crossing 50 result: {ema_5_50_crossing_result}")
                if ema_5_50_crossing_result == "ema_crossing_below_50":
                    ema_5_50_candle_check, _ = candle_with_good_volume(ema_5_50_crossing_index, stock_df)
                    logging.info(f"Candle with good volume result: {ema_5_50_candle_check}")
                    if ema_5_50_candle_check == "bearish_candle_with_good_volume":
                        check_price_below_50_ema = price_50ema(stock_df, last_close_price)
                        logging.info(f"Price below 50 EMA result: {check_price_below_50_ema}")
                        if check_price_below_50_ema == "price_below_50ema":
                            filtered_macd_df = macd_df[
                                (macd_df['datetime'] >= divergence_date) & (macd_df['datetime'] <= last_datetime)
                            ]

                            check_signal_line_over_macd = check_most_recent_macd_crossing(filtered_macd_df)
                            logging.info(f"Signal line over MACD result: {check_signal_line_over_macd}")
                            if check_signal_line_over_macd == "signal_over_macd":
                                minus_di_plus_di = directional_indicator_check(stock_df)
                                logging.info(f"Directional indicator result: {minus_di_plus_di}")
                                if minus_di_plus_di == "minus_di_above_plus_di":
                                    adx_above_14 = check_adx_greater_than_14(stock_df)
                                    logging.info(f"ADX greater than 14 result: {adx_above_14}")
                                    if adx_above_14 == "adx_greater_than_14":
                                        rsi_40 = check_rsi(stock_df)
                                        logging.info(f"RSI below 40 result: {rsi_40}")
                                        if rsi_40 == "rsi_below_40":
                                            break_limit_price = check_limit_price_break(ema_5_50_crossing_index,
                                                                                        stock_df)
                                            logging.info(f"Limit price break result: {break_limit_price}")
                                            if break_limit_price == "price_below_limit_low":
                                                current_rsi_trend = rsi_trend_check(stock_df)
                                                logging.info(f"RSI trend result: {current_rsi_trend}")
                                                if current_rsi_trend == "rsi_downtrend":
                                                    result['result'] = True
                                                    result['setup_used'] = 'bearish_momentum_setup'
                                                    result['analysis_data'] = {
                                                        'date': last_datetime,
                                                        'buy_indicator': 'bearish_momentum_setup',
                                                        'analysis_symbol': analysis_symbol,
                                                        'analysis_last_close_price': float(last_close_price),
                                                        'analysis_token': analysis_token,
                                                        'higher_interval_divergence': last_hid_result_type == 'bearish divergence',
                                                        'volume': last_volume,
                                                        'analysis_profit_percentage': target_profit_percentage,
                                                        'target': subtract_percentage(price=float(last_close_price),
                                                                                      percent=float(target_profit_percentage)),
                                                        'divergence_date': divergence_date
                                                    }
                                                    return result
                                                else:
                                                    return result
                                            else:
                                                return result
                                        else:
                                            return result
                                    else:
                                        return result
                                else:
                                    return result
                            else:
                                return result
                        else:
                            return result
                    else:
                        check_price_below_50_ema = price_50ema(stock_df, last_close_price)
                        logging.info(f"Price below 50 EMA result: {check_price_below_50_ema}")
                        if check_price_below_50_ema == "price_below_50ema":
                            filtered_macd_df = macd_df[
                                (macd_df['datetime'] >= divergence_date) & (macd_df['datetime'] <= last_datetime)
                            ]

                            check_signal_line_over_macd = check_most_recent_macd_crossing(filtered_macd_df)
                            logging.info(f"Signal line over MACD result: {check_signal_line_over_macd}")
                            if check_signal_line_over_macd == "signal_over_macd":
                                minus_di_plus_di = directional_indicator_check(stock_df)
                                logging.info(f"Directional indicator result: {minus_di_plus_di}")
                                if minus_di_plus_di == "minus_di_above_plus_di":
                                    adx_above_14 = check_adx_greater_than_14(stock_df)
                                    logging.info(f"ADX greater than 14 result: {adx_above_14}")
                                    if adx_above_14 == "adx_greater_than_14":
                                        rsi_40 = check_rsi(stock_df)
                                        logging.info(f"RSI below 40 result: {rsi_40}")
                                        if rsi_40 == "rsi_below_40":
                                            break_limit_price = check_limit_price_break(ema_5_50_crossing_index,
                                                                                        stock_df)
                                            logging.info(f"Limit price break result: {break_limit_price}")
                                            if break_limit_price == "price_below_limit_low":
                                                current_rsi_trend = rsi_trend_check(stock_df)
                                                logging.info(f"RSI trend result: {current_rsi_trend}")
                                                if current_rsi_trend == "rsi_downtrend":
                                                    result['result'] = True
                                                    result['setup_used'] = 'bearish_momentum_setup'
                                                    result['analysis_data'] = {
                                                        'date': last_datetime,
                                                        'buy_indicator': 'bearish_momentum_setup',
                                                        'analysis_symbol': analysis_symbol,
                                                        'analysis_last_close_price': float(last_close_price),
                                                        'analysis_token': analysis_token,
                                                        'higher_interval_divergence': last_hid_result_type == 'bearish divergence',
                                                        'volume': last_volume,
                                                        'analysis_profit_percentage': target_profit_percentage,
                                                        'target': subtract_percentage(price=float(last_close_price),
                                                                                      percent=float(target_profit_percentage)),
                                                        'divergence_date': divergence_date

                                                    }
                                                    return result
                                                else:
                                                    return result
                                            else:
                                                return result
                                        else:
                                            return result
                                    else:
                                        return result
                                else:
                                    return result
                            else:
                                return result
                        else:
                            return result
                else:
                    return result
            else:
                return result
        else:
            return result

    except Exception as e:
        logging.error(f"Error analyzing {analysis_symbol} on {interval}: {str(e)}", exc_info=True)
        return {
            'symbol': analysis_symbol,
            'result': False,
            'setup_used': None,
            'analysis_data': None
        }
