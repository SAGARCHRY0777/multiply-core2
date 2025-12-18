import pandas as pd
from functions.condition_utils import (timeframe_to_periods_per_day, bullish_rsi_condition,
                                       bullish_price_condition, bearish_rsi_condition,
                                       bearish_price_condition, get_bearish_prices_and_rsi_from_df,
                                       get_bullish_prices_and_rsi_from_df)


def higher_interval_divergence_check(df, timeframe):
    if timeframe == "1M":
        timeframe = "15M"
    elif timeframe == "5M":
        timeframe = "15M"
    elif timeframe == "15M":
        timeframe = "1D"
    elif timeframe == "1H":
        timeframe = "1D"
    elif timeframe == "1D":
        timeframe = "1W"
    elif timeframe == "1W":
        timeframe = "1MO"
    else:
        timeframe = "1MO"
    df['datetime'] = pd.to_datetime(df['datetime'])

    periods_per_day = timeframe_to_periods_per_day(timeframe)

    if timeframe.endswith('M'):
        total_days = 4
    elif timeframe.endswith('H'):
        total_days = 8
    elif timeframe.endswith('D'):
        total_days = 14
    elif timeframe.endswith('W'):
        total_days = 24
    elif timeframe.endswith('MO'):
        total_days = 120
    else:
        return None, None

    total_candles_to_check_for = int(periods_per_day * total_days)

    if len(df) < total_candles_to_check_for:
        return None, None

    current_date = pd.to_datetime(df['datetime'].iloc[-1])
    # Ensure that the DataFrame has enough data

    # Generate the range of dates/times for analysis
    listed_dates = df['datetime'].loc[df['datetime'] <= current_date].iloc[-total_candles_to_check_for:].tolist()
    results = []

    for date_i in listed_dates:

        current_index, current_price, price_before, lowest_price_before_previous, current_rsi, rsi_before, \
            lowest_rsi_before_previous, current_date, before_date, lowest_before_previous_date = \
            get_bullish_prices_and_rsi_from_df(df, date_i)

        condition_check = bullish_price_condition(current_price, price_before, lowest_price_before_previous)

        rsi_re = bullish_rsi_condition(current_rsi, rsi_before, lowest_rsi_before_previous)

        divergence_result = "bullish divergence" if rsi_re == 1 and condition_check == 1 else "None"

        if divergence_result == 'None':
            current_index, current_price, price_before, lowest_price_before_previous, current_rsi, rsi_before, \
                lowest_rsi_before_previous, current_date, before_date, lowest_before_previous_date = \
                get_bearish_prices_and_rsi_from_df(df, date_i)

            bearish_price_re = bearish_price_condition(current_price, price_before, lowest_price_before_previous)

            bearish_rsi_re = bearish_rsi_condition(current_rsi, rsi_before, lowest_rsi_before_previous)

            divergence_result = "bearish divergence" if bearish_rsi_re == 1 and bearish_price_re == 1 else "None"

        results.append((current_date, divergence_result))

    return results, total_days
