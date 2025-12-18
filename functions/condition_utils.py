import pandas as pd

"""Statistics calculation functions."""
from collections import Counter


def calculate_average(trades, setup):
    """Calculate average metrics for a setup."""
    setup_trades = [t for t in trades if t.get('setup') == setup]
    if not setup_trades:
        return 0, 0, [], None

    avg_duration = sum(t.get('duration', 0) for t in setup_trades) / len(setup_trades)
    avg_profit = sum(t.get('profit_percentage', 0) for t in setup_trades) / len(setup_trades)

    sell_types = [t.get('sell_point_type') for t in setup_trades if t.get('sell_point_type')]
    most_common = max(Counter(sell_types), key=Counter(sell_types).get) if sell_types else None

    return avg_duration, avg_profit, setup_trades, most_common


def extract_and_format_averages(trades, setup):
    """Extract and format averages for a setup."""
    duration, profit, setup_trades, sell_type = calculate_average(trades, setup)
    return {
        f'{setup}_average_profit_percentage': float(profit),
        f'{setup}_average_sell_point_type': sell_type,
        f'{setup}_average_duration_seconds': duration,
        f'{setup}_total_trades': len(setup_trades)
    }


def bearish_price_condition(price, price_before, highest_price_before_30):
    if price > price_before and price > highest_price_before_30:
        return 1
    else:
        return 0

def bearish_rsi_condition(current_rsi, rsi_before, highest_rsi_before_previous):

    try:
        if rsi_before < current_rsi < highest_rsi_before_previous:
            return 1
        else:
            return 0
    except Exception as e:
        f"{e}"
        return 0

def bullish_price_condition(price, price_before, lowest_price_before_30):
    if price < price_before and price < lowest_price_before_30:
        return 1
    else:
        return 0

def bullish_rsi_condition(current_rsi, rsi_before, lowest_rsi_before_previous):

    try:
        if rsi_before > current_rsi > lowest_rsi_before_previous:
            return 1
        else:
            return 0
    except Exception as e:
        f"{e}"
        return 0

def timeframe_to_periods_per_day(timeframe, trading_hours_per_day=6.5):
    if timeframe[-2:] == 'MO':
        unit = 'MO'
        value = int(timeframe[:-2])
    else:
        unit = timeframe[-1]
        value = int(timeframe[:-1])

    if unit == 'D':
        return 1
    elif unit == 'H':
        result = trading_hours_per_day // value
        return result
    elif unit == 'M':
        minutes_per_day = trading_hours_per_day * 60
        result = minutes_per_day // value
        return result
    elif unit == 'W':
        result = 5 // value
        return result
    elif unit == 'MO':
        result = 21 // value
        return result
    else:
        raise ValueError("Unsupported timeframe format. Use 'D' for days, 'H' for hours, 'M' for minutes, "
                         "'W' for weeks, or 'MO' for month.")

def get_bullish_prices_and_rsi_from_df(df, current_date):

    # Convert provided_date to a datetime object if it's not already
    current_date = pd.to_datetime(current_date)

    # Extract the row where the date matches the provided_date
    matching_row = df[df['datetime'] == current_date]

    # Get the index of the matching row
    current_index = matching_row.index[0]

    # Get the current price
    current_price = df.at[current_index, 'low']

    # Get the current RSI value
    current_rsi = df.at[current_index, 'rsi']

    # Get the price before the current price
    if current_index > 0:
        price_before = df.at[current_index - 1, 'low']
        rsi_before = df.at[current_index - 1, 'rsi']
        before_date = df.at[current_index - 1, 'datetime']
    else:
        price_before = current_price  # Assuming same price if no previous data
        rsi_before = current_rsi
        before_date = current_date

    # Get the lowest price among the 30 prices before the previous price
    if current_index > 1:

        # Find the index of the row with the minimum low price in the specified range
        min_price_index = df['low'].iloc[max(0, current_index - 31):current_index - 1].idxmin()

        # Get the lowest price before previous
        lowest_price_before_previous = df.at[min_price_index, 'low']

        # Get the date of that lowest price
        lowest_before_previous_date = df.at[min_price_index, 'datetime']

        lowest_rsi_before_previous = df.at[min_price_index, 'rsi']

    else:

        lowest_price_before_previous = current_price  # Assuming same price if less than 30 records

        # Get the date of that lowest price
        lowest_before_previous_date = current_date

        lowest_rsi_before_previous = current_rsi

    return current_index, current_price, price_before, lowest_price_before_previous, current_rsi, rsi_before, \
        lowest_rsi_before_previous, current_date, before_date, lowest_before_previous_date

def get_bearish_prices_and_rsi_from_df(df, current_date):

    # Convert provided_date to a datetime object if it's not already
    current_date = pd.to_datetime(current_date)

    # Extract the row where the date matches the provided_date
    matching_row = df[df['datetime'] == current_date]

    # Get the index of the matching row
    current_index = matching_row.index[0]

    # Get the current price
    current_price = df.at[current_index, 'high']

    # Get the current RSI value
    current_rsi = df.at[current_index, 'rsi']

    # Get the price before the current price
    if current_index > 0:
        price_before = df.at[current_index - 1, 'high']
        rsi_before = df.at[current_index - 1, 'rsi']
        before_date = df.at[current_index - 1, 'datetime']
    else:
        price_before = current_price  # Assuming same price if no previous data
        rsi_before = current_rsi
        before_date = current_date

    # Get the lowest price among the 30 prices before the previous price
    if current_index > 1:

        # Find the index of the row with the maximum high price in the specified range
        max_price_index = df['high'].iloc[max(0, current_index - 31):current_index - 1].idxmax()

        # Get the lowest price before previous
        lowest_price_before_previous = df.at[max_price_index, 'high']

        # Get the date of that lowest price
        lowest_before_previous_date = df.at[max_price_index, 'datetime']

        lowest_rsi_before_previous = df.at[max_price_index, 'rsi']

    else:

        lowest_price_before_previous = current_price  # Assuming same price if less than 30 records

        # Get the date of that lowest price
        lowest_before_previous_date = current_date

        lowest_rsi_before_previous = current_rsi

    return current_index, current_price, price_before, lowest_price_before_previous, current_rsi, rsi_before, \
        lowest_rsi_before_previous, current_date, before_date, lowest_before_previous_date

def get_last_ema_crossing(recent_crossings):

    for index, crossing in reversed(recent_crossings):
        if crossing == 1:
            return "ema_crossing_above", index
        elif crossing == -1:
            return "ema_crossing_below", index

    return None, None

def get_divergence(results):

    if results:
        # Iterate through the results list in reverse order
        for date, divergence in reversed(results):
            if divergence != 'None':
                return date, divergence  # Return the last found non-None divergence

    return None, None  # Return None if no divergence is found
