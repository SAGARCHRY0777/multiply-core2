from datetime import datetime, timedelta


def volume_volatility_check(option, thresholds=None, symbol=None):
    """
    Perform volume and volatility checks on an option.

    Parameters:
        option (dict): The option data dictionary.
        thresholds (dict): Thresholds for filtering (default values will be used if None).
        symbol (str): The analysis symbol (e.g., 'NIFTY', 'GOLD') to determine default thresholds.

    Returns:
        bool: True if the option passes all checks, False otherwise.
    """
    if thresholds is None:
        # Determine default thresholds based on symbol type
        min_vol_oi = 10000  # Default for Stocks
        if symbol:
            symbol_upper = symbol.upper()
            if symbol_upper in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]:
                min_vol_oi = 5000
            elif symbol_upper in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
                min_vol_oi = 500

        thresholds = {
            'max_inactive_time': 2700,  # 45 minutes
            'max_spread': 0.03  # 3%
        }

    try:
        # Check for required fields
        if not all(key in option for key in ['opnInterest', 'exchTradeTime', 'totBuyQuan', 'totSellQuan']):
            print("Missing required fields in option data")
            return False

        # Extract and validate market depth
        depth = option.get('depth', {})
        depth_buy = depth.get('buy', [])
        depth_sell = depth.get('sell', [])
        if not depth_buy or not depth_sell:
            print("Missing buy/sell levels in market depth")
            return False
        if len(depth_buy) < 5 or len(depth_sell) < 5:
            print("Insufficient market depth levels")
            return False

        # Extract prices
        bid_price = depth_buy[0]['price']
        ask_price = depth_sell[0]['price']
        mid_price = (bid_price + ask_price) / 2
        if mid_price <= 0:
            print("Invalid mid_price (bid/ask prices are zero)")
            return False

        # Parse trade time (timezone-naive)
        last_trade_time_str = option['exchTradeTime']
        try:
            last_trade_time = datetime.strptime(last_trade_time_str, '%d-%b-%Y %H:%M:%S')
        except ValueError:
            print(f"Invalid trade time format: {last_trade_time_str}")
            return False

        # Calculate current time in IST (timezone-naive)
        current_time_utc = datetime.utcnow()
        ist_offset = timedelta(hours=5, minutes=30)
        current_time = current_time_utc + ist_offset  # Now in IST

        # Validate time difference
        inactive_seconds = (current_time - last_trade_time).total_seconds()
        if inactive_seconds < 0:
            print("Last trade time is in the future (invalid data)")
            return False

        checks = [
            (inactive_seconds <= thresholds['max_inactive_time'], "Last Trade Time"),
            (int(option['totBuyQuan']) > int(option['totSellQuan']), "Buy > Sell Quantity"),
            (all(int(level['quantity']) > 0 for level in depth_buy[:5]), "Buy Market Depth"),
            (all(int(level['quantity']) > 0 for level in depth_sell[:5]), "Sell Market Depth"),
            ((ask_price - bid_price) / mid_price <= thresholds['max_spread'], "Bid-Ask Spread")
        ]

        for check_passed, check_name in checks:
            if not check_passed:
                print(f"Failed {check_name} Check")
                return False

        return True

    except Exception as e:
        print(f"Error in volume_volatility_check: {e}")
        return False
