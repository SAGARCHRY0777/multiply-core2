from functions.SO_angel_v2.SmartApi.smartConnect import (SmartConnect)
import pyotp
import logging
import time
import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')

def ohlc_history(exchange, symbol, symbol_token, interval, f_date, todate, login_details):

    try:
        user_name = login_details["angel_one_user_name"]
        pin = login_details["angel_one_pin"]
        otp_token = login_details["angel_one_otp_token"]
        api_key = login_details["angel_one_api_key"]

        obj = SmartConnect(api_key=api_key)
        data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
        if not data or 'data' not in data or 'refreshToken' not in data['data']:
            logger.error(
                "AngelOne generateSession returned invalid response for user_id=%s",
                login_details.get('user_id')
            )
            raise Exception("Failed to generate session: Invalid response")
        refreshtoken = data['data']['refreshToken']
        profile = obj.getProfile(refreshtoken)

        if not profile:
            raise Exception("Failed to get profile")
        # Convert datetime objects to ISO format strings
        f_date_str = f_date.strftime("%Y-%m-%d %H:%M")
        todate_str = todate.strftime("%Y-%m-%d %H:%M")
        logger.debug(
            "Requesting OHLC history | user_id=%s exchange=%s symbol=%s token=%s interval=%s from=%s to=%s",
            login_details.get('user_id'), exchange, symbol, symbol_token, interval, f_date_str, todate_str
        )
        historic_param = {
            "exchange": exchange,
            "tradingsymbol": symbol.encode('ascii', 'ignore').decode('ascii'),  # Remove non-ASCII characters
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": f_date_str,
            "todate": todate_str
        }
        # Get candle data with validation
        history = obj.getCandleData(historic_param)
        if not history:
            raise Exception("Empty response from getCandleData")

        if not isinstance(history, dict):
            raise Exception(f"Invalid response type from getCandleData: {type(history)}")

        if 'data' not in history:
            raise Exception("No 'data' field in response")

        # Clean any potential encoding issues in the response data
        if 'data' in history and isinstance(history['data'], list):
            cleaned_data = []
            for item in history['data']:
                if isinstance(item, (list, tuple)):
                    # Clean string values in the data
                    cleaned_item = []
                    for val in item:
                        if isinstance(val, str):
                            # Remove or replace problematic characters
                            val = val.encode('ascii', 'ignore').decode('ascii')
                        cleaned_item.append(val)
                    cleaned_data.append(cleaned_item)
                else:
                    cleaned_data.append(item)
            history['data'] = cleaned_data

        logger.info(
            "OHLC history fetch succeeded | user_id=%s exchange=%s symbol=%s interval=%s candles=%s",
            login_details.get('user_id'), exchange, symbol, interval, len(history.get('data') or [])
        )

        return history

    except Exception as e:
        logger.exception(
            "Historic Api failed | user_id=%s exchange=%s symbol=%s interval=%s error=%s",
            login_details.get('user_id'), exchange, symbol, interval, e
        )
        return None


def place_order(order_params, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        order_id = obj.placeOrder(order_params)

    except Exception as e:
        order_id = None
        print("Order placement failed: {}", format(e))

    return order_id


def modify_order(order_params, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        obj.modifyOrder(order_params)
        return True
    except Exception as e:
        print("Order modification failed: {}", format(e))
        return False


def cancel_order(order_id, variety, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        obj.cancelOrder(order_id, variety)
        return True
    except Exception as e:
        print("Order cancellation failed: {}", format(e))
        return False



def symbol_ltp_data(exchange, tradingsymbol, symboltoken, initial_login_details, login_details_list, max_retries=5,
                    delay=2):
    """
    Retrieve LTP (Last Traded Price) data with a retry mechanism and rotating login details on failures.

    Args:
        exchange (str): The exchange name.
        tradingsymbol (str): The trading symbol.
        symboltoken (str): The symbol token.
        initial_login_details (dict): The initial login details for the trading platform.
        login_details_list (list): A list of alternate login details dictionaries.
        max_retries (int): Maximum number of retry attempts (default: 5).
        delay (int or float): Base delay in seconds between retries (default: 2).

    Returns:
        dict: The LTP data if successful, None otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Rotate login details based on the attempt number
            if attempt == 1:
                login_details = initial_login_details
            else:
                login_details = login_details_list[attempt % len(login_details_list)]

            user_name = login_details["angel_one_user_name"]
            pin = login_details["angel_one_pin"]
            otp_token = login_details["angel_one_otp_token"]
            api_key = login_details["angel_one_api_key"]

            # Create a new SmartConnect object with the selected api_key
            obj = SmartConnect(api_key=api_key)

            # Generate session for each attempt
            data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
            refreshtoken = data['data']['refreshToken']
            obj.getProfile(refreshtoken)

            print(f"Attempt {attempt}: Retrieving LTP data for {tradingsymbol} on {exchange}")
            ltp_data = obj.ltpData(exchange, tradingsymbol, symboltoken)
            print("Successfully retrieved LTP data.")
            return ltp_data

        except Exception as e:
            print(f"Error retrieving LTP data on attempt {attempt}: {e}")

        # Delay before the next retry
        if attempt < max_retries:
            retry_delay = attempt * delay
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    print("Failed to retrieve LTP data after {} attempts.".format(max_retries))
    return None



def quote_data(mode, exchangetokens, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        quote = obj.getMarketData(mode, exchangetokens)

        return quote

    except Exception as e:
        print("Quote Data failed: {}", format(e))


def order_details(q_param, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        details = obj.individual_order_details(q_param)

    except Exception as e:
        details = None
        print("Individual Order Details Failed: {}", format(e))

    return details


def margin_info(params, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        margin = obj.getMarginApi(params)

    except Exception as e:
        margin = None
        print("Get Margin Details Failed: {}", format(e))

    return margin


def estimated_charges(params, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        brokerage = obj.estimateCharges(params)

    except Exception as e:
        brokerage = None
        print("Get Brokerage Details Failed: {}", format(e))

    return brokerage


def get_positions_data(login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        position_data = obj.position()
    except Exception as e:
        position_data = None
        print("Get Positions Details Failed: {}", format(e))

    return position_data


def get_order_book(login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:

        order_book = obj.orderBook()

    except Exception as e:
        order_book = None
        print("Get Order Book Failed: {}", format(e))

    return order_book


def instrument_data(exchange, symbol, login_details):
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
    refreshtoken = data['data']['refreshToken']

    obj.getProfile(refreshtoken)

    try:
        symbol = symbol + "-EQ"
        instrument_info = obj.searchScrip(exchange, symbol)

    except Exception as e:
        instrument_info = None
        "Get Instrument Data Failed: {}", format(e)

    return instrument_info


def get_funds_and_margins_info(login_details, max_retries=3, delay=2):
    """
    Retrieve funds and margins info with retry mechanism.
    
    Args:
        login_details: User login credentials
        max_retries: Maximum retry attempts (default: 3)
        delay: Base delay between retries in seconds (default: 2)
    
    Returns:
        dict: Funds and margin info if successful, None otherwise
    """
    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        try:
            # Generate session for each attempt
            data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
            refreshtoken = data['data']['refreshToken']
            obj.getProfile(refreshtoken)

            logger.debug(f"Attempt {attempt}: Retrieving funds and margins info")
            funds_and_margin_info = obj.rmsLimit()
            
            if funds_and_margin_info and funds_and_margin_info.get('data'):
                logger.debug("Successfully retrieved funds and margins info.")
                return funds_and_margin_info

        except Exception as e:
            logger.warning(f"Error retrieving funds info on attempt {attempt}: {e}")

        # Delay before the next retry
        if attempt < max_retries:
            retry_delay = attempt * delay
            logger.debug(f"Retrying funds retrieval in {retry_delay} seconds...")
            time.sleep(retry_delay)

    logger.error(f"Failed to retrieve funds info after {max_retries} attempts.")
    return None

def get_trade_book(login_details, max_retries=5, delay=2):

    user_name = login_details["angel_one_user_name"]
    pin = login_details["angel_one_pin"]
    otp_token = login_details["angel_one_otp_token"]
    api_key = login_details["angel_one_api_key"]

    obj = SmartConnect(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        try:
            # Generate session for each attempt
            data = obj.generateSession(user_name, pin, pyotp.TOTP(otp_token).now())
            refreshtoken = data['data']['refreshToken']
            obj.getProfile(refreshtoken)

            print(f"Attempt {attempt}: Retrieving trade book")
            trade_book = obj.tradeBook()
            print("Successfully retrieved trade book.")
            return trade_book

        except Exception as e:
            print(f"Error retrieving trade book on attempt {attempt}: {e}")

        # Delay before the next retry
        if attempt < max_retries:
            retry_delay = attempt * delay
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    print("Failed to retrieve trade book after {} attempts.".format(max_retries))
    return None
