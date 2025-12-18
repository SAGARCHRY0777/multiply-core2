from functions.SO_angel_v2.angel_one_functions import place_order


def buy_equity(login_details, symbol, symbol_token, quantity, exchange, disclosed_quantity, order_type='MARKET',
               product_type='DELIVERY',
               validity='DAY'):

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": symbol_token,
        "transactiontype": "BUY",
        "quantity": quantity,
        "ordertype": order_type,
        "producttype": product_type,
        "exchange": exchange,
        "duration": validity,
        "disclosedquantity": disclosed_quantity
    }

    # Remove None values from order_params
    order_params = {k: v for k, v in order_params.items() if v is not None}

    order_id = place_order(order_params, login_details)

    return order_id, order_params

def buy_futures(login_details, symbol, symbol_token, quantity, exchange, order_type='MARKET',
                product_type='CARRYFORWARD', validity='DAY'):

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": symbol_token,
        "transactiontype": "BUY",
        "quantity": quantity,
        "ordertype": order_type,
        "producttype": product_type,
        "exchange": exchange,
        "duration": validity
    }

    # Remove None values from order_params
    order_params = {k: v for k, v in order_params.items() if v is not None}

    order_id = place_order(order_params, login_details)

    return order_id, order_params

def buy_options_atm(login_details, symbol, symbol_token, quantity, exchange,
                    order_type='MARKET',
                    product_type='CARRYFORWARD', validity='DAY'):

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": symbol_token,
        "transactiontype": "BUY",
        "quantity": quantity,
        "ordertype": order_type,
        "producttype": product_type,
        "exchange": exchange,
        "duration": validity
    }
    # Remove None values from order_params
    order_params = {k: v for k, v in order_params.items() if v is not None}

    order_id = place_order(order_params, login_details)

    return order_id, order_params

def sell_futures(login_details, symbol, symbol_token, quantity, exchange, order_type='MARKET',
                 product_type='CARRYFORWARD', validity='DAY'):

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": symbol_token,
        "transactiontype": "SELL",
        "quantity": quantity,
        "ordertype": order_type,
        "producttype": product_type,
        "exchange": exchange,
        "duration": validity
    }

    # Remove None values from order_params
    order_params = {k: v for k, v in order_params.items() if v is not None}

    order_id = place_order(order_params, login_details)

    return order_id, order_params
