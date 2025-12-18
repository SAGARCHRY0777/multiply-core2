from itertools import cycle
from datetime import datetime, timedelta, timezone
import time
from functions.volume_volatility_check import volume_volatility_check
from functions.SO_angel_v2.angel_one_functions import quote_data


def get_next_login(user_db):
    """Returns the next user credentials in a round-robin manner using a persistent cycle."""
    if not hasattr(get_next_login, 'user_cycle'):
        get_next_login.user_cycle = cycle(user_db)
    return next(get_next_login.user_cycle)


def safe_quote_data(mode, exchangeTokens, user_db, max_retries=5):
    """
    Wrapper function for quote_data with rate limiting, retry logic, user switching, and error handling.
    """
    for attempt in range(1, max_retries + 1):
        try:
            login_details = get_next_login(user_db)  # Rotate users on every attempt
            user_id = str(login_details.get('user_id', 'anonymous'))
            masked_user_id = user_id[:2] + '***' + user_id[-2:] if len(user_id) > 4 else '***'
            print(f"Using login credentials (Attempt {attempt}) - user {masked_user_id}")

            response = quote_data(mode, exchangeTokens, login_details)

            # If response is valid, return it
            if response and isinstance(response, dict) and response.get("status"):
                return response

        except Exception as e:
            print(f"Error in quote_data (Attempt {attempt}): {e}")

        # If failed, wait before retrying
        if attempt < max_retries:
            backoff_time = 5 * attempt  # Exponential backoff-ish
            print(f"Rate limit hit! Retrying in {backoff_time} seconds with a new user...")
            time.sleep(backoff_time)

    print("Max retries reached. Skipping this request.")
    return None  # Return None after max retries


def get_itm_symbol_buy_info(analysis_symbol, buy_indicator, all_symbol_dict, segment, pi, user_db,
                            thresholds=None):
    """
    Returns the best in-the-money (ITM) symbol for a buy setup based on analysis_symbol,
    buy_indicator, segment, and a price input 'pi' (which may be string or numeric).
    """

    # ---- SAFELY COERCE price_input to float ----
    try:
        # strip commas/whitespace and convert to float
        price_input = float(str(pi).replace(',', '').strip())
    except Exception as e:
        print(f"[ERROR] price input '{pi}' is not numeric: {e}")
        return None

    # Debug print for input parameters
    print(f"[DEBUG] Input Parameters:")
    print(f"  - Analysis Symbol: {analysis_symbol}")
    print(f"  - Buy Indicator: {buy_indicator}")
    print(f"  - Segment: {segment}")
    print(f"  - Price Input: {price_input}")

    # Determine the instrument type and suffix based on segment and buy indicator
    if segment == "Futures":
        if analysis_symbol in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]:
            required_instrument_type = "FUTIDX"
            required_suffix = ""
        elif analysis_symbol in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
            required_instrument_type = "FUTCOM"
            required_suffix = ""
        else:
            required_instrument_type = "FUTSTK"
            required_suffix = ""
    elif segment == "Options" and buy_indicator == "bearish_momentum_setup":
        if analysis_symbol in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]:
            required_instrument_type = "OPTIDX"
            required_suffix = "PE"
        elif analysis_symbol in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
            required_instrument_type = "OPTFUT"
            required_suffix = "PE"
        else:
            required_instrument_type = "OPTSTK"
            required_suffix = "PE"
    elif segment == "Options" and buy_indicator == "bullish_momentum_setup":
        if analysis_symbol in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]:
            required_instrument_type = "OPTIDX"
            required_suffix = "CE"
        elif analysis_symbol in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
            required_instrument_type = "OPTFUT"
            required_suffix = "CE"
        else:
            required_instrument_type = "OPTSTK"
            required_suffix = "CE"
    elif segment == "Equity":
        required_instrument_type = ""
        required_suffix = "EQ"
    else:
        print(f"[ERROR] Invalid segment or buy indicator combination")
        return None

    # Debug print for instrument type and suffix
    print(f"[DEBUG] Instrument Configuration:")
    print(f"  - Required Instrument Type: {required_instrument_type}")
    print(f"  - Required Suffix: {required_suffix}")

    # Make datetime timezone-aware (UTC) and compute IST "today"
    current_time_utc = datetime.now(timezone.utc)
    ist_offset = timedelta(hours=5, minutes=30)
    today = current_time_utc + ist_offset  # IST-aware datetime
    # Flag for index-type symbols where fence is narrower
    is_nifty = analysis_symbol.upper().startswith('NIFTY') or analysis_symbol in ["BANKNIFTY", "SENSEX", "FINNIFTY"]
    min_expiry_days = 7 if is_nifty else 12
    if analysis_symbol in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
        min_expiry_days = 7

    # Filter the symbols based on name, instrument type, and suffix
    filtered_symbols = [
        item for item in all_symbol_dict
        if (item.get('name') == analysis_symbol and
            item.get('instrumenttype') == required_instrument_type and
            item.get('symbol', '').endswith(required_suffix))
    ]

    print(f"[DEBUG] Filtered Symbols Count: {len(filtered_symbols)}")

    # Helper to safely convert a symbol's strike to a float price (handles bad data and different scales)
    def strike_price(item):
        """
        Robust strike extraction:
        - tries direct float conversion
        - if value is extremely large (e.g., stored in paise or scaled), normalize sensibly
        """
        try:
            raw = item.get('strike', 0)
            val = float(raw)
            # If it's clearly scaled (very large), try dividing by 100
            if val > 1000000:  # extremely large -> likely paise * 100 or similar
                return val / 100.0
            # If it's in an intermediate scaled form (e.g., 2625000 -> 26250), divide by 100
            if val > 100000:
                return val / 100.0
            # If the strikes are like 26250 or 26200 this will return as-is (good)
            return val
        except Exception:
            return 0.0

    # Helper to check near-multiples with tolerance
    def is_multiple_of(value, base):
        try:
            rem = value % base
            return (rem < 1e-6) or (abs(rem - base) < 1e-6)
        except Exception:
            return False

    # FUTURES: Apply expiry filtering only (no strike filtering)
    if segment == "Futures":
        valid_symbols = []
        for item in filtered_symbols:
            expiry_date_str = item.get('expiry', '')
            try:
                expiry_date = datetime.strptime(expiry_date_str, '%d%b%Y')
                expiry_date = expiry_date.replace(tzinfo=timezone.utc)
                if expiry_date >= (today + timedelta(days=min_expiry_days)):
                    valid_symbols.append((item, expiry_date))
                    print(f"[DEBUG] Valid Future Symbol: {item.get('symbol')} , Expiry: {expiry_date}")
            except Exception:
                print(f"[WARNING] Invalid expiry date for symbol {item.get('symbol')}: {expiry_date_str}")
                continue

        if not valid_symbols:
            print("[DEBUG] No valid symbols found with required expiry window for Futures")
            return None

        valid_symbols.sort(key=lambda x: x[1])  # Sort by expiry date
        result = valid_symbols[0][0]  # Return the first symbol with nearest expiry
        print(f"[DEBUG] Selected Futures Symbol: {result}")
        return result

    # OPTIONS: Apply expiry and strike-based filtering
    elif segment == "Options":
        valid_symbols = []
        for item in filtered_symbols:
            expiry_date_str = item.get('expiry', '')
            try:
                expiry_date = datetime.strptime(expiry_date_str, '%d%b%Y').replace(tzinfo=timezone.utc)
                if expiry_date >= (today + timedelta(days=min_expiry_days)):
                    valid_symbols.append((item, expiry_date))
            except Exception:
                # skip invalid expiry strings quietly
                continue

        if not valid_symbols:
            print("[DEBUG] No valid symbols found with required expiry window for Options")
            return None

        # get nearest expiry date group
        valid_symbols.sort(key=lambda x: x[1])
        nearest_expiry_date = valid_symbols[0][1]
        nearest_expiry_symbols = [item for item, expiry in valid_symbols if expiry == nearest_expiry_date]

        # Find the closest strike to price_input safely
        try:
            closest_strike_symbol = min(
                nearest_expiry_symbols,
                key=lambda x: abs(price_input - strike_price(x))
            )
        except ValueError:
            # empty list guard
            print("[DEBUG] No nearest expiry symbols to compare strikes")
            return None

        closest_strike_price = strike_price(closest_strike_symbol)

        # Commodities need wider fence due to larger strike intervals (50 points)
        if analysis_symbol in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
            fence_percentages = [2.0, 5.0, 10.0]  # Progressive expansion for commodities
        elif is_nifty:
            fence_percentages = [1.0, 2.0, 5.0]  # Progressive expansion for indices
        else:
            fence_percentages = [2.0, 5.0, 10.0]  # Progressive expansion for equities
        
        # Try progressively wider fences
        for fence_percentage in fence_percentages:
            fence_min = closest_strike_price * (1 - fence_percentage / 100)
            fence_max = closest_strike_price * (1 + fence_percentage / 100)
            print(f"[DEBUG] Trying fence: {fence_percentage}% ({fence_min:.2f} - {fence_max:.2f})")

            # Filter strikes within the dynamic fence using strike_price helper
            strikes_within_fence = [
                item for item in nearest_expiry_symbols
                if fence_min <= strike_price(item) <= fence_max
            ]

            if not strikes_within_fence:
                print(f"[DEBUG] No strikes within {fence_percentage}% fence, expanding...")
                continue

            # Apply directional ITM filter:
            # - bullish_momentum_setup (CE) -> strike must be LESS THAN price_input (call ITM)
            # - bearish_momentum_setup (PE) -> strike must be GREATER THAN price_input (put ITM)
            directional_filtered = []
            for s in strikes_within_fence:
                sp = strike_price(s)
                if buy_indicator == "bullish_momentum_setup" and sp < price_input:
                    directional_filtered.append(s)
                elif buy_indicator == "bearish_momentum_setup" and sp > price_input:
                    directional_filtered.append(s)
                # If buy_indicator not one of above, keep all (defensive)
                elif buy_indicator not in ["bullish_momentum_setup", "bearish_momentum_setup"]:
                    directional_filtered.append(s)

            if not directional_filtered:
                print(f"[DEBUG] No strikes match directional constraints at {fence_percentage}%, expanding...")
                continue

            # Prepare token list for quote_data API call
            nfo_tokens = [str(strike.get('token')) for strike in directional_filtered if strike.get('token') is not None]
            if not nfo_tokens:
                print("[DEBUG] No tokens available for strikes after directional filter")
                continue

            # Determine the correct exchange for quote request
            if analysis_symbol in ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"]:
                quote_exchange = "MCX"
            else:
                quote_exchange = "NFO"

            # Build quote_data request payload
            quote_request = {
                "mode": "FULL",
                "exchangeTokens": {
                    quote_exchange: nfo_tokens
                }
            }

            # Validate user_db structure
            if not isinstance(user_db, (list, tuple)) or len(user_db) < 1:
                raise ValueError("Invalid user_db format")

            quote_response = safe_quote_data(quote_request['mode'], quote_request['exchangeTokens'], user_db)

            # Apply volume/volatility checks
            valid_strikes = []
            all_failed_due_to_oi_or_time = True  # Track if failures are specifically OI/time related
            
            if quote_response is None:
                print("[DEBUG] Quote response was None")
                continue

            if quote_response.get('status') and quote_response.get('data'):
                # Build token-to-quote mapping using string keys for stability
                fetched = quote_response['data'].get('fetched', [])
                token_data = {str(item.get('symbolToken')): item for item in fetched if 'symbolToken' in item}

                # Process each strike
                for strike in directional_filtered:
                    token = str(strike.get('token'))
                    print(f"[DEBUG] Checking token: {token}")
                    if token not in token_data:
                        print(f"[WARNING] Token {token} missing in quote response")
                        continue

                    quote_item = token_data[token]
                    
                    # Check if this strike has valid OI and trade time
                    oi = quote_item.get('opnInterest', 0)
                    trade_time = quote_item.get('exchTradeTime', '')
                    
                    # If has valid OI or recent trade time, it's not just an OI/time issue
                    if oi > 0 or (trade_time and '1970' not in trade_time):
                        all_failed_due_to_oi_or_time = False

                    # Build formatted dictionary for volatility check
                    formatted_data = {
                        # Preserve original strike data
                        **strike,

                        # Add fields needed for volume_volatility_check (safe get)
                        "opnInterest": oi,
                        "exchTradeTime": trade_time,
                        "totBuyQuan": quote_item.get('totBuyQuan', 0),
                        "totSellQuan": quote_item.get('totSellQuan', 0),
                        "depth": quote_item.get('depth', {'buy': [], 'sell': []})
                    }

                    print(f"[DEBUG] Formatted data for volatility check: {formatted_data}")
                    print(f"[DEBUG] Thresholds: {thresholds}")
                    print(formatted_data)
                    try:
                        if volume_volatility_check(formatted_data, thresholds, analysis_symbol):
                            valid_strikes.append(formatted_data)
                    except Exception as e:
                        print(f"[ERROR] volume_volatility_check raised exception: {e}")
                        # skip problematic strike but continue
                        continue

            if valid_strikes:
                # Calculate OI-based score for each strike
                # Score = 60% OI weight + 40% nearness weight
                max_oi = max(s.get('opnInterest', 0) for s in valid_strikes)
                distances = [abs(price_input - strike_price(s)) for s in valid_strikes]
                max_distance = max(distances) if distances else 1
                
                def calculate_strike_score(strike):
                    oi = strike.get('opnInterest', 0)
                    distance = abs(price_input - strike_price(strike))
                    
                    oi_score = (oi / max_oi) if max_oi > 0 else 0
                    nearness_score = 1 - (distance / max_distance) if max_distance > 0 else 0
                    
                    return oi_score * 0.6 + nearness_score * 0.4
                
                # Select strike with highest combined score
                best_strike = max(valid_strikes, key=calculate_strike_score)
                best_score = calculate_strike_score(best_strike)
                print(f"[DEBUG] Best strike (score={best_score:.3f}, OI={best_strike.get('opnInterest', 0)}): {best_strike.get('symbol')}")
                return best_strike
            else:
                # If all failures were due to OI=0 or old trade time, try wider fence
                if all_failed_due_to_oi_or_time:
                    print(f"[DEBUG] All strikes at {fence_percentage}% failed due to OI=0 or stale trade time, expanding fence...")
                    continue
                else:
                    # Other issues (like bid-ask spread), don't expand
                    print("[DEBUG] No valid strikes after volume/volatility checks (non-OI/time issues)")
                    return None
        
        # All fence expansions exhausted
        print("[DEBUG] No valid strikes found after all fence expansions")
        return None

    # EQUITY: Return the first filtered symbol
    elif segment == "Equity":
        if is_nifty:
            print('[DEBUG] NIFTY equity selection blocked')
            return None
        if filtered_symbols:
            print(f"[DEBUG] Selected Equity Symbol: {filtered_symbols[0]}")
            return filtered_symbols[0]

    print("[DEBUG] No matching symbols found")
    return None
