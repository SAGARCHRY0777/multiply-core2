import time
import random
import re
import pytz
from yahooquery import Ticker
from pytz import timezone
from datetime import timedelta, datetime
from functions.data_utils import get_db_url
from pytrends.request import TrendReq
from sqlalchemy import (
    text,
    MetaData,
    create_engine,
    Table,
    Column,
    TEXT,
    Numeric,
    TIMESTAMP,
    String
)
from sqlalchemy.exc import OperationalError
import traceback

# ---------- Engine (healthy pooling + keepalive) ----------
ist = timezone('Asia/Kolkata')
db_url = get_db_url()
engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=5,
    max_overflow=10,
    # connect_args={"sslmode": "require"}  # uncomment if your DB requires SSL
)

metadata = MetaData()

company_mapping = Table(
    'company_mapping', metadata,
    Column('symbol', String, primary_key=True),
    Column('company_name', String)
)

all_indicators_data = Table(
    'all_indicators_data', metadata,
    Column('symbol', TEXT, primary_key=True),
    Column('interval', TEXT, primary_key=True),
    Column('datetime', TIMESTAMP, primary_key=True),
    Column('aroon_up', Numeric),
    Column('aroon_down', Numeric),
    Column('adx', Numeric),
    Column('pdi', Numeric),
    Column('mdi', Numeric),
    Column('elderray_bull_power', Numeric),
    Column('elderray_bear_power', Numeric),
    Column('gator_upper', Numeric),
    Column('gator_lower', Numeric),
    Column('gator_is_upper_expanding', Numeric),
    Column('gator_is_lower_expanding', Numeric),
    Column('hurst', Numeric),
    Column('ichimoku_tenkan_sen', Numeric),
    Column('ichimoku_kijun_sen', Numeric),
    Column('ichimoku_senkou_span_a', Numeric),
    Column('ichimoku_senkou_span_b', Numeric),
    Column('ichimoku_chikou_span', Numeric),
    Column('macd', Numeric),
    Column('macd_signal', Numeric),
    Column('macd_histogram', Numeric),
    Column('supertrend', Numeric),
    Column('vortex_pvi', Numeric),
    Column('vortex_nvi', Numeric),
    Column('alligator_jaw', Numeric),
    Column('alligator_teeth', Numeric),
    Column('alligator_lips', Numeric),
    Column('bollingerbands_upper', Numeric),
    Column('bollingerbands_lower', Numeric),
    Column('donchianchannels_upper', Numeric),
    Column('donchianchannels_lower', Numeric),
    Column('fcb_upper', Numeric),
    Column('fcb_lower', Numeric),
    Column('keltnerchannels_upper', Numeric),
    Column('keltnerchannels_lower', Numeric),
    Column('keltnerchannels_center', Numeric),
    Column('maenvelopes_upper', Numeric),
    Column('maenvelopes_lower', Numeric),
    Column('pivotpoints_pp', Numeric),
    Column('pivotpoints_r1', Numeric),
    Column('pivotpoints_s1', Numeric),
    Column('rollingpivotpoints_pp', Numeric),
    Column('rollingpivotpoints_r1', Numeric),
    Column('rollingpivotpoints_s1', Numeric),
    Column('starc_upper', Numeric),
    Column('starc_lower', Numeric),
    Column('standarddeviationchannels_upper', Numeric),
    Column('standarddeviationchannels_lower', Numeric),
    Column('awesomeoscillator', Numeric),
    Column('cci', Numeric),
    Column('connorsrsi', Numeric),
    Column('dpo', Numeric),
    Column('rsi', Numeric),
    Column('stc', Numeric),
    Column('smi', Numeric),
    Column('stochasticoscillator_k', Numeric),
    Column('stochasticoscillator_d', Numeric),
    Column('stochasticrsi_rsi', Numeric),
    Column('stochasticrsi_signal', Numeric),
    Column('trix', Numeric),
    Column('ultimateoscillator', Numeric),
    Column('williamsr', Numeric),
    Column('chandelierexit', Numeric),
    Column('parabolicsar', Numeric),
    Column('volatilitystop', Numeric),
    Column('williamsfractal_bull', Numeric),
    Column('williamsfractal_bear', Numeric),
    Column('adl', Numeric),
    Column('cmf', Numeric),
    Column('chaikinoscillator', Numeric),
    Column('forceindex', Numeric),
    Column('kvo', Numeric),
    Column('mfi', Numeric),
    Column('obv', Numeric),
    Column('pvo', Numeric),
    Column('alma', Numeric),
    Column('dema', Numeric),
    Column('epma', Numeric),
    Column('ema5', Numeric),
    Column('ema9', Numeric),
    Column('ema13', Numeric),
    Column('ema50', Numeric),
    Column('hilberttransform', Numeric),
    Column('hma', Numeric),
    Column('kama', Numeric),
    Column('mama', Numeric),
    Column('fama', Numeric),
    Column('sma', Numeric),
    Column('smma', Numeric),
    Column('t3', Numeric),
    Column('tema', Numeric),
    Column('vwap', Numeric),
    Column('vwma', Numeric),
    Column('wma', Numeric),
    Column('fishertransform_fisher', Numeric),
    Column('fishertransform_trigger', Numeric),
    Column('zigzag', Numeric),
    Column('atr', Numeric),
    Column('bop', Numeric),
    Column('choppinessindex', Numeric),
    Column('pmo', Numeric),
    Column('pmo_signal', Numeric),
    Column('roc', Numeric),
    Column('truerange', Numeric),
    Column('tsi', Numeric),
    Column('ulcerindex', Numeric),
    Column('slope', Numeric),
    Column('standarddeviation', Numeric),
    Column('pytrends_interest', Numeric)
)

metadata.create_all(engine)

LOCAL_COMPANY_MAP = {}

# ------------------ NEW CODE FOR STORING AND READING START BATCH ------------------ #
START_BATCH_FILE = "batch_info.txt"


def get_start_batch(default=1):
    try:
        with open(START_BATCH_FILE, "r") as f:
            return int(f.read().strip())
    except Exception as e:
        print(e)
        return default


def set_start_batch(batch_number):
    with open(START_BATCH_FILE, "w") as f:
        f.write(str(batch_number))


# --- Helper Functions ---
def clean_company_name(company_name: str) -> str:
    print(f"Cleaning company name: {company_name}")
    cleaned_name = re.sub(r'\s+(private limited|limited)$', '', company_name, flags=re.IGNORECASE)
    cleaned_name = cleaned_name.strip()
    print(f"Cleaned company name: {cleaned_name}")
    return cleaned_name


def lookup_company_name_yahoo(symbol: str) -> str:
    lookup_symbol = symbol if symbol.endswith('.NS') else (symbol + '.NS')
    print(f"Looking up company name from Yahoo for symbol: {lookup_symbol}")
    try:
        ticker = Ticker(lookup_symbol)
        info = ticker.quote_type.get(lookup_symbol)
        if info:
            company_name = info.get('longName') or info.get('shortName')
            if company_name:
                print(f"Fetched company name from Yahoo: {company_name}")
                return clean_company_name(company_name)
        print(f"Company name not found via Yahoo for symbol: {lookup_symbol}")
    except Exception as e:
        print(f"Error fetching company name for symbol {lookup_symbol}: {e}")
    return "None"


def fetch_company_name_from_db(conn, symbol: str) -> str:
    query = text("SELECT company_name FROM company_mapping WHERE symbol = :symbol LIMIT 1")
    result = conn.execute(query, {"symbol": symbol}).fetchone()
    if result:
        return result[0]
    return "None"


def store_company_name_in_db(conn, symbol: str, company_name: str) -> None:
    ins = text("""
        INSERT INTO company_mapping (symbol, company_name)
        VALUES (:symbol, :company_name)
        ON CONFLICT (symbol)
        DO UPDATE SET company_name = EXCLUDED.company_name
    """)
    conn.execute(ins, {"symbol": symbol, "company_name": company_name})


def get_or_create_company_name(symbol: str) -> str:
    if symbol in LOCAL_COMPANY_MAP:
        return LOCAL_COMPANY_MAP[symbol]

    # Short-lived connection(s)
    with engine.begin() as conn:
        db_name = fetch_company_name_from_db(conn, symbol)
        if db_name and db_name != "None":
            LOCAL_COMPANY_MAP[symbol] = db_name
            return db_name

    company_name = lookup_company_name_yahoo(symbol)
    if company_name and company_name != "None":
        with engine.begin() as conn:
            store_company_name_in_db(conn, symbol, company_name)
        LOCAL_COMPANY_MAP[symbol] = company_name
        return company_name

    return "None"


def get_current_datetime_ist():
    print("Fetching current IST datetime")
    return datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(ist)


def chunk_list(lst, chunk_size):
    print(f"Splitting list of length {len(lst)} into chunks of size {chunk_size}")
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def bulk_update_pytrends(update_rows, interval='ONE_DAY', chunk_size=2000, max_retries=3):
    if not update_rows:
        print("No rows to update for pytrends_interest.")
        return

    for start in range(0, len(update_rows), chunk_size):
        chunk = update_rows[start:start+chunk_size]
        values = []
        for sym, d, v in chunk:
            v = int(v) if v is not None else None
            values.append((sym, interval, d, v))

        # Use positional bindparams instead of %s
        placeholders = ",".join(["(:sym{}, :interval{}, :the_date{}, :val{})".format(i, i, i, i)
                                 for i in range(len(values))])

        sql = f"""
            WITH updates(symbol, interval, the_date, val) AS (
                VALUES {placeholders}
            )
            UPDATE all_indicators_data AS a
            SET pytrends_interest = u.val
            FROM updates u
            WHERE a.symbol = u.symbol
              AND a.interval = u.interval
              AND a.datetime >= u.the_date
              AND a.datetime <  (u.the_date + INTERVAL '1 day');
        """

        # Build params dict for SQLAlchemy
        params = {}
        for i, (sym, interval_val, the_date, val) in enumerate(values):
            params[f"sym{i}"] = sym
            params[f"interval{i}"] = interval_val
            params[f"the_date{i}"] = the_date
            params[f"val{i}"] = val

        attempt = 0
        while True:
            try:
                with engine.begin() as conn:
                    conn.execute(text(sql), params)
                print(f"Committed {len(values)} pytrends_interest updates.")
                break
            except OperationalError as e:
                attempt += 1
                if attempt >= max_retries:
                    print(f"FAILED chunk {start}-{start+len(chunk)} after {attempt} attempts: {e}")
                    raise
                sleep_s = 2 ** attempt
                print(f"OperationalError, retrying in {sleep_s}s (attempt {attempt}/{max_retries})...")
                time.sleep(sleep_s)


def  update_missing_trend_data():
    try:
        """
        Process each symbol to fill missing pytrends_interest data in all_indicators_data.
        Fetch data in ~3-month intervals from pytrends and store in the pytrends_interest column.

        Rules:
          - Open DB connections only when reading/writing (short-lived).
          - Use bulk, sargable updates.
          - Stop after 3 AM IST.
        """
        print("Starting update of missing trend indicator data for symbols.")
        min_allowed_date = datetime(2015, 1, 1).date()

        # ---- 1) Gather symbols (short connection) ----
        with engine.connect() as conn:
            print("Fetching distinct symbols from all_indicators_data...")
            sel_symbols = text("SELECT DISTINCT symbol FROM all_indicators_data")
            symbols_list = [row[0] for row in conn.execute(sel_symbols)]
            print(f"Found {len(symbols_list)} symbols in the database.")

        symbol_data = {}

        # ---- 2) For each symbol, compute earliest/missing ranges (short connections) ----
        with engine.connect() as conn:
            for symbol in symbols_list:
                earliest_row = conn.execute(text("""
                    SELECT datetime
                    FROM all_indicators_data
                    WHERE symbol = :symbol
                    ORDER BY datetime ASC
                    LIMIT 1
                """), {"symbol": symbol}).fetchone()

                if not earliest_row:
                    print(f"No candle/indicator data found for symbol: {symbol}")
                    continue

                earliest_date = earliest_row[0].date()
                print(f"Earliest date for {symbol} is {earliest_date}")

                max_trend_row = conn.execute(text("""
                    SELECT MAX(datetime) AS max_dt
                    FROM all_indicators_data
                    WHERE symbol = :symbol
                      AND pytrends_interest IS NOT NULL
                """), {"symbol": symbol}).fetchone()

                if max_trend_row and max_trend_row[0]:
                    missing_start_date = max_trend_row[0].date() + timedelta(days=1)
                else:
                    missing_start_date = earliest_date

                if missing_start_date < min_allowed_date:
                    print(
                        f"For symbol {symbol}, missing start date {missing_start_date} < {min_allowed_date}. "
                        f"Updating rows to pytrends_interest=0 for those dates."
                    )
                    # Sargable (no DATE(datetime))
                    with engine.begin() as wconn:
                        wconn.execute(text("""
                            UPDATE all_indicators_data
                            SET pytrends_interest = 0
                            WHERE symbol = :symbol
                              AND datetime < :min_date
                        """), {"symbol": symbol, "min_date": min_allowed_date})
                    missing_start_date = min_allowed_date

                if missing_start_date > get_current_datetime_ist().date():
                    print(f"No missing trend data for {symbol} (start date {missing_start_date} > today).")
                    continue

                missing_count = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM all_indicators_data
                    WHERE symbol = :symbol
                      AND interval = 'ONE_DAY'
                      AND pytrends_interest IS NULL
                """), {"symbol": symbol}).scalar()

                full_company_name = get_or_create_company_name(symbol) or symbol
                keyword = f"{full_company_name.lower()} share"
                print(f"Using keyword '{keyword}' for pytrends for symbol {symbol} (missing rows: {missing_count}).")

                symbol_data[symbol] = {
                    "missing_start_date": missing_start_date,
                    "keyword": keyword,
                    "missing_count": missing_count
                }

        if not symbol_data:
            print("No symbols require trend data update.")
            return

        # ---- 3) Sort + batch symbols (no DB connection held) ----
        sorted_symbol_data = dict(sorted(symbol_data.items(), key=lambda item: item[1]["missing_count"]))

        batch_size = 5
        symbol_batches = list(chunk_list(list(sorted_symbol_data.items()), batch_size))
        print(f"Processing symbols in batches of {batch_size}.")
        global_end_date = get_current_datetime_ist().date()
        request_count = 0

        # Resume support
        start_batch = get_start_batch(default=1)
        print(f"Resuming from batch {start_batch}...")

        for i, batch in enumerate(symbol_batches, start=1):
            current_time = get_current_datetime_ist()
            """
            if current_time.hour >= 3:
                print("Current time is after 3 AM IST. Stopping the update process.")
                return
            """
            print(i)
            print(start_batch)
            if i < start_batch:
                continue

            set_start_batch(i)
            print(f"Processing batch {i}/{len(symbol_batches)} with {len(batch)} symbols.")

            batch_missing_start = min(item[1]["missing_start_date"] for item in batch)
            keyword_symbol_map = {data["keyword"]: sym for (sym, data) in batch}

            current_start = batch_missing_start
            pytrends = TrendReq(hl='en-US', tz=330)

            batch_all_updates = []

            # ---- 4) Pytrends fetch loop (no DB connection held) ----
            while current_start < global_end_date:
                """
                if get_current_datetime_ist().hour >= 3:
                    print("Current time is after 3 AM IST. Stopping the update process.")
                    return
                """
                current_end = current_start + timedelta(days=90)  # ~3 months
                timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
                print(f"[Batch {i}] Fetching pytrends data for timeframe {timeframe} for up to {len(batch)} symbols.")

                keywords_batch = [data["keyword"] for (sym, data) in batch if data["missing_start_date"] < current_end]

                if not keywords_batch:
                    print("No keywords in this batch need data for the current window.")
                    current_start = current_end
                    continue

                try:
                    while True:
                        """
                        if get_current_datetime_ist().hour >= 3:
                            print("Current time is after 3 AM IST. Stopping the update process.")
                            return
                        """
                        try:
                            pytrends.build_payload(kw_list=keywords_batch, timeframe=timeframe, geo='IN')
                            trend_df = pytrends.interest_over_time()
                            break
                        except Exception as e:
                            print(f"Error fetching data for timeframe {timeframe}: {e}")
                            sleep_time = random.randint(5, 15)
                            print(f"Sleeping for {sleep_time} seconds before retrying...")
                            time.sleep(sleep_time)

                    if not trend_df.empty:
                        print(f"Pytrends data received with shape {trend_df.shape}.")
                        if 'isPartial' in trend_df.columns:
                            trend_df = trend_df.drop(columns='isPartial')
                            print("Dropped 'isPartial' column from pytrends data.")

                        updates_for_this_timeframe = []
                        for idx, row in trend_df.iterrows():
                            date_for_db = idx.date()  # use + timedelta(days=1) if you want "next day"
                            for kw_col in trend_df.columns:
                                if kw_col in keyword_symbol_map:
                                    symbol_for_kw = keyword_symbol_map[kw_col]
                                    value = row[kw_col]
                                    if value is not None:
                                        value = int(value)
                                    updates_for_this_timeframe.append((symbol_for_kw, date_for_db, value))

                        batch_all_updates.extend(updates_for_this_timeframe)
                        del trend_df
                    else:
                        print("Pytrends returned an empty dataframe for this timeframe.")

                except Exception as e:
                    print(f"Error fetching data for timeframe {timeframe}: {e}")

                request_count += 1
                if request_count % random.randint(4, 5) == 0:
                    sleep_time = random.randint(5, 10)
                    print(f"Sleeping for {sleep_time} seconds after {request_count} requests...")
                    time.sleep(sleep_time)

                current_start = current_end

            # ---- 5) Bulk write this batch (short transaction) ----
            if batch_all_updates:
                print(f"[Batch {i}] Updating pytrends_interest for {len(batch_all_updates)} rows collected in this batch.")
                bulk_update_pytrends(batch_all_updates, interval='ONE_DAY', chunk_size=2000, max_retries=3)

                print(f"Successfully processed batch {i}/{len(symbol_batches)} for missing trend data.")
            else:
                print(f"[Batch {i}] No pytrends data to update for this batch.")

        print("Missing trend indicator data update completed for all symbols.")
    except Exception as e:
        print(f"An error occurred in update_missing_trend_data: {e}\n\n{traceback.format_exc()}")
        return False
