from datetime import datetime
from sqlalchemy import create_engine, TIMESTAMP, Column, TEXT, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pytz
import logging
from functions.data_utils import get_db_url

# We keep basic logging setup, but will only use error logs plus a couple of info logs for start and completion
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# TimescaleDB connection details
db_url = get_db_url()
engine = create_engine(
    db_url, 
    connect_args={
        'sslmode': 'require',
        'connect_timeout': 60,  # 60 second connection timeout
        'options': '-c statement_timeout=120000'  # 120 second query timeout (in ms)
    },
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600  # Recycle connections after 1 hour
)


Base = declarative_base()


class AllIndicatorData(Base):
    __tablename__ = 'all_indicators_data'

    symbol = Column(TEXT, primary_key=True)
    interval = Column(TEXT, primary_key=True)
    datetime = Column(TIMESTAMP, primary_key=True)
    aroon_up = Column(Numeric)
    aroon_down = Column(Numeric)
    adx = Column(Numeric)
    pdi = Column(Numeric)
    mdi = Column(Numeric)
    elderray_bull_power = Column(Numeric)
    elderray_bear_power = Column(Numeric)
    gator_upper = Column(Numeric)
    gator_lower = Column(Numeric)
    gator_is_upper_expanding = Column(Numeric)
    gator_is_lower_expanding = Column(Numeric)
    hurst = Column(Numeric)
    ichimoku_tenkan_sen = Column(Numeric)
    ichimoku_kijun_sen = Column(Numeric)
    ichimoku_senkou_span_a = Column(Numeric)
    ichimoku_senkou_span_b = Column(Numeric)
    ichimoku_chikou_span = Column(Numeric)
    macd = Column(Numeric)
    macd_signal = Column(Numeric)
    macd_histogram = Column(Numeric)
    supertrend = Column(Numeric)
    vortex_pvi = Column(Numeric)
    vortex_nvi = Column(Numeric)
    alligator_jaw = Column(Numeric)
    alligator_teeth = Column(Numeric)
    alligator_lips = Column(Numeric)
    bollingerbands_upper = Column(Numeric)
    bollingerbands_lower = Column(Numeric)
    donchianchannels_upper = Column(Numeric)
    donchianchannels_lower = Column(Numeric)
    fcb_upper = Column(Numeric)
    fcb_lower = Column(Numeric)
    keltnerchannels_upper = Column(Numeric)
    keltnerchannels_lower = Column(Numeric)
    keltnerchannels_center = Column(Numeric)
    maenvelopes_upper = Column(Numeric)
    maenvelopes_lower = Column(Numeric)
    pivotpoints_pp = Column(Numeric)
    pivotpoints_r1 = Column(Numeric)
    pivotpoints_s1 = Column(Numeric)
    rollingpivotpoints_pp = Column(Numeric)
    rollingpivotpoints_r1 = Column(Numeric)
    rollingpivotpoints_s1 = Column(Numeric)
    starc_upper = Column(Numeric)
    starc_lower = Column(Numeric)
    standarddeviationchannels_upper = Column(Numeric)
    standarddeviationchannels_lower = Column(Numeric)
    awesomeoscillator = Column(Numeric)
    cci = Column(Numeric)
    connorsrsi = Column(Numeric)
    dpo = Column(Numeric)
    rsi = Column(Numeric)
    stc = Column(Numeric)
    smi = Column(Numeric)
    stochasticoscillator_k = Column(Numeric)
    stochasticoscillator_d = Column(Numeric)
    stochasticrsi_rsi = Column(Numeric)
    stochasticrsi_signal = Column(Numeric)
    trix = Column(Numeric)
    ultimateoscillator = Column(Numeric)
    williamsr = Column(Numeric)
    chandelierexit = Column(Numeric)
    parabolicsar = Column(Numeric)
    volatilitystop = Column(Numeric)
    williamsfractal_bull = Column(Numeric)
    williamsfractal_bear = Column(Numeric)
    adl = Column(Numeric)
    cmf = Column(Numeric)
    chaikinoscillator = Column(Numeric)
    forceindex = Column(Numeric)
    kvo = Column(Numeric)
    mfi = Column(Numeric)
    obv = Column(Numeric)
    pvo = Column(Numeric)
    alma = Column(Numeric)
    dema = Column(Numeric)
    epma = Column(Numeric)
    ema5 = Column(Numeric)
    ema9 = Column(Numeric)
    ema13 = Column(Numeric)
    ema50 = Column(Numeric)
    hilberttransform = Column(Numeric)
    hma = Column(Numeric)
    kama = Column(Numeric)
    mama = Column(Numeric)
    fama = Column(Numeric)
    sma = Column(Numeric)
    smma = Column(Numeric)
    t3 = Column(Numeric)
    tema = Column(Numeric)
    vwap = Column(Numeric)
    vwma = Column(Numeric)
    wma = Column(Numeric)
    fishertransform_fisher = Column(Numeric)
    fishertransform_trigger = Column(Numeric)
    zigzag = Column(Numeric)
    atr = Column(Numeric)
    bop = Column(Numeric)
    choppinessindex = Column(Numeric)
    pmo = Column(Numeric)
    pmo_signal = Column(Numeric)
    roc = Column(Numeric)
    truerange = Column(Numeric)
    tsi = Column(Numeric)
    ulcerindex = Column(Numeric)
    slope = Column(Numeric)
    standarddeviation = Column(Numeric)
    pytrends_interest = Column(Numeric)


def format_indicator_data(symbol, interval, indicator_data):
    """
    Format indicator data for TimescaleDB insertion, handling timezone information.
    Includes all indicators in the provided data.
    """

    formatted_data = {}

    if not indicator_data:
        return []

    # Initialize dates based on the first indicator
    first_indicator_key = list(indicator_data.keys())[2]

    for date, _ in indicator_data[first_indicator_key]:
        try:
            # Parse the datetime and ensure it is in IST
            parsed_datetime = datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
            ist_datetime = parsed_datetime.astimezone(pytz.timezone('Asia/Kolkata'))
        except ValueError:
            # If no timezone info, assume UTC and convert to IST
            parsed_datetime = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)
            ist_datetime = parsed_datetime.astimezone(pytz.timezone('Asia/Kolkata'))
        except Exception as e:
            logging.error(f"ERROR parsing date {date}: {str(e)}")
            continue

        formatted_data[date] = {
            "symbol": symbol,
            "interval": interval,
            "datetime": ist_datetime
        }

    if not formatted_data:
        logging.error("ERROR: No dates parsed into formatted_data")
        return []

    # Add indicator values to corresponding dates
    for indicator, values in indicator_data.items():
        for date, value in values:
            if date in formatted_data:
                # Convert booleans to numeric explicitly
                value = 1 if isinstance(value, bool) and value else 0 if isinstance(value, bool) else value
                formatted_data[date][indicator] = value
            else:
                pass

    if not formatted_data:
        logging.error("ERROR: No data in formatted_data after adding indicators")

    return list(formatted_data.values())  # Convert to a list of rows


def save_all_indicator_data_by_interval(symbol_interval_data):
    """
    Bulk insert or update (overwrite) all indicator data for all symbols within each interval in TimescaleDB.
    Data is saved in chunks to manage memory efficiently.
    """
    # Start log
    logging.info("Starting save all indicator data by interval....")

    # Validate database connection
    try:
        with engine.connect():
            pass
    except Exception as e:
        logging.error(f"ERROR: Database connection failed: {str(e)}")
        return False

    Session = sessionmaker(bind=engine)

    if not symbol_interval_data:
        logging.error("ERROR: No data provided to save")
        return False

    all_formatted_data = []
    for idx, (symbol, interval, indicator_data) in enumerate(symbol_interval_data):
        if not indicator_data:
            # Not an error, just empty data; silently skip
            continue

        try:
            formatted_data = format_indicator_data(symbol, interval, indicator_data)

            if formatted_data:
                all_formatted_data.extend(formatted_data)
        except Exception as e:
            logging.error(f"ERROR formatting data for {symbol} {interval}: {str(e)}")
            continue

    if not all_formatted_data:
        logging.error("ERROR: No data to save after formatting")
        return False

    chunk_size = 2000  # Reduced to 2000 for faster commits and to avoid timeouts
    total_rows = len(all_formatted_data)
    processed_rows = 0
    error_count = 0

    for i in range(0, total_rows, chunk_size):
        chunk = all_formatted_data[i:i + chunk_size]
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            session = Session()
            try:
                # Create insert statement for the chunk
                stmt = insert(AllIndicatorData.__table__).values(chunk)

                # Update all columns except the primary keys in case of conflict
                update_dict = {
                    col.name: stmt.excluded[col.name]
                    for col in AllIndicatorData.__table__.columns
                    if col.name not in ['symbol', 'interval', 'datetime']
                }

                stmt = stmt.on_conflict_do_update(
                    constraint='all_indicators_data_pkey',
                    set_=update_dict
                )

                # Execute the statement for this chunk
                session.execute(stmt)
                session.commit()

                processed_rows += len(chunk)
                break  # Success - exit retry loop
                
            except Exception as e:
                error_msg = str(e).lower()
                session.rollback()
                
                # Check if it's a timeout error
                if 'timeout' in error_msg or 'timed out' in error_msg:
                    retry_count += 1
                    if retry_count < max_retries:
                        import time
                        wait_time = retry_count * 2  # Exponential backoff: 2s, 4s, 6s
                        logging.warning(f"Timeout error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logging.error(f"Max retries reached for chunk {i} to {i + chunk_size} due to timeout")
                        error_count += 1
                else:
                    # Non-timeout error - log and continue
                    error_count += 1
                    logging.error(f"ERROR processing chunk {i} to {i + chunk_size}: {str(e)}")
                    if chunk:
                        logging.error(f"First row in failing chunk: {chunk[0]}")
                    break  # Don't retry non-timeout errors

                if error_count >= 5:  # Increased tolerance to 5 errors
                    logging.error("Too many errors encountered, stopping process")
                    session.close()
                    return False
            finally:
                session.close()

    success = processed_rows > 0
    if success:
        logging.info(f"Successfully saved {processed_rows}/{total_rows} rows of data")
    else:
        logging.info("No data was saved successfully")

    return success
