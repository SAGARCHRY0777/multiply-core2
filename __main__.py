from functions.fetch_historical_trades import fetch_historical_trades

if __name__ == "__main__":
    print("Running Fetch Historical Trades")
    fetch_historical_trades()

"""
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')

# Target interval
INTERVALS = ['FIFTEEN_MINUTE']

# Valid 15-minute marks to run on
VALID_MINUTES = (0, 15, 30, 45)

# State file path
STATE_FILE = os.path.join(os.path.dirname(__file__), 'execution_state.json')


# =============================================================================
# State Management
# =============================================================================

def load_state() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading state file: {e}")
    return None


def save_state(last_step: str, interval: str, minute_mark: int, comm_time_iso: str) -> None:
    state = {
        'last_step': last_step,
        'interval': interval,
        'minute_mark': minute_mark,
        'comm_time': comm_time_iso,
        'timestamp': datetime.now(IST).isoformat()
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved state: last_step={last_step}, interval={interval}, minute_mark={minute_mark}")
    except Exception as e:
        logger.error(f"Error saving state file: {e}")


def clear_state() -> None:
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            logger.info("State file cleared")
    except Exception as e:
        logger.error(f"Error clearing state file: {e}")


def save_analysis_completed(minute_mark: int, comm_time: datetime) -> None:
    state = {
        'last_step': 'analysis_completed',
        'interval': INTERVALS[0],
        'minute_mark': minute_mark,
        'hour': comm_time.hour,
        'date': comm_time.strftime('%Y-%m-%d'),
        'comm_time': comm_time.isoformat(),
        'timestamp': datetime.now(IST).isoformat()
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved analysis_completed state: minute_mark={minute_mark}, hour={comm_time.hour}")
    except Exception as e:
        logger.error(f"Error saving analysis completed state: {e}")


# =============================================================================
# Data Loading
# =============================================================================

def load_initial_data():
    from functions.data_utils import (
        fetch_config, load_user_db, get_all_symbol_data, filter_symbol_data
    )
    from functions.data_utils import get_profit_dict
    
    logger.info("Loading configuration and data...")
    
    config = fetch_config()
    all_symbols_data = get_all_symbol_data()
    user_db = load_user_db()
    profit_dict = get_profit_dict()
    
    # Get equity symbols (filtered symbol names)
    equity_symbol_dicts = filter_symbol_data(all_symbols_data)
    equity_symbols = [sd['name'] for sd in equity_symbol_dicts]
    
    logger.info(f"Loaded config, {len(all_symbols_data)} symbols, {len(user_db)} users")
    logger.info(f"Equity symbols: {len(equity_symbols)}")
    
    return config, all_symbols_data, user_db, profit_dict, equity_symbols


# =============================================================================
# Execution Steps
# =============================================================================

def run_fetch_step(comm_time: datetime) -> bool:
    from functions.fetch_candle_data_live import fetch_candle_data_live
    
    try:
        logger.info("=" * 60)
        logger.info(f"STEP: fetch_candle_data_live at {comm_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        logger.info("=" * 60)
        
        config, all_symbols_data, login_details_list, _, equity_symbols = load_initial_data()
        
        fetch_candle_data_live(
            config=config,
            comm_time=comm_time,
            intervals=INTERVALS,
            all_symbols_data=all_symbols_data,
            login_details_list=login_details_list,
            equity_symbols=equity_symbols
        )
        
        logger.info("fetch_candle_data_live completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Error in fetch_candle_data_live: {e}")
        return False


def run_analysis_step(comm_time: datetime) -> bool:
    from functions.entry_point_analysis import entry_point_analysis
    
    try:
        logger.info("=" * 60)
        logger.info(f"STEP: entry_point_analysis at {comm_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        logger.info("=" * 60)
        
        config, all_symbols_data, user_db, profit_dict, equity_symbols = load_initial_data()
        
        entry_point_analysis(
            config=config,
            comm_time=comm_time,
            intervals=INTERVALS,
            all_symbols_data=all_symbols_data,
            profit_dict=profit_dict,
            user_db=user_db,
            equity_symbols=equity_symbols
        )
        
        logger.info("entry_point_analysis completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Error in entry_point_analysis: {e}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    current_time = datetime.now(IST)
    current_minute = current_time.minute
    current_hour = current_time.hour
    current_date = current_time.strftime('%Y-%m-%d')
    interval = INTERVALS[0]
    
    # Determine which 15-minute mark we're in (round down to nearest 15)
    current_15min_mark = (current_minute // 15) * 15
    
    logger.info(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
    logger.info(f"Current 15-minute period: {current_hour}:{current_15min_mark:02d}")
    
    # Load current state
    state = load_state()
    
    # PRIORITY 1: If last step was 'fetch', run analysis immediately
    if state is not None:
        last_step = state.get('last_step')
        cached_comm_time = state.get('comm_time')
        last_minute_mark = state.get('minute_mark')
        last_hour = state.get('hour')
        last_date = state.get('date')
        
        logger.info(f"Loaded state: last_step={last_step}, minute_mark={last_minute_mark}, hour={last_hour}")
        
        if last_step == 'fetch':
            logger.info("Previous step was 'fetch'. Running analysis now...")
            
            # Use cached comm_time from fetch step
            if cached_comm_time:
                comm_time = datetime.fromisoformat(cached_comm_time)
                logger.info(f"Using cached comm_time from fetch step: {comm_time}")
            else:
                comm_time = current_time.replace(second=0, microsecond=0)
                logger.warning("No cached comm_time found, using current time")
            
            success = run_analysis_step(comm_time)
            if success:
                # Save analysis_completed state instead of clearing
                save_analysis_completed(last_minute_mark, comm_time)
            return
        
        # PRIORITY 2: If analysis was already completed for this 15-minute period, skip
        if last_step == 'analysis_completed':
            # Check if we're still in the same 15-minute period
            if (last_date == current_date and 
                last_hour == current_hour and 
                last_minute_mark == current_15min_mark):
                logger.info(f"Analysis already completed for {current_hour}:{current_15min_mark:02d}. Skipping until next 15-minute mark.")
                return
            else:
                logger.info(f"New 15-minute period detected. Previous: {last_hour}:{last_minute_mark:02d}, Current: {current_hour}:{current_15min_mark:02d}")
    
    # PRIORITY 3: Check if we're at a valid 15-minute mark for fetch
    if current_minute not in VALID_MINUTES:
        logger.info(f"Not a valid 15-minute mark (minute={current_minute}). Skipping.")
        return
    
    # Run fetch step
    comm_time = current_time.replace(second=0, microsecond=0)
    logger.info(f"Running fetch step with comm_time: {comm_time}")
    success = run_fetch_step(comm_time)
    if success:
        save_state('fetch', interval, current_minute, comm_time.isoformat())


if __name__ == "__main__":
    main()
"""
