"""
Post-Market Data Pipeline Runner

Executes the following in order:
1. fetch_candle_data_post_market - Get candle and indicator data
2. update_profit_dict - Calculate profit dictionary
3. update_boson_system - Train ML models

Usage:
    python run_post_market_pipeline.py [--step STEP]
    
    --step: Optional, run only specific step (1, 2, or 3)
            If not specified, runs all steps.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import pytz

sys.path.insert(0, '/home/azureuser/multiply_core')

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

import psutil

def log_memory_usage(tag=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"[MEMORY] {tag} - Usage: {mem:.2f} MB")

IST = pytz.timezone('Asia/Kolkata')


def run_fetch_candle_data():
    """Step 1: Fetch candle and indicator data post-market."""
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching Candle Data Post-Market")
    logger.info("=" * 60)
    
    from functions.data_utils import (
        fetch_config, load_user_db, get_all_symbol_data, filter_symbol_data
    )
    from functions.fetch_candle_data_post_market import fetch_candle_data_post_market
    from datetime import time
    
    config = fetch_config()
    all_symbols_data = get_all_symbol_data()
    login_details_list = load_user_db()
    
    comm_time = datetime.now(IST)
    intervals = config.get('allowed_indices_intervals', [
        'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 
        'ONE_HOUR', 'ONE_DAY'
    ])
    
    # Get equity symbols (Nifty 50 stocks)
    equity_symbols = filter_symbol_data(all_symbols_data)
    
    logger.info(f"Config loaded: {len(config)} keys")
    logger.info(f"All Symbols: {len(all_symbols_data)}")
    logger.info(f"Equity Symbols: {len(equity_symbols) if equity_symbols else 0}")
    logger.info(f"Users: {len(login_details_list)}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Current time: {comm_time.strftime('%H:%M:%S')}")
    
    # Show what will run based on time
    current_time = comm_time.time()
    if time(16, 0) <= current_time <= time(18, 0):
        logger.info("✅ Equity market hours (4-6 PM) - Indices & Equities will run")
    else:
        logger.warning(f"⚠️ Outside equity hours (4-6 PM) - Current: {current_time}")
    
    if time(0, 0) <= current_time <= time(2, 0):
        logger.info("✅ Commodity market hours (12-2 AM) - Commodities will run")
    else:
        logger.info(f"ℹ️ Outside commodity hours (12-2 AM) - Current: {current_time}")
    
    fetch_candle_data_post_market(
        config=config,
        comm_time=comm_time,
        intervals=intervals,
        all_symbols_data=all_symbols_data,
        login_details_list=login_details_list,
        equity_symbols=equity_symbols
    )
    
    logger.info("Step 1 COMPLETE: Candle data fetched")




def run_update_profit_dict():
    """Step 2: Update profit dictionary."""
    logger.info("=" * 60)
    logger.info("STEP 2: Updating Profit Dictionary")
    logger.info("=" * 60)
    
    from functions.data_utils import fetch_config
    # from functions.update_profit_dict import update_profit_dict
    # from functions.update_profit_dict import update_profit_dict
    from functions.fetch_historical_trades import fetch_historical_trades as update_profit_dict
    
    config = fetch_config()
    intervals = [
        'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 
        'ONE_HOUR', 'ONE_DAY'
    ]
    
    logger.info(f"Intervals: {intervals}")
    # success = update_profit_dict(config=config, intervals=intervals)
    # HARDCODED TEST SYMBOLS
    test_symbols = ['NIFTY', 'RELIANCE']
    logger.info(f"Running TEST MODE for symbols: {test_symbols}")

    log_memory_usage("Before Profit Dict Update")
    success = update_profit_dict(intervals=intervals, symbols=test_symbols)
    log_memory_usage("After Profit Dict Update")
    
    if success:
        logger.info("Step 2 COMPLETE: Profit dict updated")
    else:
        logger.error("Step 2 FAILED: Profit dict update failed")
    
    return success


def run_update_boson_system():
    """Step 3: Update boson system (train models)."""
    logger.info("=" * 60)
    logger.info("STEP 3: Updating Boson System")
    logger.info("=" * 60)
    
    from functions.update_boson_system import update_boson_system
    
    result = update_boson_system(
        steps=['generate_datasets', 'train_general_models']
    )
    
    if result:
        logger.info("Step 3 COMPLETE: Boson system updated")
    else:
        logger.warning("Step 3: Some steps may have failed")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run post-market data pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], 
                        help='Run only specific step (1: candles, 2: profit_dict, 3: boson)')
    args = parser.parse_args()
    
    start_time = datetime.now(IST)
    logger.info(f"Pipeline started at: {start_time}")
    
    try:
        if args.step is None or args.step == 1:
            run_fetch_candle_data()
        
        if args.step is None or args.step == 2:
            run_update_profit_dict()
        
        if args.step is None or args.step == 3:
            run_update_boson_system()
        
        end_time = datetime.now(IST)
        duration = end_time - start_time
        logger.info(f"Pipeline completed in: {duration}")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
