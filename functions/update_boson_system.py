"""
Update Boson System - Main Orchestrator

The main entry point for the Boson ML training pipeline.
Orchestrates dataset generation, feature computation, and model training.

Usage:
    from functions.update_boson_system import update_boson_system
    
    # Run full pipeline
    update_boson_system()
    
    # Run specific intervals
    update_boson_system(intervals=['FIFTEEN_MINUTE'])
"""

import gc
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import pytz

from functions.boson_dataset_generator import (
    update_buy_datasets_main,
    combine_and_upload_buy_datasets
)
from functions.boson_model_trainer import (
    train_general_models,
    train_interval_models,
    train_market_models,
    train_symbol_models
)
from functions.boson_storage import BosonStorage, get_storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')
STATE_FILE = "boson_update_state.json"


# =============================================================================
# Configuration
# =============================================================================

def _load_config() -> dict:
    """Load boson configuration from config.json."""
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('boson', {})
    return {}


DEFAULT_INTERVALS = [
    'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE',
    'ONE_HOUR', 'ONE_DAY', 'ONE_WEEK', 'ONE_MONTH'
]


# =============================================================================
# State Management
# =============================================================================

def _load_state() -> dict:
    """Load pipeline state from file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not decode state file, resetting")
    return {}


def _save_state(state: dict):
    """Save pipeline state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def _clear_state():
    """Clear pipeline state."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


# =============================================================================
# Pipeline Steps
# =============================================================================

def step_generate_datasets(
    intervals: List[str],
    features: List[str] = None,
    storage: BosonStorage = None
) -> bool:
    """
    Step 1: Generate training datasets for all intervals.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Generating Datasets")
    logger.info("=" * 60)
    
    try:
        success = update_buy_datasets_main(intervals)
        logger.info(f"Dataset generation: {'success' if success else 'failed'}")
        return success
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        logger.error(traceback.format_exc())
        return False


def step_combine_datasets(
    intervals: List[str] = None,
    storage: BosonStorage = None
) -> bool:
    """
    Step 2: Combine interval datasets into general datasets.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Combining Datasets for General Models")
    logger.info("=" * 60)
    
    try:
        success = combine_and_upload_buy_datasets(intervals, storage)
        logger.info(f"Dataset combination: {'success' if success else 'failed'}")
        return success
    except Exception as e:
        logger.error(f"Dataset combination failed: {e}")
        logger.error(traceback.format_exc())
        return False


def step_train_general_models(
    backend: str = 'xgboost',
    storage: BosonStorage = None
) -> bool:
    """
    Step 3: Train general (Tier 1) models.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Training General Models (Tier 1)")
    logger.info("=" * 60)
    
    try:
        models = train_general_models(storage, backend)
        logger.info(f"Trained {len(models)} general models")
        return len(models) > 0
    except Exception as e:
        logger.error(f"General model training failed: {e}")
        logger.error(traceback.format_exc())
        return False


def step_train_interval_models(
    intervals: List[str],
    backend: str = 'xgboost',
    fine_tune: bool = True,
    storage: BosonStorage = None
) -> bool:
    """
    Step 4: Train interval-specific (Tier 2) models.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Training Interval Models (Tier 2)")
    logger.info("=" * 60)
    
    try:
        models = train_interval_models(intervals, storage, backend, fine_tune)
        logger.info(f"Trained {len(models)} interval models")
        return len(models) > 0
    except Exception as e:
        logger.error(f"Interval model training failed: {e}")
        logger.error(traceback.format_exc())
        return False


# =============================================================================
# Main Pipeline
# =============================================================================

def update_boson_system(
    intervals: List[str] = None,
    features: List[str] = None,
    backend: str = None,
    skip_dataset_generation: bool = False,
    skip_general_training: bool = False
) -> bool:
    """
    Run the complete Boson ML training pipeline.
    
    Args:
        intervals: Intervals to process (default: all)
        features: Feature names to compute (default: standard preset)
        backend: Model backend ('xgboost', 'lightgbm', 'rf')
        skip_dataset_generation: Skip dataset generation step
        skip_general_training: Skip general model training
        
    Returns:
        True if successful, False otherwise
    """
    start_time = datetime.now(IST)
    logger.info("=" * 60)
    logger.info("STARTING BOSON SYSTEM UPDATE")
    logger.info(f"Time: {start_time}")
    logger.info("=" * 60)
    
    # Load configuration
    config = _load_config()
    intervals = intervals or config.get('intervals', DEFAULT_INTERVALS)
    backend = backend or config.get('model_backend', 'xgboost')
    features = features or config.get('enabled_features')
    
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Features: {features or 'standard preset'}")
    
    # Initialize storage
    storage = get_storage()
    
    # Load state for resume
    state = _load_state()
    
    try:
        gc.enable()
        
        # Step 1: Generate datasets
        if not skip_dataset_generation:
            if not state.get('datasets_generated'):
                if step_generate_datasets(intervals, features, storage):
                    state['datasets_generated'] = True
                    _save_state(state)
                else:
                    logger.warning("Dataset generation incomplete, continuing...")
        else:
            logger.info("Skipping dataset generation")
        
        gc.collect()
        
        # Step 2: Combine datasets
        if not state.get('datasets_combined'):
            if step_combine_datasets(intervals, storage):
                state['datasets_combined'] = True
                _save_state(state)
        
        gc.collect()
        
        # Step 3: Train general models
        if not skip_general_training:
            if not state.get('general_models_trained'):
                if step_train_general_models(backend, storage):
                    state['general_models_trained'] = True
                    _save_state(state)
        else:
            logger.info("Skipping general model training")
        
        gc.collect()
        
        # Step 4: Train interval models
        if not state.get('interval_models_trained'):
            fine_tune = not skip_general_training and backend == 'xgboost'
            if step_train_interval_models(intervals, backend, fine_tune, storage):
                state['interval_models_trained'] = True
                _save_state(state)
        
        gc.collect()
        
        # Success
        end_time = datetime.now(IST)
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 BOSON SYSTEM UPDATE COMPLETED")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 60)
        
        # Clear state on success
        _clear_state()
        
        # Send notification
        try:
            print(
                "Boson Update Complete",
                f"Pipeline completed in {duration}\nIntervals: {intervals}\nBackend: {backend}"
            )
        except Exception as e:
            print(e)
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ BOSON SYSTEM UPDATE FAILED: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        
        # Save state for resume
        _save_state(state)
        
        # Send error notification
        try:
            print("Boson Update Failed", str(e))
        except Exception as e:
            print(e)
        
        return False


# =============================================================================
# Extended Pipeline (4-Tier)
# =============================================================================

def update_boson_system_extended(
    intervals: List[str] = None,
    features: List[str] = None,
    backend: str = None
) -> bool:
    """
    Run the extended 4-tier pipeline.
    
    Tiers:
    1. General Models (all data)
    2. Interval Models (fine-tuned from General)
    3. Market Models (indices/commodities/equity)
    4. Symbol Models (per-symbol)
    """
    logger.info("Starting Extended 4-Tier Pipeline")
    
    # Load configuration
    config = _load_config()
    intervals = intervals or config.get('intervals', DEFAULT_INTERVALS)
    backend = backend or config.get('model_backend', 'xgboost')
    
    # Initialize storage
    storage = get_storage()
    
    # Run standard 2-tier first (General + Interval)
    success = update_boson_system(intervals, features, backend)
    
    if not success:
        return False
    
    # Tier 3: Market Models
    logger.info("=" * 60)
    logger.info("STEP 5: Training Market Models (Tier 3)")
    logger.info("=" * 60)
    
    try:
        train_market_models(intervals, storage, backend)
        logger.info("Market model training complete")
    except Exception as e:
        logger.error(f"Market model training failed: {e}")
        logger.error(traceback.format_exc())
    
    gc.collect()
    
    # Tier 4: Symbol Models
    logger.info("=" * 60)
    logger.info("STEP 6: Training Symbol Models (Tier 4)")
    logger.info("=" * 60)
    
    try:
        train_symbol_models(intervals, storage, backend)
        logger.info("Symbol model training complete")
    except Exception as e:
        logger.error(f"Symbol model training failed: {e}")
        logger.error(traceback.format_exc())
    
    gc.collect()
    
    logger.info("=" * 60)
    logger.info("🎉 EXTENDED 4-TIER PIPELINE COMPLETED")
    logger.info("=" * 60)
    
    return True
