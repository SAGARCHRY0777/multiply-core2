"""Conditions module - Consolidated condition checks.

All condition functions are now in 4 consolidated files:
- divergence.py: detect_divergence, get_divergence
- ema_crossings.py: check_ema_5_crossing_13, check_ema_5_crossing_50
- indicators.py: MACD, ADX, DI, RSI, price/EMA checks
- volume.py: candle_with_good_volume
"""

# Divergence detection
from .divergence import detect_divergence, get_divergence

# EMA crossings
from .ema_crossings import check_ema_5_crossing_13, check_ema_5_crossing_50

# Technical indicators
from .indicators import (
    check_most_recent_macd_crossing,
    directional_indicator_check,
    check_adx_greater_than_14,
    check_adx_greater_than_25,
    check_rsi,
    rsi_trend_check,
    price_50ema,
    check_limit_price_break
)

# Volume checks
from .volume import candle_with_good_volume

# Higher interval divergence (not consolidated - separate import)
from .higher_interval_divergence_check import higher_interval_divergence_check

__all__ = [
    'detect_divergence',
    'get_divergence',
    'check_ema_5_crossing_13',
    'check_ema_5_crossing_50',
    'check_most_recent_macd_crossing',
    'directional_indicator_check',
    'check_adx_greater_than_14',
    'check_adx_greater_than_25',
    'check_rsi',
    'rsi_trend_check',
    'price_50ema',
    'check_limit_price_break',
    'candle_with_good_volume',
    'higher_interval_divergence_check'
]
