from collections import OrderedDict
from stock_indicators import indicators
import concurrent.futures
import gc
import logging

def calculate_indicators_data_live(candle_data, from_date, to_date, required_indicators=None):
    gc.enable()

    # --- Start Diagnostic Logging ---
    logging.info(f"calculate_indicators_data_live received {len(candle_data)} candles.")
    logging.info(f"Date range for filtering: from {from_date} to {to_date}")
    # --- End Diagnostic Logging ---

    def safe_get_date(cd):
        """Safely extract date from Quote object, handling PropertyObject issues."""
        try:
            # Try standard .date access
            d = cd.date
            if hasattr(d, 'strftime'):
                return d
            # If it's a PropertyObject, try to get the underlying value
            if hasattr(d, 'ToString'):
                from datetime import datetime
                return datetime.fromisoformat(d.ToString("s"))
            # Try accessing via _date attribute
            if hasattr(cd, '_date'):
                return cd._date
            # Last resort: return None and filter out
            return None
        except Exception as e:
            logging.warning(f"Failed to extract date from quote: {e}")
            return None

    # Pre-format dates and create a date mask for filtering
    dates = []
    valid_candles = []
    for cd in candle_data:
        d = safe_get_date(cd)
        if d is not None:
            dates.append(d.strftime("%Y-%m-%d %H:%M:%S%z") if d.tzinfo else d.strftime("%Y-%m-%d %H:%M:%S"))
            valid_candles.append(cd)
    
    if not valid_candles:
        logging.warning("No valid candles with extractable dates")
        return {}
    
    from_date_naive = from_date.replace(tzinfo=None)
    to_date_naive = to_date.replace(tzinfo=None)
    date_mask = []
    for cd in valid_candles:
        d = safe_get_date(cd)
        if d:
            d_naive = d.replace(tzinfo=None) if hasattr(d, 'replace') else d
            date_mask.append(from_date_naive <= d_naive <= to_date_naive)
        else:
            date_mask.append(False)

    # --- Start Diagnostic Logging ---
    logging.info(f"Date mask contains {sum(date_mask)} matching date(s).")
    # --- End Diagnostic Logging ---


    def safe_float(val):
        try:
            return float(val) if val is not None else float('nan')
        except Exception as er:
            print(er)
            return float('nan')

    def compute_adx_dmi(candle_data, dates, date_mask):
        try:
            adx_list = indicators.get_adx(candle_data, lookback_periods=14)
            return {
                'adx': [[d, safe_float(adx.adx)] for d, adx, m in zip(dates, adx_list, date_mask) if m],
                'pdi': [[d, safe_float(adx.pdi)] for d, adx, m in zip(dates, adx_list, date_mask) if m],
                'mdi': [[d, safe_float(adx.mdi)] for d, adx, m in zip(dates, adx_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing ADX/DMI: {e}")
            return {}

    def compute_macd(candle_data, dates, date_mask):
        try:
            macd_list = indicators.get_macd(candle_data)
            return {
                'macd': [[d, safe_float(m.macd)] for d, m, msk in zip(dates, macd_list, date_mask) if msk],
                'macd_signal': [[d, safe_float(m.signal)] for d, m, msk in zip(dates, macd_list, date_mask) if msk],
            }
        except Exception as e:
            logging.error(f"Error computing MACD: {e}")
            return {}

    def compute_rsi(candle_data, dates, date_mask):
        try:
            rsi_vals = indicators.get_rsi(candle_data)
            return {'rsi': [[d, safe_float(r.rsi)] for d, r, m in zip(dates, rsi_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing RSI: {e}")
            return {}

    def compute_ema5(candle_data, dates, date_mask):
        try:
            ema5_vals = indicators.get_ema(candle_data, lookback_periods=5)
            return {'ema5': [[d, safe_float(e.ema)] for d, e, m in zip(dates, ema5_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing EMA5: {e}")
            return {}

    def compute_ema13(candle_data, dates, date_mask):
        try:
            ema13_vals = indicators.get_ema(candle_data, lookback_periods=13)
            return {'ema13': [[d, safe_float(e.ema)] for d, e, m in zip(dates, ema13_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing EMA13: {e}")
            return {}

    def compute_ema50(candle_data, dates, date_mask):
        try:
            ema50_vals = indicators.get_ema(candle_data, lookback_periods=50)
            return {'ema50': [[d, safe_float(e.ema)] for d, e, m in zip(dates, ema50_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing EMA50: {e}")
            return {}

    def compute_parabolic_sar(candle_data, dates, date_mask):
        try:
            psar_vals = indicators.get_parabolic_sar(candle_data)
            return {'parabolicsar': [[d, safe_float(ps.sar)] for d, ps, m in zip(dates, psar_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Parabolic SAR: {e}")
            return {}

    tasks = {
        'adx_dmi': compute_adx_dmi,
        'macd': compute_macd,
        'rsi': compute_rsi,
        'ema5': compute_ema5,
        'ema13': compute_ema13,
        'ema50': compute_ema50,
        'parabolicsar': compute_parabolic_sar
    }

    if required_indicators:
        tasks = {key: func for key, func in tasks.items() if key in required_indicators}

    results = {}
    # Use ThreadPoolExecutor (threads share memory, so pickling is not needed)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(func, valid_candles, dates, date_mask): key for key, func in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_task):
            key = future_to_task[future]
            try:
                task_result = future.result()
                results.update(task_result)
            except Exception as e:
                logging.error(f"Error in task {key}: {e}")

    gc.collect()
    desired_order = [
        'aroon_up',
        'aroon_down',
        'adx',
        'pdi',
        'mdi',
        'elderray_bull_power',
        'elderray_bear_power',
        'gator_upper',
        'gator_lower',
        'gator_is_upper_expanding',
        'gator_is_lower_expanding',
        'hurst',
        'ichimoku_tenkan_sen',
        'ichimoku_kijun_sen',
        'ichimoku_senkou_span_a',
        'ichimoku_senkou_span_b',
        'ichimoku_chikou_span',
        'macd',
        'macd_signal',
        'macd_histogram',
        'supertrend',
        'vortex_pvi',
        'vortex_nvi',
        'alligator_jaw',
        'alligator_teeth',
        'alligator_lips',
        'bollingerbands_upper',
        'bollingerbands_lower',
        'donchianchannels_upper',
        'donchianchannels_lower',
        'fcb_upper',
        'fcb_lower',
        'keltnerchannels_upper',
        'keltnerchannels_lower',
        'keltnerchannels_center',
        'maenvelopes_upper',
        'maenvelopes_lower',
        'pivotpoints_pp',
        'pivotpoints_r1',
        'pivotpoints_s1',
        'rollingpivotpoints_pp',
        'rollingpivotpoints_r1',
        'rollingpivotpoints_s1',
        'starc_upper',
        'starc_lower',
        'standarddeviationchannels_upper',
        'standarddeviationchannels_lower',
        'awesomeoscillator',
        'cci',
        'connorsrsi',
        'dpo',
        'rsi',
        'stc',
        'smi',
        'stochasticoscillator_k',
        'stochasticoscillator_d',
        'stochasticrsi_rsi',
        'stochasticrsi_signal',
        'trix',
        'ultimateoscillator',
        'williamsr',
        'chandelierexit',
        'parabolicsar',
        'volatilitystop',
        'williamsfractal_bull',
        'williamsfractal_bear',
        'adl',
        'cmf',
        'chaikinoscillator',
        'forceindex',
        'kvo',
        'mfi',
        'obv',
        'pvo',
        'alma',
        'dema',
        'epma',
        'ema5',
        'ema9',
        'ema13',
        'ema50',
        'hilberttransform',
        'hma',
        'kama',
        'mama',
        'fama',
        'sma',
        'smma',
        't3',
        'tema',
        'vwap',
        'vwma',
        'wma',
        'fishertransform_fisher',
        'fishertransform_trigger',
        'zigzag',
        'atr',
        'bop',
        'choppinessindex',
        'pmo',
        'pmo_signal',
        'roc',
        'truerange',
        'tsi',
        'ulcerindex',
        'slope',
        'standarddeviation'
    ]

    ordered_results = OrderedDict()
    for key in desired_order:
        # Use an empty list if a key is missing to mimic sequential behavior
        ordered_results[key] = results.get(key, [])

    return ordered_results

def process_indicators_data_live(args):
    quotes, from_date, to_date, interval = args
    output = calculate_indicators_data_live(quotes, from_date, to_date)
    return output
