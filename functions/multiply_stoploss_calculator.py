from __future__ import annotations
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
import json
import pandas as pd
from sqlalchemy import text, create_engine
from functions.data_utils import get_db_url


db_url = get_db_url()

engine = create_engine(db_url, connect_args={'sslmode': 'require'})

logger = logging.getLogger(__name__)


def _coerce_timestamp(entry_timestamp: Any) -> Optional[datetime]:
    """Best-effort conversion of inbound timestamp to ``datetime``."""
    if entry_timestamp is None:
        return None

    if isinstance(entry_timestamp, datetime):
        return entry_timestamp

    if isinstance(entry_timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(entry_timestamp)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(entry_timestamp, str) and entry_timestamp.strip():
        text = entry_timestamp.strip()
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    return None

def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    coerced = _to_float(value)
    return coerced if coerced is not None else 0.0


def _combine_price_indicator_rows(symbol: str, interval: str) -> List[Dict[str, Any]]:
    print("Fetching candle and indicator data with SQL JOIN")

    sql = """
    SELECT c.datetime,
           c.open,
           c.high,
           c.low,
           c.close,
           c.volume,
           i.parabolicsar
    FROM candle_data c
    JOIN all_indicators_data i
      ON c.symbol = i.symbol
     AND c.interval = i.interval
     AND c.datetime = i.datetime
    WHERE c.symbol = :symbol
      AND c.interval = :interval
    ORDER BY c.datetime ASC
    """

    with engine.connect() as connection:
        df = pd.read_sql(text(sql), connection, params={"symbol": symbol, "interval": interval})

    if df.empty:
        return []

    # Convert datetime once for all rows
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Ensure numeric values are floats
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'parabolicsar', 'buying_volume', 'selling_volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Combined price/indicator rows (SQL join)")
    return df.to_dict('records')

def _determine_trend(rows: List[Dict[str, Any]], index: int) -> Optional[str]:
    close_price = rows[index]['close']
    sar_value = rows[index]['parabolicsar']

    if close_price is None or sar_value is None:
        print("Trend determination failed: missing price or SAR")
        return None

    if close_price > sar_value:
        return 'bullish'
    if close_price < sar_value:
        return 'bearish'

    # If price equals SAR, inspect the previous candle to infer trend
    if index > 0:
        prev_close = rows[index - 1]['close']
        prev_sar = rows[index - 1]['parabolicsar']
        if prev_close is None or prev_sar is None:
            return None
        if prev_close > prev_sar:
            return 'bullish'
        if prev_close < prev_sar:
            return 'bearish'

    print("Trend determination ambiguous even after fallback")
    return None


def _find_trend_start(rows: List[Dict[str, Any]], entry_index: int, trend: str) -> int:
    for idx in range(entry_index - 1, -1, -1):
        close_price = rows[idx]['close']
        sar_value = rows[idx]['parabolicsar']

        if close_price is None or sar_value is None:
            continue

        if trend == 'bullish' and close_price <= sar_value:
            return idx + 1
        if trend == 'bearish' and close_price >= sar_value:
            return idx + 1

    print("Trend start inferred")
    return 0


def _extreme_point(rows: List[Dict[str, Any]], start: int, end: int, trend: str) -> Optional[float]:
    window = rows[start:end + 1]
    if trend == 'bullish':
        highs = [row['high'] for row in window if row['high'] is not None]
        value = max(highs) if highs else None
    else:
        lows = [row['low'] for row in window if row['low'] is not None]
        value = min(lows) if lows else None

    print("Computed extreme point")
    return value


def _clamp_level(rows: List[Dict[str, Any]], entry_index: int, trend: str) -> Optional[float]:
    candidates: List[float] = []
    value_key = 'low' if trend == 'bullish' else 'high'

    for offset in (1, 2):
        idx = entry_index - offset
        if idx < 0:
            continue
        candidate = rows[idx].get(value_key)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        print("Clamp level unavailable due to insufficient history")
        return None

    clamp = min(candidates)
    print("Derived clamp level")
    return clamp


def _confidence_score(reasons: List[str]) -> float:
    penalty = sum(0.15 for reason in reasons if reason.startswith('[low-confidence]'))
    return max(0.2, round(1.0 - penalty, 2))


def _normalise_direction(direction: str) -> Optional[str]:
    if not direction:
        return None
    direction = direction.strip().lower()
    if direction in {'bullish', 'long', 'buy', 'BULL', 'bull'}:
        return 'bullish'
    if direction in {'bearish', 'short', 'sell', 'BEAR', 'bear'}:
        return 'bearish'
    return None


_HIGHER_INTERVAL_MAP = {
    'ONE_MINUTE': 'FIVE_MINUTE',
    'THREE_MINUTE': 'FIFTEEN_MINUTE',
    'FIVE_MINUTE': 'FIFTEEN_MINUTE',
    'FIFTEEN_MINUTE': 'ONE_HOUR',
    'ONE_HOUR': 'ONE_DAY',
}


def _get_higher_interval(interval: str) -> Optional[str]:
    if not interval:
        return None
    return _HIGHER_INTERVAL_MAP.get(interval.upper())


def _locate_entry_index(rows: List[Dict[str, Any]], timestamp: datetime) -> int:
    for idx, row in enumerate(rows):
        if row['datetime'] >= timestamp:
            return idx
    return max(len(rows) - 1, 0)


def _compute_trend_volume_metrics(
    rows: List[Dict[str, Any]],
    entry_index: int,
) -> Tuple[Dict[str, Any], List[str]]:
    if not rows:
        return {}, []

    window_start = max(0, entry_index - 2)
    window = rows[window_start: entry_index + 1]

    reasons: List[str] = []

    current_row = rows[entry_index]
    current_close = _to_float(current_row.get('close'))
    previous_close = _to_float(rows[entry_index - 1]['close']) if entry_index - 1 >= 0 else None

    if current_close is None:
        current_close = 0.0
        reasons.append('[low-confidence] Current close missing; defaulting to 0 for trend strength.')

    trend_strength_percentage = 0.0
    previous_close_valid = previous_close not in (None, 0)
    if previous_close_valid:
        trend_strength_percentage = ((current_close - previous_close) / previous_close) * 100
    else:
        reasons.append('[low-confidence] Insufficient price history for trend strength calculation.')

    if trend_strength_percentage > 1.5:
        trend_band = 'Strong Trend'
    elif 0.5 <= trend_strength_percentage <= 1.5:
        trend_band = 'Moderate Trend'
    elif trend_strength_percentage < 0.5:
        trend_band = 'Weak Trend'
    else:
        trend_band = 'No Trend'

    # Volume aggregation over the three-candle window
    current_volume = _safe_float(current_row.get('volume'))
    previous_volume = _safe_float(rows[entry_index - 1].get('volume')) if entry_index - 1 >= 0 else 0.0
    previous_previous_volume = (
        _safe_float(rows[entry_index - 2].get('volume')) if entry_index - 2 >= 0 else 0.0
    )

    total_volume = current_volume + previous_volume + previous_previous_volume

    # Approximate buying/selling volume using last three candles only
    buying_volume = 0.0
    selling_volume = 0.0
    missing_ohlc_reason_added = False
    for row in window:
        row_volume = _safe_float(row.get('volume'))
        row_open = _to_float(row.get('open'))
        row_close = _to_float(row.get('close'))

        if row_open is None or row_close is None:
            buy_share = 0.5
            if not missing_ohlc_reason_added:
                reasons.append('[low-confidence] Missing OHLC data for volume split; assuming balanced flow.')
                missing_ohlc_reason_added = True
            buying_volume += row_volume * buy_share
            selling_volume += row_volume * (1.0 - buy_share)
            continue

        if row_close > row_open:
            buying_volume += row_volume
            continue

        if row_close < row_open:
            selling_volume += row_volume
            continue

        # Doji candle (close == open): assume balanced buying/selling.
        buying_volume += row_volume * 0.5
        selling_volume += row_volume * 0.5

    if total_volume == 0:
        absorption_ratio = 0.0
        continuation_percentage = 0.0
    else:
        absorption_ratio = (buying_volume - selling_volume) / total_volume
        continuation_percentage = absorption_ratio * 100

    average_volume_change = absorption_ratio
    average_price_change = trend_strength_percentage if previous_close_valid else 0.0

    signal_output = 'Nothing'
    if average_volume_change > 0.3 > average_price_change > -0.3:
        signal_output = 'absorption or trapped sellers'
    elif average_volume_change < -0.3 and average_price_change < -0.3:
        signal_output = 'Continuation of Weakness'
    elif average_price_change > 0.3 and average_volume_change < -0.3:
        signal_output = 'False Breakout'

    metrics = {
        'trend_strength_percentage': trend_strength_percentage,
        'trend_band': trend_band,
        'total_volume_window': total_volume,
        'buying_volume_window': buying_volume,
        'selling_volume_window': selling_volume,
        'absorption_ratio': absorption_ratio,
        'average_volume_change': average_volume_change,
        'average_price_change': average_price_change,
        'volume_signal': signal_output,
        'continuation_percentage': continuation_percentage,
    }

    return metrics, reasons


def _prepare_stoploss_context(rows: List[Dict[str, Any]], entry_index: int) -> Dict[str, Any]:
    sar_current = rows[entry_index]['parabolicsar']
    if sar_current is None:
        raise ValueError("Parabolic SAR value unavailable at the entry point")

    trend = _determine_trend(rows, entry_index)
    if trend is None:
        raise ValueError("Unable to determine trend direction from Parabolic SAR")

    reasons: List[str] = []
    trend_start = _find_trend_start(rows, entry_index, trend)
    point_count = entry_index - trend_start + 1
    if point_count <= 0:
        point_count = 1
        reasons.append('[low-confidence] Trend point count adjusted to minimum due to data gap')

    acceleration_factor = min(0.02 * point_count, 0.2)
    extreme_point = _extreme_point(rows, trend_start, entry_index, trend)
    if extreme_point is None:
        reasons.append('[low-confidence] Unable to identify extreme point; using entry close price')
        extreme_point = rows[entry_index]['close']

    sar_next = sar_current + acceleration_factor * (extreme_point - sar_current)

    clamp_level = _clamp_level(rows, entry_index, trend)
    if clamp_level is None:
        reasons.append('[low-confidence] Insufficient swing references for clamp')

    stoploss_candidates = [value for value in (clamp_level, sar_next) if value is not None]
    if not stoploss_candidates:
        raise ValueError("Unable to calculate stoploss due to missing data")

    stoploss_value = min(stoploss_candidates)

    reasons.append(
        f"Trend detected as {trend} with {point_count} SAR steps (AF={acceleration_factor:.2f}, EP={extreme_point:.2f})."
    )
    reasons.append(
        f"SAR projection at entry: current={sar_current:.2f}, next={sar_next:.2f}."
    )
    if clamp_level is not None:
        reasons.append(f"Clamp reference derived from recent swings: {clamp_level:.2f}.")

    entry_price = rows[entry_index]['close']

    result = {
        "trend": trend,
        "trend_start": trend_start,
        "point_count": point_count,
        "acceleration_factor": acceleration_factor,
        "extreme_point": extreme_point,
        "sar_current": sar_current,
        "sar_next": sar_next,
        "clamp_level": clamp_level,
        "stoploss_value": stoploss_value,
        "reasons": reasons,
        "entry_price": entry_price,
    }
    return result


def _clean_json_response(raw_text: str) -> Dict[str, Any]:
    cleaned = (raw_text or '').strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Warning: Failed to decode LLM response as JSON")
        return {}


def multiply_stoploss_calculator(
    symbol: str,
    interval: str,
    entry_timestamp: Any,
    direction: str,
) -> Dict[str, Any]:
    if not symbol:
        raise ValueError("symbol is required")
    if not interval:
        raise ValueError("interval is required")
    if not direction:
        raise ValueError("direction is required")

    parsed_timestamp = _coerce_timestamp(entry_timestamp)
    if parsed_timestamp is None:
        raise ValueError("entry_timestamp is invalid or missing")

    rows = _combine_price_indicator_rows(symbol, interval)
    if not rows:
        raise ValueError("No candle or indicator data available for the requested symbol/interval")

    entry_index = _locate_entry_index(rows, parsed_timestamp)
    active_rows = rows
    active_entry_index = entry_index

    context = _prepare_stoploss_context(active_rows, active_entry_index)
    trend_interval = interval
    higher_interval_attempted: Optional[str] = None
    logger.info(f"direction: {direction}")
    direction_normalised = _normalise_direction(direction)
    direction_matches = direction_normalised == context['trend'] if direction_normalised else True
    alignment_notes: List[str] = []
    entry_price = context['entry_price']

    if direction_normalised and not direction_matches:
        alignment_notes.append(
            f"{interval} trend is {context['trend']} while user requested {direction_normalised}."
        )
        higher_interval = _get_higher_interval(interval)
        higher_interval_attempted = higher_interval
        if higher_interval:
            higher_rows = _combine_price_indicator_rows(symbol, higher_interval)
            if higher_rows:
                higher_entry_index = _locate_entry_index(higher_rows, parsed_timestamp)
                try:
                    higher_context = _prepare_stoploss_context(higher_rows, higher_entry_index)
                except ValueError as err:
                    alignment_notes.append(f"Unable to evaluate {higher_interval} trend: {err}")
                else:
                    if higher_context['trend'] == direction_normalised:
                        trend_interval = higher_interval
                        context = higher_context
                        active_rows = higher_rows
                        active_entry_index = higher_entry_index
                        direction_matches = True
                        entry_price = context['entry_price']
                        alignment_notes.append(
                            f"Using {higher_interval} trend ({direction_normalised}) which aligns with requested direction."
                        )
                    else:
                        alignment_notes.append(
                            f"{higher_interval} trend remains {higher_context['trend']}; still conflicts with request."
                        )
            else:
                alignment_notes.append(f"No candle data available for {symbol} on {higher_interval} interval.")
        else:
            alignment_notes.append(f"No higher interval configured beyond {interval}.")

        if not direction_matches:
            unresolved_interval = higher_interval or 'higher timeframe'
            alignment_notes.append(
                f"Neither {interval} nor {unresolved_interval} trends align with the requested {direction_normalised} direction; stoploss withheld."
            )

    reasons = context['reasons']
    volume_metrics, metric_reasons = _compute_trend_volume_metrics(active_rows, active_entry_index)
    if metric_reasons:
        reasons.extend(metric_reasons)

    if volume_metrics:
        reasons.append(
            "Trend strength {0:.2f}% classified as {1}.".format(
                volume_metrics['trend_strength_percentage'],
                volume_metrics['trend_band'],
            )
        )
        reasons.append(
            "Volume signal {0} with absorption ratio {1:.2f}.".format(
                volume_metrics['volume_signal'],
                volume_metrics['absorption_ratio'],
            )
        )

    trend = context['trend']
    trend_start = context['trend_start']
    point_count = context['point_count']
    acceleration_factor = context['acceleration_factor']
    extreme_point = context['extreme_point']
    sar_current = context['sar_current']
    sar_next = context['sar_next']
    clamp_level = context['clamp_level']
    stoploss_value = context['stoploss_value']
    if alignment_notes:
        reasons.extend(alignment_notes)

    llm_payload = {
        "trend": trend,
        "trend_interval": trend_interval,
        "interval_requested": interval,
        "higher_interval_attempted": higher_interval_attempted,
        "user_direction": direction_normalised or direction,
        "direction_matches": direction_matches,
        "stoploss_candidate": stoploss_value,
        "entry_price": entry_price,
        "acceleration_factor": acceleration_factor,
        "point_count": point_count,
        "extreme_point": extreme_point,
        "clamp_level": clamp_level,
        "sar_current": sar_current,
        "sar_next": sar_next,
        "collected_reasons": reasons,
    }

    llm_payload.update(volume_metrics)

    print(llm_payload)
    llm_instruction = (
        "You are a trading risk assistant that explains stoploss recommendations clearly and simply."
        " Carefully read the JSON context provided. It contains candle data insights, SAR calculations,"
        " and user intent (direction). Follow these exact rules when deciding and writing your JSON response:"

        " 1️⃣ If the detected trend matches the user's direction on the requested interval:"
        "     - set 'stoploss_allowed' to true."
        "     - use the computed stoploss_candidate as the stoploss value."
        "     - write short, clear reasons (2–4 sentences total) describing:"
        "         • which interval was used,"
        "         • the trend (bullish or bearish),"
        "         • how the SAR and clamp support this trend,"
        "         • and why this stoploss is suitable."

        " 2️⃣ If the trend does NOT match on the requested interval but matches on the higher interval:"
        "     - set 'stoploss_allowed' to true."
        "     - use the higher interval's stoploss_candidate as the final stoploss value."
        "     - clearly explain that the lower timeframe showed an opposite trend,"
        "       but the higher timeframe confirmed the user's direction."
        "     - speak in simple, natural terms such as:"
        "         'On the 15m chart trend was opposite, but on the 1h chart it agrees with your direction, so this stoploss is based on the higher interval.'"

        " 3️⃣ If the trend does NOT match on both intervals:"
        "     - set 'stoploss_allowed' to false."
        "     - do not suggest a stoploss value."
        "     - write an easy-to-understand reason like:"
        "         'Both the current and higher timeframes are moving against your chosen direction, so placing a stoploss now would be risky and unreliable.'"

        " 4️⃣ In all cases:"
        "     - Assign a 'confidence_score' between 30 and 70 inclusive based on alignment:"
        "         • 50–70 if the requested interval already matches the user's direction."
        "         • 40–50 if only the higher interval aligns while the requested interval does not."
        "         • 30–40 if neither interval aligns with the user's direction."
        "         • Stay within 30–70 for any other situation."
        "     - Include a field 'summary' that summarizes your reasoning in one sentence, using friendly plain English."
        "     - Include a 'reasons' array listing the main factors that led to your decision."
        "     - Avoid technical jargon; speak like explaining to a smart beginner."

        " Respond ONLY in valid JSON with the fields:"
        "   confidence_score (number),"
        "   reasons (array of short strings),"
        "   stoploss_allowed (boolean),"
        "   summary (string)."
    )

    llm_data = {}
    confidence = ""
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None
    if not isinstance(confidence, (int, float)):
        confidence = _confidence_score(reasons)

    llm_reasons = llm_data.get('reasons')
    if isinstance(llm_reasons, str):
        llm_reasons = [llm_reasons]
    if not isinstance(llm_reasons, list) or not llm_reasons:
        llm_reasons = reasons

    stoploss_allowed = llm_data.get('stoploss_allowed') if llm_data else direction_matches
    if direction_normalised and not direction_matches:
        stoploss_allowed = False

    direction_supplied = bool(direction_normalised)
    interval_alignment = direction_supplied and direction_matches and trend_interval == interval
    higher_interval_alignment = direction_supplied and direction_matches and trend_interval != interval
    both_intervals_failed = direction_supplied and not direction_matches

    def _map_confidence_to_band(value: float, lower: float, upper: float) -> float:
        span = upper - lower
        if span <= 0:
            return lower
        if value <= 1:
            # Treat as 0-1 score and scale into band
            normalised = max(0.0, min(1.0, value))
            return lower + normalised * span
        return max(lower, min(upper, value))

    if interval_alignment:
        confidence = _map_confidence_to_band(confidence, 50.0, 70.0)
    elif higher_interval_alignment:
        confidence = _map_confidence_to_band(confidence, 40.0, 50.0)
    elif both_intervals_failed:
        confidence = _map_confidence_to_band(confidence, 30.0, 40.0)
    else:
        confidence = _map_confidence_to_band(confidence, 30.0, 70.0)

    stoploss_output = stoploss_value if stoploss_allowed else None
    if stoploss_output is not None and direction_normalised in {'bullish', 'bearish'}:
        adjustment = stoploss_output * 0.0004  # 0.04% directional buffer
        if direction_normalised == 'bullish':
            stoploss_output -= adjustment
        else:
            stoploss_output += adjustment
    summary_note = llm_data.get('summary') if isinstance(llm_data.get('summary'), str) else None
    if summary_note:
        llm_reasons.insert(0, summary_note)

    if alignment_notes:
        for note in alignment_notes:
            if note not in llm_reasons:
                llm_reasons.append(note)

    trend_source = 'higher_interval' if trend_interval != interval else 'requested_interval'

    result = {
        "stoploss": None if stoploss_output is None else round(stoploss_output, 4),
        "confidence_score": round(confidence, 2),
        "reasons": llm_reasons,
        "inputs": {
            "symbol": symbol,
            "interval": interval,
            "entry_timestamp": parsed_timestamp.isoformat(),
            "direction": direction,
        },
        "metadata": {
            "trend": trend,
            "trend_interval": trend_interval,
            "trend_source": trend_source,
            "higher_interval_checked": higher_interval_attempted,
            "trend_start": active_rows[trend_start]['datetime'].isoformat() if active_rows else None,
            "sar_current": sar_current,
            "sar_next": sar_next,
            "extreme_point": extreme_point,
            "clamp_level": clamp_level,
            "point_count": point_count,
            "af": acceleration_factor,
            "stoploss_allowed": stoploss_allowed,
        },
    }

    if volume_metrics:
        result['metadata'].update(volume_metrics)

    return result
