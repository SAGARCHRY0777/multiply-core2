from collections import OrderedDict
from stock_indicators import indicators, PivotPointType, PeriodSize
import concurrent.futures
import gc
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_indicators_data(candle_data, from_date, to_date, required_indicators=None):
    """
    Parallelized indicator calculation.

    Parameters:
      candle_data: A list of candle objects (each having a .date attribute).
      from_date, to_date: datetime objects defining the date range.
      required_indicators: A list of keys (strings) corresponding to indicator groups to process.
                           If None or empty, all indicators are computed.

    Returns:
      A dictionary with keys such as 'aroon_up', 'rsi', etc. and values as lists of [date, value].
    """
    gc.enable()
    # Pre-format dates and create a date mask for filtering
    dates = [cd.date.strftime("%Y-%m-%d %H:%M:%S%z") for cd in candle_data]
    from_date_naive = from_date.replace(tzinfo=None)
    to_date_naive = to_date.replace(tzinfo=None)
    date_mask = [from_date_naive <= cd.date.replace(tzinfo=None) <= to_date_naive for cd in candle_data]

    def safe_float(val):
        try:
            return float(val) if val is not None else float('nan')
        except Exception:
            return float('nan')

    # === Define indicator group functions =====================================

    def compute_aroon(candle_data, dates, date_mask):
        try:
            aroon = indicators.get_aroon(candle_data, lookback_periods=14)
            return {
                'aroon_up': [[d, safe_float(ar.aroon_up)] for d, ar, m in zip(dates, aroon, date_mask) if m],
                'aroon_down': [[d, safe_float(ar.aroon_down)] for d, ar, m in zip(dates, aroon, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing aroon: {e}")
            return {}

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

    def compute_elder_ray(candle_data, dates, date_mask):
        try:
            er_list = indicators.get_elder_ray(candle_data, lookback_periods=13)
            return {
                'elderray_bull_power': [[d, safe_float(er.bull_power)] for d, er, m in zip(dates, er_list, date_mask) if
                                        m],
                'elderray_bear_power': [[d, safe_float(er.bear_power)] for d, er, m in zip(dates, er_list, date_mask) if
                                        m],
            }
        except Exception as e:
            logging.error(f"Error computing Elder Ray: {e}")
            return {}

    def compute_gator(candle_data, dates, date_mask):
        try:
            g_list = indicators.get_gator(candle_data)
            return {
                'gator_upper': [[d, safe_float(g.upper)] for d, g, m in zip(dates, g_list, date_mask) if m],
                'gator_lower': [[d, safe_float(g.lower)] for d, g, m in zip(dates, g_list, date_mask) if m],
                'gator_is_upper_expanding': [[d, 1 if g.is_upper_expanding else 0] for d, g, m in
                                             zip(dates, g_list, date_mask) if m],
                'gator_is_lower_expanding': [[d, 1 if g.is_lower_expanding else 0] for d, g, m in
                                             zip(dates, g_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Gator: {e}")
            return {}

    def compute_hurst(candle_data, dates, date_mask):
        try:
            hurst_list = indicators.get_hurst(candle_data)
            return {'hurst': [[d, safe_float(h.hurst_exponent)] for d, h, m in zip(dates, hurst_list, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Hurst: {e}")
            return {}

    def compute_ichimoku(candle_data, dates, date_mask):
        try:
            ichi_list = indicators.get_ichimoku(candle_data)
            return {
                'ichimoku_tenkan_sen': [[d, safe_float(ic.tenkan_sen)] for d, ic, m in zip(dates, ichi_list, date_mask)
                                        if m],
                'ichimoku_kijun_sen': [[d, safe_float(ic.kijun_sen)] for d, ic, m in zip(dates, ichi_list, date_mask) if
                                       m],
                'ichimoku_senkou_span_a': [[d, safe_float(ic.senkou_span_a)] for d, ic, m in
                                           zip(dates, ichi_list, date_mask) if m],
                'ichimoku_senkou_span_b': [[d, safe_float(ic.senkou_span_b)] for d, ic, m in
                                           zip(dates, ichi_list, date_mask) if m],
                'ichimoku_chikou_span': [[d, safe_float(ic.chikou_span)] for d, ic, m in
                                         zip(dates, ichi_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Ichimoku: {e}")
            return {}

    def compute_macd(candle_data, dates, date_mask):
        try:
            macd_list = indicators.get_macd(candle_data)
            return {
                'macd': [[d, safe_float(m.macd)] for d, m, msk in zip(dates, macd_list, date_mask) if msk],
                'macd_signal': [[d, safe_float(m.signal)] for d, m, msk in zip(dates, macd_list, date_mask) if msk],
                'macd_histogram': [[d, safe_float(m.histogram)] for d, m, msk in zip(dates, macd_list, date_mask) if
                                   msk],
            }
        except Exception as e:
            logging.error(f"Error computing MACD: {e}")
            return {}

    def compute_supertrend(candle_data, dates, date_mask):
        try:
            st_list = indicators.get_super_trend(candle_data)
            return {'supertrend': [[d, safe_float(st.super_trend)] for d, st, m in zip(dates, st_list, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing SuperTrend: {e}")
            return {}

    def compute_vortex(candle_data, dates, date_mask):
        try:
            vortex_list = indicators.get_vortex(candle_data, lookback_periods=14)
            return {
                'vortex_pvi': [[d, safe_float(v.pvi)] for d, v, m in zip(dates, vortex_list, date_mask) if m],
                'vortex_nvi': [[d, safe_float(v.nvi)] for d, v, m in zip(dates, vortex_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Vortex: {e}")
            return {}

    def compute_alligator(candle_data, dates, date_mask):
        try:
            alligator_list = indicators.get_alligator(candle_data)
            return {
                'alligator_jaw': [[d, safe_float(a.jaw)] for d, a, m in zip(dates, alligator_list, date_mask) if m],
                'alligator_teeth': [[d, safe_float(a.teeth)] for d, a, m in zip(dates, alligator_list, date_mask) if m],
                'alligator_lips': [[d, safe_float(a.lips)] for d, a, m in zip(dates, alligator_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Alligator: {e}")
            return {}

    def compute_bollinger(candle_data, dates, date_mask):
        try:
            bb_list = indicators.get_bollinger_bands(candle_data)
            return {
                'bollingerbands_upper': [[d, safe_float(b.upper_band)] for d, b, m in zip(dates, bb_list, date_mask) if
                                         m],
                'bollingerbands_lower': [[d, safe_float(b.lower_band)] for d, b, m in zip(dates, bb_list, date_mask) if
                                         m],
            }
        except Exception as e:
            logging.error(f"Error computing Bollinger Bands: {e}")
            return {}

    def compute_donchian(candle_data, dates, date_mask):
        try:
            donchian_list = indicators.get_donchian(candle_data)
            return {
                'donchianchannels_upper': [[d, safe_float(dch.upper_band)] for d, dch, m in
                                           zip(dates, donchian_list, date_mask) if m],
                'donchianchannels_lower': [[d, safe_float(dch.lower_band)] for d, dch, m in
                                           zip(dates, donchian_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Donchian Channels: {e}")
            return {}

    def compute_fcb(candle_data, dates, date_mask):
        try:
            fcb_list = indicators.get_fcb(candle_data)
            return {
                'fcb_upper': [[d, safe_float(f.upper_band)] for d, f, m in zip(dates, fcb_list, date_mask) if m],
                'fcb_lower': [[d, safe_float(f.lower_band)] for d, f, m in zip(dates, fcb_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing FCB: {e}")
            return {}

    def compute_keltner(candle_data, dates, date_mask):
        try:
            keltner_list = indicators.get_keltner(candle_data)
            return {
                'keltnerchannels_upper': [[d, safe_float(k.upper_band)] for d, k, m in
                                          zip(dates, keltner_list, date_mask) if m],
                'keltnerchannels_lower': [[d, safe_float(k.lower_band)] for d, k, m in
                                          zip(dates, keltner_list, date_mask) if m],
                'keltnerchannels_center': [[d, safe_float(k.center_line)] for d, k, m in
                                           zip(dates, keltner_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Keltner Channels: {e}")
            return {}

    def compute_ma_envelopes(candle_data, dates, date_mask):
        try:
            envelopes_list = indicators.get_ma_envelopes(candle_data, lookback_periods=20)
            return {
                'maenvelopes_upper': [[d, safe_float(e.upper_envelope)] for d, e, m in
                                      zip(dates, envelopes_list, date_mask) if m],
                'maenvelopes_lower': [[d, safe_float(e.lower_envelope)] for d, e, m in
                                      zip(dates, envelopes_list, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing MA Envelopes: {e}")
            return {}

    def compute_pivot_points(candle_data, dates, date_mask):
        try:
            pivots = indicators.get_pivot_points(candle_data, PeriodSize.ONE_HOUR)
            return {
                'pivotpoints_pp': [[d, safe_float(p.pp)] for d, p, m in zip(dates, pivots, date_mask) if m],
                'pivotpoints_r1': [[d, safe_float(p.r1)] for d, p, m in zip(dates, pivots, date_mask) if m],
                'pivotpoints_s1': [[d, safe_float(p.s1)] for d, p, m in zip(dates, pivots, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Pivot Points: {e}")
            return {}

    def compute_rolling_pivots(candle_data, dates, date_mask):
        try:
            rolling_pivots = indicators.get_rolling_pivots(candle_data, window_periods=14, offset_periods=3,
                                                           point_type=PivotPointType.STANDARD)
            return {
                'rollingpivotpoints_pp': [[d, safe_float(rp.pp)] for d, rp, m in zip(dates, rolling_pivots, date_mask)
                                          if m],
                'rollingpivotpoints_r1': [[d, safe_float(rp.r1)] for d, rp, m in zip(dates, rolling_pivots, date_mask)
                                          if m],
                'rollingpivotpoints_s1': [[d, safe_float(rp.s1)] for d, rp, m in zip(dates, rolling_pivots, date_mask)
                                          if m],
            }
        except Exception as e:
            logging.error(f"Error computing Rolling Pivots: {e}")
            return {}

    def compute_starc(candle_data, dates, date_mask):
        try:
            starc = indicators.get_starc_bands(candle_data)
            return {
                'starc_upper': [[d, safe_float(s.upper_band)] for d, s, m in zip(dates, starc, date_mask) if m],
                'starc_lower': [[d, safe_float(s.lower_band)] for d, s, m in zip(dates, starc, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Starc Bands: {e}")
            return {}

    def compute_std_dev_channels(candle_data, dates, date_mask):
        try:
            std_dev_channels = indicators.get_stdev_channels(candle_data)
            return {
                'standarddeviationchannels_upper': [[d, safe_float(sdc.upper_channel)] for d, sdc, m in
                                                    zip(dates, std_dev_channels, date_mask) if m],
                'standarddeviationchannels_lower': [[d, safe_float(sdc.lower_channel)] for d, sdc, m in
                                                    zip(dates, std_dev_channels, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Standard Deviation Channels: {e}")
            return {}

    def compute_awesome(candle_data, dates, date_mask):
        try:
            ao = indicators.get_awesome(candle_data)
            return {'awesomeoscillator': [[d, safe_float(a.oscillator)] for d, a, m in zip(dates, ao, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Awesome Oscillator: {e}")
            return {}

    def compute_cci(candle_data, dates, date_mask):
        try:
            cci_vals = indicators.get_cci(candle_data)
            return {'cci': [[d, safe_float(c.cci)] for d, c, m in zip(dates, cci_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing CCI: {e}")
            return {}

    def compute_connors_rsi(candle_data, dates, date_mask):
        try:
            crsi = indicators.get_connors_rsi(candle_data)
            return {'connorsrsi': [[d, safe_float(c.connors_rsi)] for d, c, m in zip(dates, crsi, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Connors RSI: {e}")
            return {}

    def compute_dpo(candle_data, dates, date_mask):
        try:
            dpo_vals = indicators.get_dpo(candle_data, lookback_periods=20)
            return {'dpo': [[d, safe_float(dpo.dpo)] for d, dpo, m in zip(dates, dpo_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing DPO: {e}")
            return {}

    def compute_rsi(candle_data, dates, date_mask):
        try:
            rsi_vals = indicators.get_rsi(candle_data)
            return {'rsi': [[d, safe_float(r.rsi)] for d, r, m in zip(dates, rsi_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing RSI: {e}")
            return {}

    def compute_stc(candle_data, dates, date_mask):
        try:
            stc_vals = indicators.get_stc(candle_data)
            return {'stc': [[d, safe_float(s.stc)] for d, s, m in zip(dates, stc_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing STC: {e}")
            return {}

    def compute_smi(candle_data, dates, date_mask):
        try:
            smi_vals = indicators.get_smi(candle_data)
            return {'smi': [[d, safe_float(s.smi)] for d, s, m in zip(dates, smi_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing SMI: {e}")
            return {}

    def compute_stoch(candle_data, dates, date_mask):
        try:
            stoch_vals = indicators.get_stoch(candle_data)
            return {
                'stochasticoscillator_k': [[d, safe_float(s.k)] for d, s, m in zip(dates, stoch_vals, date_mask) if m],
                'stochasticoscillator_d': [[d, safe_float(s.d)] for d, s, m in zip(dates, stoch_vals, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Stochastic Oscillator: {e}")
            return {}

    def compute_stoch_rsi(candle_data, dates, date_mask):
        try:
            stoch_rsi_vals = indicators.get_stoch_rsi(candle_data, rsi_periods=14, stoch_periods=14, signal_periods=3)
            return {
                'stochasticrsi_rsi': [[d, safe_float(sr.stoch_rsi)] for d, sr, m in
                                      zip(dates, stoch_rsi_vals, date_mask) if m],
                'stochasticrsi_signal': [[d, safe_float(sr.signal)] for d, sr, m in
                                         zip(dates, stoch_rsi_vals, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing Stochastic RSI: {e}")
            return {}

    def compute_trix(candle_data, dates, date_mask):
        try:
            trix_vals = indicators.get_trix(candle_data, lookback_periods=15)
            return {'trix': [[d, safe_float(t.trix)] for d, t, m in zip(dates, trix_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing TRIX: {e}")
            return {}

    def compute_ultimate(candle_data, dates, date_mask):
        try:
            ult_vals = indicators.get_ultimate(candle_data)
            return {
                'ultimateoscillator': [[d, safe_float(u.ultimate)] for d, u, m in zip(dates, ult_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Ultimate Oscillator: {e}")
            return {}

    def compute_williamsr(candle_data, dates, date_mask):
        try:
            will_vals = indicators.get_williams_r(candle_data)
            return {'williamsr': [[d, safe_float(w.williams_r)] for d, w, m in zip(dates, will_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Williams %R: {e}")
            return {}

    def compute_chandelier(candle_data, dates, date_mask):
        try:
            chand_vals = indicators.get_chandelier(candle_data)
            return {
                'chandelierexit': [[d, safe_float(ce.chandelier_exit)] for d, ce, m in zip(dates, chand_vals, date_mask)
                                   if m]}
        except Exception as e:
            logging.error(f"Error computing Chandelier Exit: {e}")
            return {}

    def compute_parabolic_sar(candle_data, dates, date_mask):
        try:
            psar_vals = indicators.get_parabolic_sar(candle_data)
            return {'parabolicsar': [[d, safe_float(ps.sar)] for d, ps, m in zip(dates, psar_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Parabolic SAR: {e}")
            return {}

    def compute_volatility_stop(candle_data, dates, date_mask):
        try:
            vs_vals = indicators.get_volatility_stop(candle_data)
            return {'volatilitystop': [[d, 1 if vs.is_stop else 0] for d, vs, m in zip(dates, vs_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Volatility Stop: {e}")
            return {}

    def compute_williams_fractal(candle_data, dates, date_mask):
        try:
            wf_vals = indicators.get_fractal(candle_data)
            return {
                'williamsfractal_bull': [[d, safe_float(wf.fractal_bull)] for d, wf, m in zip(dates, wf_vals, date_mask)
                                         if m],
                'williamsfractal_bear': [[d, safe_float(wf.fractal_bear)] for d, wf, m in zip(dates, wf_vals, date_mask)
                                         if m],
            }
        except Exception as e:
            logging.error(f"Error computing Williams Fractal: {e}")
            return {}

    def compute_adl(candle_data, dates, date_mask):
        try:
            adl_vals = indicators.get_adl(candle_data)
            return {'adl': [[d, safe_float(a.adl)] for d, a, m in zip(dates, adl_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing ADL: {e}")
            return {}

    def compute_cmf(candle_data, dates, date_mask):
        try:
            cmf_vals = indicators.get_cmf(candle_data)
            return {'cmf': [[d, safe_float(c.cmf)] for d, c, m in zip(dates, cmf_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing CMF: {e}")
            return {}

    def compute_chaikin_oscillator(candle_data, dates, date_mask):
        try:
            co_vals = indicators.get_chaikin_osc(candle_data)
            return {
                'chaikinoscillator': [[d, safe_float(co.oscillator)] for d, co, m in zip(dates, co_vals, date_mask) if
                                      m]}
        except Exception as e:
            logging.error(f"Error computing Chaikin Oscillator: {e}")
            return {}

    def compute_force_index(candle_data, dates, date_mask):
        try:
            fi_vals = indicators.get_force_index(candle_data, lookback_periods=13)
            return {'forceindex': [[d, safe_float(fi.force_index)] for d, fi, m in zip(dates, fi_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Force Index: {e}")
            return {}

    def compute_kvo(candle_data, dates, date_mask):
        try:
            kvo_vals = indicators.get_kvo(candle_data)
            return {'kvo': [[d, safe_float(kv.oscillator)] for d, kv, m in zip(dates, kvo_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing KVO: {e}")
            return {}

    def compute_mfi(candle_data, dates, date_mask):
        try:
            mfi_vals = indicators.get_mfi(candle_data)
            return {'mfi': [[d, safe_float(m.mfi)] for d, m, msk in zip(dates, mfi_vals, date_mask) if msk]}
        except Exception as e:
            logging.error(f"Error computing MFI: {e}")
            return {}

    def compute_obv(candle_data, dates, date_mask):
        try:
            obv_vals = indicators.get_obv(candle_data)
            return {'obv': [[d, safe_float(o.obv)] for d, o, m in zip(dates, obv_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing OBV: {e}")
            return {}

    def compute_pvo(candle_data, dates, date_mask):
        try:
            pvo_vals = indicators.get_pvo(candle_data)
            return {'pvo': [[d, safe_float(p.pvo)] for d, p, m in zip(dates, pvo_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing PVO: {e}")
            return {}

    def compute_alma(candle_data, dates, date_mask):
        try:
            alma_vals = indicators.get_alma(candle_data)
            return {'alma': [[d, safe_float(a.alma)] for d, a, m in zip(dates, alma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing ALMA: {e}")
            return {}

    def compute_dema(candle_data, dates, date_mask):
        try:
            dema_vals = indicators.get_dema(candle_data, lookback_periods=20)
            return {'dema': [[d, safe_float(dv.dema)] for d, dv, m in zip(dates, dema_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing DEMA: {e}")
            return {}

    def compute_epma(candle_data, dates, date_mask):
        try:
            epma_vals = indicators.get_epma(candle_data, lookback_periods=20)
            return {'epma': [[d, safe_float(ev.epma)] for d, ev, m in zip(dates, epma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing EPMA: {e}")
            return {}

    def compute_ema5(candle_data, dates, date_mask):
        try:
            ema5_vals = indicators.get_ema(candle_data, lookback_periods=5)
            return {'ema5': [[d, safe_float(e.ema)] for d, e, m in zip(dates, ema5_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing EMA5: {e}")
            return {}

    def compute_ema9(candle_data, dates, date_mask):
        try:
            ema9_vals = indicators.get_ema(candle_data, lookback_periods=9)
            return {'ema9': [[d, safe_float(e.ema)] for d, e, m in zip(dates, ema9_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing EMA9: {e}")
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

    def compute_hilbert(candle_data, dates, date_mask):
        try:
            hilbert_vals = indicators.get_ht_trendline(candle_data)
            return {
                'hilberttransform': [[d, safe_float(h.trendline)] for d, h, m in zip(dates, hilbert_vals, date_mask) if
                                     m]}
        except Exception as e:
            logging.error(f"Error computing Hilbert Transform: {e}")
            return {}

    def compute_hma(candle_data, dates, date_mask):
        try:
            hma_vals = indicators.get_hma(candle_data, lookback_periods=9)
            return {'hma': [[d, safe_float(h.hma)] for d, h, m in zip(dates, hma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing HMA: {e}")
            return {}

    def compute_kama(candle_data, dates, date_mask):
        try:
            kama_vals = indicators.get_kama(candle_data)
            return {'kama': [[d, safe_float(k.kama)] for d, k, m in zip(dates, kama_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing KAMA: {e}")
            return {}

    def compute_mama_fama(candle_data, dates, date_mask):
        try:
            mama_vals = indicators.get_mama(candle_data)
            return {
                'mama': [[d, safe_float(m.mama)] for d, m, msk in zip(dates, mama_vals, date_mask) if msk],
                'fama': [[d, safe_float(m.fama)] for d, m, msk in zip(dates, mama_vals, date_mask) if msk],
            }
        except Exception as e:
            logging.error(f"Error computing MAMA/FAMA: {e}")
            return {}

    def compute_sma(candle_data, dates, date_mask):
        try:
            sma_vals = indicators.get_sma(candle_data, lookback_periods=20)
            return {'sma': [[d, safe_float(s.sma)] for d, s, m in zip(dates, sma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing SMA: {e}")
            return {}

    def compute_smma(candle_data, dates, date_mask):
        try:
            smma_vals = indicators.get_smma(candle_data, lookback_periods=14)
            return {'smma': [[d, safe_float(s.smma)] for d, s, m in zip(dates, smma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing SMMA: {e}")
            return {}

    def compute_t3(candle_data, dates, date_mask):
        try:
            t3_vals = indicators.get_t3(candle_data)
            return {'t3': [[d, safe_float(t.t3)] for d, t, m in zip(dates, t3_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing T3: {e}")
            return {}

    def compute_tema(candle_data, dates, date_mask):
        try:
            tema_vals = indicators.get_tema(candle_data, lookback_periods=9)
            return {'tema': [[d, safe_float(t.tema)] for d, t, m in zip(dates, tema_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing TEMA: {e}")
            return {}

    def compute_vwap(candle_data, dates, date_mask):
        try:
            vwap_vals = indicators.get_vwap(candle_data)
            return {'vwap': [[d, safe_float(v.vwap)] for d, v, m in zip(dates, vwap_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing VWAP: {e}")
            return {}

    def compute_vwma(candle_data, dates, date_mask):
        try:
            vwma_vals = indicators.get_vwma(candle_data, lookback_periods=20)
            return {'vwma': [[d, safe_float(v.vwma)] for d, v, m in zip(dates, vwma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing VWMA: {e}")
            return {}

    def compute_wma(candle_data, dates, date_mask):
        try:
            wma_vals = indicators.get_wma(candle_data, lookback_periods=20)
            return {'wma': [[d, safe_float(w.wma)] for d, w, m in zip(dates, wma_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing WMA: {e}")
            return {}

    def compute_fisher_transform(candle_data, dates, date_mask):
        try:
            ft_vals = indicators.get_fisher_transform(candle_data, lookback_periods=10)
            return {
                'fishertransform_fisher': [[d, safe_float(ft.fisher)] for d, ft, m in zip(dates, ft_vals, date_mask) if
                                           m],
                'fishertransform_trigger': [[d, safe_float(ft.trigger)] for d, ft, m in zip(dates, ft_vals, date_mask)
                                            if m],
            }
        except Exception as e:
            logging.error(f"Error computing Fisher Transform: {e}")
            return {}

    def compute_zigzag(candle_data, dates, date_mask):
        try:
            zz_vals = indicators.get_zig_zag(candle_data)
            return {'zigzag': [[d, safe_float(zz.zig_zag)] for d, zz, m in zip(dates, zz_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Zig Zag: {e}")
            return {}

    def compute_atr(candle_data, dates, date_mask):
        try:
            atr_vals = indicators.get_atr(candle_data)
            return {'atr': [[d, safe_float(a.atr)] for d, a, m in zip(dates, atr_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing ATR: {e}")
            return {}

    def compute_bop(candle_data, dates, date_mask):
        try:
            bop_vals = indicators.get_bop(candle_data)
            return {'bop': [[d, safe_float(b.bop)] for d, b, m in zip(dates, bop_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing BOP: {e}")
            return {}

    def compute_choppiness(candle_data, dates, date_mask):
        try:
            chop_vals = indicators.get_chop(candle_data)
            return {'choppinessindex': [[d, safe_float(ci.chop)] for d, ci, m in zip(dates, chop_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Choppiness Index: {e}")
            return {}

    def compute_pmo(candle_data, dates, date_mask):
        try:
            pmo_vals = indicators.get_pmo(candle_data)
            return {
                'pmo': [[d, safe_float(p.pmo)] for d, p, m in zip(dates, pmo_vals, date_mask) if m],
                'pmo_signal': [[d, safe_float(p.signal)] for d, p, m in zip(dates, pmo_vals, date_mask) if m],
            }
        except Exception as e:
            logging.error(f"Error computing PMO: {e}")
            return {}

    def compute_roc(candle_data, dates, date_mask):
        try:
            roc_vals = indicators.get_roc(candle_data, lookback_periods=14)
            return {'roc': [[d, safe_float(r.roc)] for d, r, m in zip(dates, roc_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing ROC: {e}")
            return {}

    def compute_truerange(candle_data, dates, date_mask):
        try:
            tr_vals = indicators.get_tr(candle_data)
            return {'truerange': [[d, safe_float(t.tr)] for d, t, m in zip(dates, tr_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing True Range: {e}")
            return {}

    def compute_tsi(candle_data, dates, date_mask):
        try:
            tsi_vals = indicators.get_tsi(candle_data)
            return {'tsi': [[d, safe_float(t.tsi)] for d, t, m in zip(dates, tsi_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing TSI: {e}")
            return {}

    def compute_ulcer(candle_data, dates, date_mask):
        try:
            ui_vals = indicators.get_ulcer_index(candle_data)
            return {'ulcerindex': [[d, safe_float(u.ui)] for d, u, m in zip(dates, ui_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Ulcer Index: {e}")
            return {}

    def compute_slope(candle_data, dates, date_mask):
        try:
            slope_vals = indicators.get_slope(candle_data, lookback_periods=14)
            return {'slope': [[d, safe_float(s.slope)] for d, s, m in zip(dates, slope_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Slope: {e}")
            return {}

    def compute_standard_deviation(candle_data, dates, date_mask):
        try:
            std_vals = indicators.get_stdev(candle_data, lookback_periods=14)
            return {'standarddeviation': [[d, safe_float(s.stdev)] for d, s, m in zip(dates, std_vals, date_mask) if m]}
        except Exception as e:
            logging.error(f"Error computing Standard Deviation: {e}")
            return {}

    # === Map task keys to their corresponding functions ======================
    tasks = {
        'aroon': compute_aroon,
        'adx_dmi': compute_adx_dmi,
        'elder_ray': compute_elder_ray,
        'gator': compute_gator,
        'hurst': compute_hurst,
        'ichimoku': compute_ichimoku,
        'macd': compute_macd,
        'supertrend': compute_supertrend,
        'vortex': compute_vortex,
        'alligator': compute_alligator,
        'bollinger': compute_bollinger,
        'donchian': compute_donchian,
        'fcb': compute_fcb,
        'keltner': compute_keltner,
        'ma_envelopes': compute_ma_envelopes,
        'pivot_points': compute_pivot_points,
        'rolling_pivots': compute_rolling_pivots,
        'starc': compute_starc,
        'std_dev_channels': compute_std_dev_channels,
        'awesome': compute_awesome,
        'cci': compute_cci,
        'connors_rsi': compute_connors_rsi,
        'dpo': compute_dpo,
        'rsi': compute_rsi,
        'stc': compute_stc,
        'smi': compute_smi,
        'stoch': compute_stoch,
        'stoch_rsi': compute_stoch_rsi,
        'trix': compute_trix,
        'ultimate': compute_ultimate,
        'williamsr': compute_williamsr,
        'chandelier': compute_chandelier,
        'parabolic_sar': compute_parabolic_sar,
        'volatility_stop': compute_volatility_stop,
        'williams_fractal': compute_williams_fractal,
        'adl': compute_adl,
        'cmf': compute_cmf,
        'chaikin_oscillator': compute_chaikin_oscillator,
        'force_index': compute_force_index,
        'kvo': compute_kvo,
        'mfi': compute_mfi,
        'obv': compute_obv,
        'pvo': compute_pvo,
        'alma': compute_alma,
        'dema': compute_dema,
        'epma': compute_epma,
        'ema5': compute_ema5,
        'ema9': compute_ema9,
        'ema13': compute_ema13,
        'ema50': compute_ema50,
        'hilbert': compute_hilbert,
        'hma': compute_hma,
        'kama': compute_kama,
        'mama_fama': compute_mama_fama,
        'sma': compute_sma,
        'smma': compute_smma,
        't3': compute_t3,
        'tema': compute_tema,
        'vwap': compute_vwap,
        'vwma': compute_vwma,
        'wma': compute_wma,
        'fisher_transform': compute_fisher_transform,
        'zigzag': compute_zigzag,
        'atr': compute_atr,
        'bop': compute_bop,
        'choppiness': compute_choppiness,
        'pmo': compute_pmo,
        'roc': compute_roc,
        'truerange': compute_truerange,
        'tsi': compute_tsi,
        'ulcer': compute_ulcer,
        'slope': compute_slope,
        'standard_deviation': compute_standard_deviation,
    }

    if required_indicators:
        tasks = {key: func for key, func in tasks.items() if key in required_indicators}

    results = {}
    # Use ThreadPoolExecutor (threads share memory, so pickling is not needed)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(func, candle_data, dates, date_mask): key for key, func in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_task):
            key = future_to_task[future]
            try:
                task_result = future.result()
                results.update(task_result)
            except Exception as e:
                logging.error(f"Error in task {key}: {e}")

    gc.collect()

    # Reorder the final results to mimic the sequential function output
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

def process_indicators_data(args):
    quotes, from_date, to_date, interval = args
    output = calculate_indicators_data(quotes, from_date, to_date)
    return output
