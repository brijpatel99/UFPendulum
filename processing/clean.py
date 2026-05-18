"""
processing/clean.py

Clean CAP data: calibrate, fix slow data timing, and compute
signal combinations from raw loaded data.

Typical usage:
    from processing.clean import clean_cap_data 
    from config.config_loader import load_config

    cfg    = load_config("experiment_config.toml")
    result = clean_cap_data(cfg, cap_aos, ifo_ao)

    # Access results
    result.sens       # dict of calibrated sensing channel aos
    result.act        # dict of actuation channel aos
    result.slow       # dict of slow data aos (timing-corrected)
    result.signals    # dict of calibrated signal combinations (PHI, X, etc.)
"""

import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from ltpda import ao, consolidate
logger = logging.getLogger(__name__)

ARM = .222 # pendulum arm length [m] (brij - update to get with pend params)

# ======================================================================= #
# Output dataclass
# ======================================================================= #

@dataclass
class CAPData:
    """
    Clean CAP data output from process_cap_data().

    Attributes:
        sens    : calibrated sensing channel aos  (CH1–CH4)
        act     : actuation channel aos           (ACh1–ACh4)
        slow    : timing-corrected slow data aos  (keyed by friendly name)
        signals : computed signal combinations    (CAP_PHI, CAP_X, etc.)
    """
    sens:    dict = field(default_factory=dict)
    act:     dict = field(default_factory=dict)
    slow:    dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)


# ======================================================================= #
# Main pipeline
# ======================================================================= #

def clean_cap_data(cfg: dict, cap_aos: dict[str, ao], ifo_ao: Optional[ao] = None) -> CAPData:
    """
    Full CAP data cleaning pipeline.

    Steps:
        1. Calibrate sensing channels (V → m or rad)
        2. Scale actuation channels (counts → V)
        3. Fix slow data periodic time shift
        4. Compute signal combinations per config
        5. Calibrate signals to IFO

    Args:
        cfg       : config dict from load_config()
        cap_aos   : raw ao dict from loadCAP_ao()
        ifo_ao    : raw IFO PHI ao from loadIFO_ao()

    Returns:
        CAPData with sens, act, slow, signals populated
    """
    result = CAPData()

    # Step 1 — sensing channels
    result.sens = _calibrate_sensing(cap_aos, cfg)

    # Step 2 — actuation channels
    result.act  = _scale_actuation(cap_aos, cfg)

    # Step 3 — slow data timing fix
    result.slow = _fix_slow_timing(cap_aos, cfg)

    # Step 4 — signal combinations
    result.signals = _compute_signal_combinations(result.sens, cfg)

    #Step 5 — consolidate and calibrate signals to IFO
    if ifo_ao is not None:
        result.signals, result.sens, result.act, result.slow, ifo_ao = _consolidate_groups(
        result.signals, result.sens, result.act, result.slow, ifo_ao, cfg)

        result.signals = _calibrate_signals_to_ifo(result.signals, ifo_ao)
    else:
        logger.info("No IFO ao provided — skipping IFO consolidation/calibration")
    
    # Send out cleaned CAP data
    logger.info("clean_cap_data complete — %d sens, %d act, %d slow, %d signals",
                len(result.sens), len(result.act), len(result.slow), len(result.signals))
    return result, ifo_ao

# ======================================================================= #
# Private pipeline steps
# ======================================================================= #

def _calibrate_sensing(cap_aos: dict, cfg: dict) -> dict[str, ao]:
    """
    Apply sensing calibration values to CH1–CH4.
    Returns dict of calibrated sensing aos.
    """
    sens_cfg   = cfg['sensing']
    calib_vals = sens_cfg['calib_vals']
    units      = sens_cfg.get('calib_units', 'V/m')

    # Sensing channel names from loaded CAP data
    ch_names = sens_cfg.get('sensChannels', ['CH1', 'CH2', 'CH3Data', 'CH4Data'])
    sens     = {}

    for i, name in enumerate(ch_names):
        if name not in cap_aos:
            logger.warning("Sensing channel '%s' not found in cap_aos — skipping", name)
            continue
        cal  = ao(calib_vals[i], yunits=units)
        ch   = cap_aos[name]
        sens[name] = ch / cal
        sens[name].setYunits('V')
        logger.info("Calibrated %s with %.6f %s", name, calib_vals[i], units)

    return sens


def _scale_actuation(cap_aos: dict, cfg: dict) -> dict[str, ao]:
    """
    Scale actuation channels by the actuation scale factor.
    Returns dict of scaled actuation aos.
    """
    act_cfg   = cfg['actuation']
    scale     = act_cfg['scale']
    units     = act_cfg.get('scale_units', 'V^-1')
    act_scale = ao(scale, yunits=units)

    # Actuation channel names from loaded CAP data
    ch_names = act_cfg.get('actChannels', ['Elec1DCAct', 'Elec2DCAct', 'Elec3DCAct', 'Elec4DCAct'])
    act       = {}

    for i, name in enumerate(ch_names):
        if name not in cap_aos:
            logger.warning("Actuation channel '%s' not found in cap_aos — skipping", name)
            continue
        ch   = cap_aos[name]
        act[name] = ch / act_scale
        logger.info("Scaled actuation channel '%s' by 1/%.4f %s", name, scale, units)

    return act


def _fix_slow_timing(cap_aos: dict, cfg: dict) -> dict[str, ao]:
    """
    Correct the periodic time shift in slow data channels.

    The slow data writer introduces a periodic timing error. This fixes it
    by rebuilding the time axis from the mean sample interval, preserving
    the original t0 and first sample offset.

    Slow channel names are auto-detected from cap_aos.slow keys,
    then optionally renamed via the [slow_data.name_map] config section.

    Args:
        cap_aos    : full ao dict from loadCAP() — slow aos prefixed 'slow_'
        cfg        : config dict

    Returns:
        dict of timing-corrected slow data aos
    """
    slow_cfg  = cfg.get('slow_data', {})
    name_map  = slow_cfg.get('name_map', {})

    # Slow data aos in cap_aos are prefixed 'slow_'
    slow_keys = [k for k in cap_aos if k.startswith('slow_')]
    if not slow_keys:
        logger.warning("No slow_ channels found in cap_aos")
        return {}

    # Compute corrected sample interval from raw slow timing
    t_raw  = cap_aos[slow_keys[0]].xdata()
    if t_raw is None:
        logger.warning("No Time_s in data_orig.slow — cannot fix slow timing")
        return {k: cap_aos[k] for k in slow_keys}

    ts = (np.max(t_raw) - np.min(t_raw)) / len(t_raw)   # mean interval
    logger.debug("Slow data mean sample interval: %.4f s (fs=%.6f Hz)", ts, 1.0/ts)

    slow_out = {}
    for key in slow_keys:
        raw_ao    = cap_aos[key]
        # Rebuild ao with corrected fs, preserving t0 and x(0) offset
        fixed     = ao(yvals=raw_ao.ydata(), fs=1.0/ts, type='tsdata')
        fixed.t0  = raw_ao.t0
        #fixed.timeshift(float(raw_ao.xdata()[0])) #brij - fix this when timeshift is available

        # Determine friendly name
        raw_name  = key[5:]   # strip 'slow_' prefix
        friendly  = name_map.get(raw_name, raw_name)
        fixed.setName(friendly)

        slow_out[friendly] = fixed
        logger.debug("Fixed slow timing for '%s' to '%s'", key, friendly)

    logger.info("Fixed timing for %d slow channel(s)", len(slow_out))
    return slow_out


def _compute_signal_combinations(sens: dict[str, ao], cfg: dict) -> dict[str, ao]:
    """
    Compute signal combinations from calibrated sensing channels per config.

    Reads [signal_combinations] section of config — each entry is:
        D* = CH*

    Args:
        sens  : calibrated sensing channel dict (CH1–CH4)
        cfg   : config dict
        ppars : PendulumParams for arm length

    Returns:
        dict mapping signal name → ao
    """
    sig_cfg   = cfg.get('signal_combinations', {})

    # Get which channels are connected to specific differential measurements
    D1        = sig_cfg.get('D1', 'CH1')
    D2        = sig_cfg.get('D2', 'CH2')
    D3        = sig_cfg.get('D3', 'CH3DATA')

    # Get combinations to compute
    compute   = sig_cfg.get('signals', ['PHI', 'X_BULK'])
    signals   = {}

    for name in compute:
        try:
            if name == 'X_LL':
                signals[name] = (sens[D1] + sens[D2])/2
            elif name == 'PHI_LL':
                signals[name] = (sens[D1] - sens[D2])/(2*ARM)
            elif name == 'X_BULK':
                signals[name] = ((sens[D1] + sens[D2])/2 + sens[D3])/2
            elif name == 'PHI':
                signals[name] = ((sens[D1] + sens[D2])/2 - sens[D3])/(2*ARM)
            elif name == 'X_DIFF':
                signals[name] = ((sens[D1] + sens[D2])/2 - sens[D3])/2
            else:
                logger.warning("Provided unknown signal to compute: '%s' (check experiment config file)", name)
        except Exception as e:
            logger.warning("Failed to compute signal '%s': %s", name, e)
        logger.info("Computed signal '%s'", name)
    
    return signals


def _consolidate_groups(signals: dict[str, ao], sens: dict[str, ao], act: dict[str, ao], slow: dict[str, ao],
                        ifo_ao: Optional[ao], cfg: dict) -> tuple[dict, dict, dict, dict, Optional[ao]]:
 
    do_signals = cfg['signal_combinations'].get('consolidate')
    do_sens    = cfg['sensing'].get('consolidate')
    do_act     = cfg['actuation'].get('consolidate')
    do_slow    = cfg['slow_data'].get('consolidate')
 
    # Build flat list of all aos to consolidate, tracking group membership
    # so we can repack after. IFO is always included as reference.
    all_aos   = []
    all_names = []   # (group_label, key) tuples

    def _add_group(group: dict, label: str, enabled: bool):
        if enabled and group:
            for name, a in group.items():
                all_aos.append(a)
                all_names.append((label, name))
            logger.debug("Added '%s' group (%d ao(s)) to consolidation", label, len(group))
        elif enabled and not group:
            logger.warning("'%s' group enabled for consolidation but is empty", label)

    _add_group(signals, 'signals', do_signals)
    _add_group(sens,    'sens',    do_sens)
    _add_group(act,     'act',     do_act)
    _add_group(slow,    'slow',    do_slow)

    if not all_aos:
        logger.info("No groups enabled for consolidation — skipping")
        return signals, sens, act, slow, ifo_ao

    # Always append IFO last as reference
    all_aos.append(ifo_ao)
    all_names.append(('__ifo__', '__ifo__'))

    logger.info("Consolidating %d ao(s) + IFO at fs=%.6f Hz (from IFO)",len(all_aos) - 1, ifo_ao.fs)

    # Single consolidate call — fs taken from IFO
    result = consolidate(*tuple(all_aos), fs=ifo_ao.fs)

    # Repack results back into their original group dicts
    out = {'signals': {}, 'sens': {}, 'act': {}, 'slow': {}}
    out_ifo = ifo_ao

    for i, (label, name) in enumerate(all_names):
        if label == '__ifo__':
            out_ifo = result[i]
        else:
            out[label][name] = result[i]

    # Merge back — unconsolidated groups pass through unchanged
    return (out['signals'] if do_signals else signals,
            out['sens']    if do_sens    else sens,
            out['act']     if do_act     else act,
            out['slow']    if do_slow    else slow,
            out_ifo)

def _calibrate_signals_to_ifo(signals: dict[str, ao], ifo_ao: ao) -> dict[str, ao]:
    """
    Calibrate each signal ao against IFO reference using bilinfit.
    Replaces raw signal with calibrated version in-place.

    Uses the full loaded time interval for fitting — no windowing needed
    since data is already trimmed and consolidated on load.

    Args:
        signals : computed signal dict from _compute_signal_combinations()
        ifo_ao  : consolidated IFO PHI ao (same time grid as signals)

    Returns:
        dict of calibrated signal aos, same keys as input
    """
    from ltpda import bilinfit

    calibrated = {}
    for name, sig in signals.items():
        try:
            pp = bilinfit([sig], ifo_ao)
            cal_sig = pp.eval([sig])
            #cal_sig.setName(name)
  
            calibrated[name] = cal_sig
            logger.info("Calibrated '%s' to IFO: coeff=%.6f, offset=%.6f", name, pp.y[0], pp.y[1])
        except Exception as e:
            logger.warning("Failed to calibrate '%s' to IFO: %s — keeping raw", name, e) 
            calibrated[name] = sig

    return calibrated