"""
processing/ao_builder.py

All functions that produce ao (TSData) objects from raw data.

    loadCAP_ao      -- load CAP binary data → dict of named ao objects
    loadIFO_ao      -- load IFO text data   → PHI ao object
    build_aos       -- raw DataBundle + calibration → dict of named ao objects

Raw loading:
    loaders/CAP_loader.py        -- loadCAP()  → DataBundle
    loaders/IFO_loader.py        -- loadIFO()  → np.ndarray

Calibration:    processing/calibration.py
Downsampling:   processing/resample.py
"""

import logging
import numpy as np
from typing import Optional

from ltpda.ao import ao

from ..time_utils.pendTimeLTPDA import pendTimeLTPDA
from ..loaders.CAP_loader import loadCAP, DataBundle
from ..loaders.IFO_loader import loadIFO
from ..processing.calibration import get_calibration

logger = logging.getLogger(__name__)

# Physical conversion factor: laser wavelength / (8 * pi * arm_length)
TO_PHI = 1064e-9 / (8 * np.pi * 0.22)   # rad per fringe unit

# Channels requiring special construction (derived or unit-labelled)
_FAST_CH_PATTERN = {'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'}

DEFAULT_CHANNELS = ['PHI', 'X', 'CH1', 'CH2']


# ======================================================================= #
# Public API
# ======================================================================= #

def loadCAP_ao(filepath: str, tstart: float, tstop: float, chan_list: Optional[list[str]] = None, downsample_factor: int = 1, 
               recalib: Optional[list] = None, slow_only: bool = False, trim: bool = False) -> tuple[dict, DataBundle]:
    """
    Load CAP binary data and return calibrated ao objects.

    Enforces interp=True (uniform sampling) before building aos,
    matching the MATLAB loaddataao2024 behavior.

    Args:
        filepath            : folder containing *_Time Series*.bin files
        tstart              : interval start (pendulum time, seconds)
        tstop               : interval end (pendulum time, seconds)
        chan_list           : additional channel names beyond the default
                              ['PHI', 'X', 'CH1', 'CH2']. Slow channels
                              should be prefixed with 'slow_' e.g. 'slow_Temp'.
                              Duplicates are dropped while preserving order.
        downsample_factor   : integer downsampling factor (1 = no downsampling)
        recalib (legacy)    : [refCH, tstart, tstop] to compute calibration via
                              relcal() rather than using slow data coefficients.
                              refCH=0 means CH1 is the reference channel.
        slow_only           : skip fast binary data entirely — only slow sidecar
                              data is loaded. PHI/X/CHx channels are excluded.
        trim                : trim output to exactly [tstart, tstop]

    Returns:
        (aos, data) :
            aos  -- dict mapping channel name → TSData ao
            data -- raw DataBundle from loadCAP()
    """
    # Build deduplicated channel list preserving order
    aolist = list(dict.fromkeys(DEFAULT_CHANNELS + (chan_list or [])))
    if slow_only:
        aolist = [c for c in aolist if c not in DEFAULT_CHANNELS]

    logger.debug("Channel list: %s", aolist)

    # 1. Load raw data (interp enforced)
    data = loadCAP(filepath, tstart, tstop, slow_only=slow_only, trim=trim, interp=True)

    # 2. Calibration coefficients
    calcoeff = get_calibration(data, timespan=(tstart, tstop), recalib=recalib, slow_only=slow_only)

    # 3. Build ao objects
    aos = build_aos(data, calcoeff, aolist, slow_only=slow_only)

    # 4. Downsample
    aos_ds = {}
    if downsample_factor > 1:
        for name,_ in aos.items():
            if name.startswith('slow_'):
                aos_ds[name] = aos[name]
                continue
            aos_ds[name] = aos[name].downsample(downsample_factor)
            logger.info("Downsampling '%s' by factor of %d", name, downsample_factor)

    logger.info("loadCAP_ao complete — %d channel(s) returned", len(aos_ds))
    return aos_ds, data


def loadIFO_ao(filepath: str, tstart: float, tstop: float, interp_param=None, 
               downsample_factor: int = 1, tcorr: bool = False, trim: bool = False) -> ao:
    """
    Load IFO text data and return a calibrated PHI ao object.

    Enforces interp=True (uniform sampling), matching MATLAB behaviour
    where passarg = {'interp', 0} is always set.

    Args:
        filepath     : folder containing *IFODataFile*.txt files
        tstart       : interval start (pendulum time, seconds)
        tstop        : interval end (pendulum time, seconds)
        interp_param : target sample rate / time vector for interpolation onto uniform grid.
                            - None      : no interpolation
                            - 0         : use modal fs of original data
                            - float     : specifies sample rate (fs) to use
                            - np.ndarray: specifies uniform time vector to use
        downsample_factor   : integer downsampling factor (1 = no downsampling)
        tcorr               : apply time vector correction for known DAQ bug
                              (corrupted initial timestamps, present since 2016-11-21)
        trim                : trim output to exactly [tstart, tstop]

    Returns:
        TSData ao — PHIifo in radians
    """

    # Load raw IFO data (interp always enforced)
    data = loadIFO(filepath, tstart, tstop, trim=trim, interp_param=interp_param, tcorr=tcorr)

    # Build PHI ao
    t      = data[:, 0]
    phi    = data[:, 1] * TO_PHI
    t0     = pendTimeLTPDA(t[0])
    ao_fs  = 1.0 / (t[1] - t[0])
    logger.info("Loaded IFO ao: fs=%.6f Hz, %d samples", ao_fs, len(phi))

    aos = {'PHI': ao(yvals=phi, fs=ao_fs, yunits='rad', name='PHIifo', t0=t0, type='tsdata')}
    logger.info("Built %d ao object(s): %s", len(aos), list(aos.keys()))

    # Downsample
    if downsample_factor > 1:
        aos_ds = {}
        for name,_ in aos.items():
            aos_ds[name] = aos[name].downsample(downsample_factor)
            logger.info("Downsampling '%s' by factor of %d", name, downsample_factor)

        logger.info("Returning IFO ao: fs=%.6f Hz, %d samples", aos_ds[name].fs, aos_ds[name].size())
        return aos_ds
    else:
        logger.info("Returning IFO ao: fs=%.6f Hz, %d samples", aos['PHI'].fs, aos['PHI'].size())
        return aos

def build_aos(data: DataBundle, calcoeff: np.ndarray, aolist: list[str], slow_only: bool = False) -> dict[str, ao]:
    """
    Construct named ao objects from a loaded DataBundle.

    Args:
        data      : DataBundle from loadCAP()
        calcoeff  : (1, 2) calibration array [coeff_CH1, coeff_CH2]
        aolist    : channel names to build
        slow_only : if True, fast data is unavailable — only slow_ channels built

    Returns:
        dict mapping channel name → TSData ao
    """
    # Fast data time axis and metadata
    if not slow_only and data.t is not None:
        aotime = data.t - data.info['startTime_fast']
        fs     = 1.0 / (aotime[1] - aotime[0])
        t0     = pendTimeLTPDA(data.info['startTime_fast'])
    else:
        aotime, fs, t0 = None, None, None

    result = {}
    for name in aolist:
        ts = _build_single_ao(name, data, calcoeff, aotime, fs, t0, slow_only)
        if ts is not None:
            result[name] = ts
        else:
            logger.warning("Could not build ao for '%s' — skipping", name)

    logger.info("Built %d ao object(s): %s", len(result), list(result.keys()))
    return result


# ======================================================================= #
# Private helpers
# ======================================================================= #

def _build_single_ao(name: str, data: DataBundle, calcoeff: np.ndarray, aotime: Optional[np.ndarray], 
                     fs: Optional[float], t0, slow_only: bool) -> Optional[ao]:
    """
    Build a single named ao. Returns None if channel unavailable.
    """
    try:
        if name == 'PHI':
            if slow_only or aotime is None:
                logger.warning("PHI requires fast data — skipping")
                return None
            y = (data.fast['CH1Data'] / calcoeff[0, 0] - data.fast['CH2Data'] / calcoeff[0, 1]) / 0.44
            return ao(yvals=y, fs=fs, yunits='rad', name='PHI', t0=t0, type='tsdata')

        elif name == 'X':
            if slow_only or aotime is None:
                logger.warning("X requires fast data — skipping")
                return None
            y = (data.fast['CH1Data'] / calcoeff[0, 0] + data.fast['CH2Data'] / calcoeff[0, 1]) / 2.0
            return ao(yvals=y, fs=fs, yunits='m', name='X', t0=t0, type='tsdata')

        elif name.upper() in _FAST_CH_PATTERN:
            if slow_only or aotime is None:
                logger.warning("%s requires fast data — skipping", name)
                return None
            field = "%sData" % name.upper()
            if field not in data.fast:
                logger.warning("Field '%s' not found in fast data", field)
                return None
            return ao(yvals=data.fast[field], fs=fs,
                      yunits='V', name=name.upper(), t0=t0, type='tsdata')

        elif name.startswith('slow_'):
            slow_name = name[5:]   # strip 'slow_' prefix
            if not data.slow or slow_name not in data.slow:
                logger.warning("Slow channel '%s' not found in data.slow", slow_name)
                return None
            slow_t  = data.slow['Time(s)'] - data.info['startTime_slow']
            slow_fs = 1.0 / (slow_t[1] - slow_t[0])
            return ao(yvals=data.slow[slow_name], fs=slow_fs, name=name, t0=t0, type='tsdata')

        else:
            # Generic fast channel by exact field name
            if name not in data.fast:
                logger.warning("Channel '%s' not found in fast data", name)
                return None
            return ao(yvals=data.fast[name], fs=fs, name=name.upper(), t0=t0, type='tsdata')

    except KeyError as e:
        logger.warning("Missing field %s while building ao for '%s'", e, name)
        return None

