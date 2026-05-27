# processing/pendulum_period.py
 
import logging
import numpy as np
import matplotlib.pyplot as plt
 
from ltpda.dsp.filter import IIR
#from ltpda.pzmodel import PZModel
logger = logging.getLogger(__name__)
 
 
def pendperiod(phi_ao, label: str = "", plot: bool = False) -> np.ndarray:
    """
    Estimate the pendulum period from a CAP PHI time series.
 
    Bandpass filters around the expected resonance frequency (~1e-3 Hz)
    to isolate the torsional oscillation, then finds peaks and troughs
    to estimate the half-period from successive extrema.
 
    Args:
        phi_ao : TSData ao of CAP PHI (radians)
        label  : label string for plot title (e.g. datestring)
        plot   : if True, show diagnostic plots
 
    Returns:
        np.ndarray of estimated periods in seconds (half-cycles doubled,
        with first and last dropped to avoid edge effects)
    """
    fs = phi_ao.fs
    y  = np.asarray(phi_ao.ydata(), dtype=float)
    x  = np.asarray(phi_ao.xdata(), dtype=float)
 
    # ------------------------------------------------------------------ #
    # Bandpass filter around pendulum resonance
    # Matches MATLAB pzmodel(1, {1e-3, 1.05e-3, 1.1e-3}, {}) --> pz = PZModel(poles=[1e-3, 1.05e-3, 1.1e-3])
    # which is a narrow bandpass centred on ~1e-3 Hz
    # ------------------------------------------------------------------ #
    filt = IIR.bandpass(fc=[1e-5, 1.1e-3], fs=fs, order=1)
    y_filt = np.asarray(filt.filtfilt(phi_ao).ydata(), dtype=float)
    logger.debug("Bandpass filter applied (1e-3 – 1.1e-3 Hz)")

    # ------------------------------------------------------------------ #
    # Find peaks and troughs
    # Matches MATLAB local max/min detection via shifted comparison
    # ------------------------------------------------------------------ #
    max_idx = _find_extrema(y_filt, kind='max')
    min_idx = _find_extrema(y_filt, kind='min')
    logger.debug("Found %d maxima, %d minima", len(max_idx), len(min_idx))
 
    # ------------------------------------------------------------------ #
    # Compute periods from successive extrema (peaks + troughs combined)
    # Each consecutive extremum is a half-cycle → multiply diff by 2
    # ------------------------------------------------------------------ #
    all_times = np.union1d(x[max_idx], x[min_idx])
    periods   = 2 * np.diff(all_times)
    periods   = periods[1:-1]   # drop first and last to remove edge effects
    logger.info("Period estimate: mean=%.1f s, std=%.1f s, n=%d half-cycles", 
                np.mean(periods), np.std(periods), len(periods))
 
    # ------------------------------------------------------------------ #
    # Diagnostic plots
    # ------------------------------------------------------------------ #
    if plot:
        _plot_period(x, y_filt, max_idx, min_idx, periods, label)
 
    return periods
 
 
# ======================================================================= #
# Private helpers
# ======================================================================= #
 
def _find_extrema(y: np.ndarray, kind: str) -> np.ndarray:
    """
    Find local maxima or minima using shifted array comparison.
    Matches MATLAB's vectorised peak detection exactly.
 
    A point y[i] is a maximum if y[i-1] < y[i] and y[i+1] < y[i].
    A point y[i] is a minimum if y[i-1] > y[i] and y[i+1] > y[i].
    """
    if kind == 'max':
        mask = (y[:-2] < y[1:-1]) & (y[2:] < y[1:-1])
    else:
        mask = (y[:-2] > y[1:-1]) & (y[2:] > y[1:-1])
 
    # Offset by 1 to account for the trimmed edges
    return np.where(mask)[0] + 1
 
 
def _plot_period(x: np.ndarray, y_filt: np.ndarray, max_idx: np.ndarray,
                 min_idx: np.ndarray, periods: np.ndarray, label: str):
    """Diagnostic plots matching MATLAB output."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
 
    # Filtered signal with extrema marked
    ax1.plot(x, y_filt, label='Filtered PHI')
    ax1.plot(x[max_idx], y_filt[max_idx], '*', label='Maxima')
    ax1.plot(x[min_idx], y_filt[min_idx], '*', label='Minima')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude [rad]')
    ax1.set_title('%s Filtered PHI with extrema' % label)
    ax1.legend()
    ax1.grid(True)
 
    # Period estimates
    n = len(periods)
    mean_p = np.mean(periods)
    std_p  = np.std(periods)
 
    ax2.plot(periods, '*', label='Data')
    ax2.axhline(mean_p,           color='r',     label='Mean (%.0f s)' % mean_p)
    ax2.axhline(mean_p + std_p,   color='r', linestyle='--', label='Std dev (%.0f s)' % std_p)
    ax2.axhline(mean_p - std_p,   color='r', linestyle='--')
    ax2.set_xlim([0, n])
    ax2.set_xlabel('Half-cycle number')
    ax2.set_ylabel('Estimated period [s]')
    ax2.set_title('%s Periods' % label)
    ax2.legend(loc='best')
    ax2.grid(True)
 
    plt.tight_layout()
    plt.show()