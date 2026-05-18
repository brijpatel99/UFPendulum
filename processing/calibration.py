import logging
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import signal
from typing import Union

logger = logging.getLogger(__name__)

TALE = 300  # filter settling buffer (seconds)


@dataclass
class CalibCoeffs:
    """
    Relative calibration coefficients between CH1 and CH2.

    relcal  : CH2/CH1 ratio from lowpass fit (DC regime)
    sol1    : [CH1_cal, CH2_cal] assuming CH1 nominal cal is correct
    sol2    : [CH1_cal, CH2_cal] assuming CH2 nominal cal is correct
    relcalx : CH2/CH1 ratio from bandpass fit (signal regime)
    sol1x   : same as sol1 but using bandpass-derived relcal
    sol2x   : same as sol2 but using bandpass-derived relcal
    """
    relcal:  float = np.nan
    sol1:    np.ndarray = None
    sol2:    np.ndarray = None
    relcalx: float = np.nan
    sol1x:   np.ndarray = None
    sol2x:   np.ndarray = None

    def get_coeffs(self, use_xcal: bool = False) -> np.ndarray:
        """
        Return a (3, 2) array of coefficient rows:
            [1, relcal]   — relative only
            sol1 or sol1x — absolute assuming CH1 cal correct
            sol2 or sol2x — absolute assuming CH2 cal correct

        Selects lowpass or bandpass variant based on use_xcal.
        """
        if use_xcal:
            return np.array([ [1.0, self.relcalx],
                               self.sol1x,
                               self.sol2x ])
        return np.array([ [1.0, self.relcal],
                           self.sol1,
                           self.sol2 ])

    def best_row(self, use_xcal: bool = False) -> np.ndarray:
        """
        Return the best available coefficient row:
        sol1 (or sol1x) if not NaN, else the relative-only row.
        """
        rows = self.get_coeffs(use_xcal)
        return rows[1] if not np.isnan(rows[1, 0]) else rows[0]


def relcal(data, timespan: tuple[float, float], use_xcal: bool = False, plot: bool = False,
           ) -> tuple[CalibCoeffs, dict]:
    """
    Compute relative calibration between CH1 and CH2.

    Performs two fits:
      - Lowpass  (< 0.1 Hz) : CH1_lpf vs -CH2_lpf  → relcal  (DC/drift regime)
      - Bandpass (0.1–1 Hz) : CH1_bpf vs  CH2_bpf  → relcalx (signal regime)

    Args:
        data      : DataBundle (from loaddata) or dict of aos.
                    If DataBundle, must have .fast['CH1Data'], .fast['CH2Data'],
                    .t, and optionally .slow['CH1Cal'], .slow['CH2Cal'].
                    If ao dict, must have keys 'CH1' and 'CH2', optionally
                    'CH1Cal' and 'CH2Cal' as slow ao objects.
        timespan  : (tstart, tstop) in pendulum time (seconds)
        use_xcal  : if True, use bandpass-derived coefficients for caldata output
        plot      : if True, show diagnostic plots

    Returns:
        coeff   : CalibCoeffs dataclass
        caldata : dict with 'phi' and 'x' arrays (or aos if input was ao dict)
    """
    tstart, tstop = timespan

    # ------------------------------------------------------------------ #
    # Unpack input — DataBundle or ao dict
    # ------------------------------------------------------------------ #
    from ..loaders.CAP_loader import DataBundle  # avoid circular at module level

    if isinstance(data, DataBundle):
        t    = data.t
        fs   = 1.0 / (t[1] - t[0])
        CH1  = data.fast['CH1Data'].astype(float)
        CH2  = data.fast['CH2Data'].astype(float)

        CH1cal, CH2cal = np.nan, np.nan
        if data.slow and 'CH1Cal' in data.slow and 'Time(s)' in data.slow:
            idx    = np.searchsorted(data.slow['Time(s)'], tstart)
            CH1cal = float(data.slow['CH1Cal'][idx])
            CH2cal = float(data.slow['CH2Cal'][idx])

        _ao_mode = False

    elif isinstance(data, dict):
        # ao dict — extract numpy arrays from TSData objects
        ch1_ao = data['CH1']
        ch2_ao = data['CH2']
        t   = ch1_ao.xdata() #+ ch1_ao.t0.timestamp()  # back to pendulum-like absolute t
        fs  = ch1_ao.fs
        CH1 = np.array(ch1_ao.ydata(), dtype=float)
        CH2 = np.array(ch2_ao.ydata(), dtype=float)

        CH1cal, CH2cal = np.nan, np.nan
        if 'CH1Cal' in data and 'CH2Cal' in data:
            idx    = np.searchsorted(np.array(data['CH1Cal'].xdata()), tstart)
            CH1cal = float(data['CH1Cal'].ydata()[idx])
            CH2cal = float(data['CH2Cal'].ydata()[idx])

        _ao_mode = True

    else:
        raise TypeError(f"data must be a DataBundle or a dict of aos, got {type(data)}")

    # ------------------------------------------------------------------ #
    # Coverage warnings
    # ------------------------------------------------------------------ #
    if t[0] > tstart or t[-1] < tstop:
        logger.warning("Available data does not cover requested calibration timespan "
                       "[%.3f, %.3f] — results may be inaccurate", tstart, tstop)
    elif t[0] > tstart - TALE or t[-1] < tstop + TALE:
        logger.warning("Less than %d s of filter settling buffer available — "
                       "low-frequency results may be inaccurate", TALE)
    # ------------------------------------------------------------------ #
    # Design filters
    # ------------------------------------------------------------------ #
    logger.debug("Designing lowpass and bandpass filters (fs=%.4f Hz)", fs)
    lpf_sos = _butter_lowpass(fs=fs)
    bpf_sos = _butter_bandpass(fs=fs)

    # ------------------------------------------------------------------ #
    # Filter
    # ------------------------------------------------------------------ #
    CH1_lpf = signal.sosfiltfilt(lpf_sos, CH1)
    CH2_lpf = signal.sosfiltfilt(lpf_sos, CH2)
    CH1_bpf = signal.sosfiltfilt(bpf_sos, CH1)
    CH2_bpf = signal.sosfiltfilt(bpf_sos, CH2)
    logger.debug("Filtering complete")

    # ------------------------------------------------------------------ #
    # Trim to requested timespan
    # ------------------------------------------------------------------ #
    t_mask = (t >= tstart) & (t <= tstop)
    CH1_lpft = CH1_lpf[t_mask]
    CH2_lpft = CH2_lpf[t_mask]
    CH1_bpft = CH1_bpf[t_mask]
    CH2_bpft = CH2_bpf[t_mask]
    t_trim   = t[t_mask]

    # ------------------------------------------------------------------ #
    # Fit — polyfit(x, y, 1): y = slope*x + intercept
    # relcal  = slope of CH1_lpf vs -CH2_lpf (normalised by std)
    # relcalx = slope of CH1_bpf vs  CH2_bpf (normalised by std)
    # The MATLAB polyfit normalises by mu=[mean, std]; slope/mu[1] = true slope
    # ------------------------------------------------------------------ #
    relcal_val  = _normalised_slope(CH1_lpft,  -CH2_lpft)
    relcalx_val = _normalised_slope(CH1_bpft,   CH2_bpft)
    logger.info("relcal  (lowpass)  = %.6f", relcal_val)
    logger.info("relcalx (bandpass) = %.6f", relcalx_val)

    # ------------------------------------------------------------------ #
    # Build coefficient struct
    # ------------------------------------------------------------------ #
    coeff = CalibCoeffs(
        relcal  = relcal_val,
        sol1    = np.array([CH1cal, CH1cal * relcal_val]),
        sol2    = np.array([CH2cal / relcal_val, CH2cal]),
        relcalx = relcalx_val,
        sol1x   = np.array([CH1cal, CH1cal * relcalx_val]),
        sol2x   = np.array([CH2cal / relcalx_val, CH2cal]),
    )

    # ------------------------------------------------------------------ #
    # Calibrated output
    # ------------------------------------------------------------------ #
    c = coeff.best_row(use_xcal=use_xcal)
    if not _ao_mode:
        caldata = { 'phi': (CH1 / c[0] - CH2 / c[1]) / 0.44,
                    'x':   (CH1 / c[0] + CH2 / c[1]) / 2.0}
    else:
        from ltpda.ao import ao
        from ..time_utils import pendtime_to_datetime
        t0  = pendtime_to_datetime(t[0])
        phi = ao(yvals=(CH1 / c[0] - CH2 / c[1]) / 0.44,
                 fs=fs, yunits='rad', name='PHI', t0=t0, type='tsdata')
        x   = ao(yvals=(CH1 / c[0] + CH2 / c[1]) / 2.0,
                 fs=fs, yunits='m',   name='X',   t0=t0, type='tsdata')
        caldata = {'phi': phi, 'x': x}

    # ------------------------------------------------------------------ #
    # Diagnostic plot
    # ------------------------------------------------------------------ #
    if plot:
        _plot_relcal(t_trim, CH1_lpft, CH2_lpft, CH1_bpft, CH2_bpft, coeff)

    return coeff, caldata


def get_calibration(data, timespan: tuple[float, float] = None, recalib: list = None, slow_only: bool = False, use_xcal: bool = False) -> np.ndarray:
    """
    Return a (1, 2) calibration coefficient array [coeff_CH1, coeff_CH2].

    If recalib=[refCH, tstart, tstop], runs relcal() over that interval
    and picks the appropriate solution row.
    Otherwise, reads nominal coefficients from slow data.

    Args:
        data      : DataBundle or ao dict
        timespan  : (tstart, tstop) — only used if recalib is None and
                    we need to find the right slow data index
        recalib   : [refCH, tstart, tstop] for relative calibration
        slow_only  : only slow data fields, calibration is taken from slow data
        use_xcal  : passed through to relcal()

    Returns:
        np.ndarray shape (1, 2)
    """
    if recalib is not None and not slow_only:
        ref_ch = int(recalib[0])
        ts = (float(recalib[1]), float(recalib[2]))
        logger.info("Running relcal (refCH=%d, %.3f-%.3f)", ref_ch, ts[0], ts[1])
        coeff, _ = relcal(data, timespan=ts)
        rows = coeff.get_coeffs()
        return rows[ref_ch + 1].reshape(1, 2)
    elif slow_only:
        if data.slow and 'CH1Cal' in data.slow:
            idx = np.searchsorted(data.slow['Time(s)'], timespan[0]) if timespan else 0
            ch1cal = float(data.slow['CH1Cal'][idx])
            ch2cal = float(data.slow['CH2Cal'][idx])
            logger.info("Using slow data calibration: CH1=%.6f, CH2=%.6f", ch1cal, ch2cal)
            return np.array([[ch1cal, ch2cal]])
        else:
            logger.warning("No slow calibration data found — using [1.0, 1.0]")
            return np.array([[1.0, 1.0]])
    else:
            logger.warning("No calibration data found — using [1.0, 1.0]")
            return np.array([[1.0, 1.0]])

# ======================================================================= #
# Private helpers
# ======================================================================= #

def _butter_lowpass(fs: float) -> np.ndarray:
    """
    Match MATLAB fdesign.lowpass(0.01, 0.1, 1, 60, fs) 
    with MatchExactly='stopband'.
    
    fpass=0.01 Hz, fstop=0.1 Hz, Apass=1 dB, Astop=60 dB
    MatchExactly stopband → meet Astop exactly, Apass is conservative.
    """
    nyq    = fs / 2.0
    wp     = 0.01 / nyq   # passband edge
    ws     = 0.10 / nyq   # stopband edge
    gpass  = 1            # dB ripple in passband
    gstop  = 60           # dB attenuation in stopband

    order, wn = signal.buttord(wp, ws, gpass, gstop)
    # buttord returns wn at the -3dB point that meets stopband exactly
    logger.debug("Lowpass filter: order=%d, wn=%.6f (normalised)", order, wn)
    return signal.butter(order, wn, btype='low', output='sos')


def _butter_bandpass(fs: float) -> np.ndarray:
    """
    Match MATLAB fdesign.bandpass(0.01, 0.3, 0.7, min(10,fs/2), 60, 1, 60, fs)
    with MatchExactly='stopband'.

    fstop1=0.01, fpass1=0.3, fpass2=0.7, fstop2=min(10, fs/2)
    Astop1=60dB, Apass=1dB, Astop2=60dB
    """
    nyq   = fs / 2.0
    fstop2 = min(10.0, fs / 2.0 - 0.01)  # keep away from Nyquist

    wp = [0.30  / nyq, 0.70   / nyq]   # passband edges
    ws = [0.01  / nyq, fstop2 / nyq]   # stopband edges
    gpass = 1
    gstop = 60

    order, wn = signal.buttord(wp, ws, gpass, gstop)
    logger.debug("Bandpass filter: order=%d, wn=%s (normalised)", order, wn)
    return signal.butter(order, wn, btype='band', output='sos')

def _normalised_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit y = slope * x + intercept and return slope.

    Matches MATLAB polyfit's normalised output: polyfit returns slope/mu[1]
    where mu[1] = std(x). We replicate this exactly so coefficients are
    consistent with the original pipeline.
    """
    x_std  = np.std(x)
    x_norm = (x - np.mean(x)) / x_std
    slope, _ = np.polyfit(x_norm, y, 1)
    return float(slope / x_std)


def _plot_relcal(t, CH1_lpf, CH2_lpf, CH1_bpf, CH2_bpf, coeff: CalibCoeffs):
    """Diagnostic 2x2 plot matching the MATLAB output."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Relative Calibration Diagnostics")

    # Scatter: lowpass
    axes[0, 0].plot(CH1_lpf, -CH2_lpf, '.', markersize=1)
    axes[0, 0].set_xlabel("CH1 (low pass) [V]")
    axes[0, 0].set_ylabel("-CH2 (low pass) [V]")
    axes[0, 0].grid(True)

    # Scatter: bandpass
    axes[0, 1].plot(CH1_bpf, CH2_bpf, '.', markersize=1)
    axes[0, 1].set_xlabel("CH1 (band pass) [V]")
    axes[0, 1].set_ylabel("CH2 (band pass) [V]")
    axes[0, 1].grid(True)

    # Time series: lowpass
    ax = axes[1, 0]
    rc = coeff.relcal
    ax.plot(t, _detrend(CH1_lpf),         'b',  lw=2, label='CH1 nominal')
    ax.plot(t, _detrend(-CH2_lpf),        'r',  lw=2, label='CH2 nominal')
    ax.plot(t, _detrend(CH1_lpf * rc),    'm--', lw=2, label=f'CH1 recal (×{rc:.3f})')
    ax.plot(t, _detrend(-CH2_lpf / rc),   'c--', lw=2, label=f'CH2 recal (×{1/rc:.3f})')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.legend(loc='best', fontsize=7)
    ax.grid(True)

    # Time series: bandpass
    ax = axes[1, 1]
    rcx = coeff.relcalx
    ax.plot(t, _detrend(CH1_bpf),          'b',  lw=2, label='CH1 nominal')
    ax.plot(t, _detrend(CH2_bpf),          'r',  lw=2, label='CH2 nominal')
    ax.plot(t, _detrend(CH1_bpf * rcx),   'm--', lw=2, label=f'CH1 recal (×{rcx:.3f})')
    ax.plot(t, _detrend(CH2_bpf / rcx),   'c--', lw=2, label=f'CH2 recal (×{1/rcx:.3f})')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.legend(loc='best', fontsize=7)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove mean (matches MATLAB detrend(...,'constant'))."""
    return x - np.mean(x)