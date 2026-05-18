import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import os
import re
from .parser import parsedata

logger = logging.getLogger(__name__)

FPGA_RATE = 4e7  # Hz

@dataclass
class DataBundle:
    """Equivalent to MATLAB's data struct output from loaddata()."""
    t: np.ndarray = None
    T: np.ndarray = None
    fast: dict = field(default_factory=dict)   # all fast data fields
    slow: dict = field(default_factory=dict)   # slowData fields
    info: dict = field(default_factory=dict)


def loadCAP(filepath: str, tstart: float, tstop: float,
    slow_only: bool = False, trim: bool = False, interp: bool = False) -> Optional[DataBundle]:
    """
    Identify and stitch together binary time series files spanning [tstart, tstop].
    
    Args:
        filepath  : folder containing the .bin files
        tstart    : start of requested interval (pendulum time, seconds)
        tstop     : end of requested interval (pendulum time, seconds)
        slow_only : only load slow data fields
        trim      : trim output to exactly [tstart, tstop]
        interp    : interpolate fast data to constant sample rate

    Returns:
        DataBundle, or None if no files match the requested interval
    """
    # ------------------------------------------------------------------ #
    # Check and standardize inputs
    # ------------------------------------------------------------------ #
    filepath = filepath.rstrip('/\\') + '/'

    if tstart > tstop:
        logger.warning("tstart > tstop — swapping values")
        tstart, tstop = tstop, tstart

    # ------------------------------------------------------------------ #
    # Select files
    # ------------------------------------------------------------------ #
    logger.info("Selecting CAP files to load")
    filenames, stimes = _get_sorted_bin_files(filepath)
    stimes = np.array(stimes)

    selected_files, selected_times = _select_cap_files(filenames, stimes, tstart, tstop)
    nfiles = len(selected_files)
    logger.info("Selected %d CAP file(s) to load", nfiles)
    logger.info("Files: %s", selected_files)

    # ------------------------------------------------------------------ #
    # Load files
    # ------------------------------------------------------------------ #
    logger.debug("Loading %d file(s)", nfiles)
    fast_chunks = []   # list of dicts, one per file
    slow_chunks = []
    file_versions = np.zeros(nfiles, dtype=float)
    jump_flags    = np.zeros(nfiles, dtype=bool)
    samp_ints     = np.zeros(nfiles)

    for kk, fname in enumerate(selected_files):
        logger.debug(f"Loading file {kk+1}/{nfiles}: {fname}")
        raw = parsedata(filepath + fname, slow_only=slow_only)

        has_slow = 'slowData' in raw
        fast_chunk = {k: v for k, v in raw.items() if k != 'slowData'}
        file_versions[kk] = raw.get('info', {}).get('version', 1)

        if not slow_only:
            # Handle T field
            if 'T' in fast_chunk:
                T = np.array(fast_chunk['T'], dtype=np.float64)
                fast_chunk['T'] = T

                if file_versions[kk] <= 1.0:
                    jump_flags[kk] = T[0] == 0
                    samp_ints[kk]  = _mode(T)
                else:
                    jump_flags[kk] = fname[-5] == 'e'
                    samp_ints[kk]  = _mode(np.mod(np.diff(T), 2**32))

            fast_chunks.append(fast_chunk)

        if has_slow:
            slow_chunks.append(raw['slowData'])

    # ------------------------------------------------------------------ #
    # Fill missing fields with NaN so concatenation stays in sync
    # ------------------------------------------------------------------ #
    fast_chunks = _fill_missing_fields(fast_chunks)
    if slow_chunks:
        slow_chunks = _fill_missing_fields(slow_chunks)

    # ------------------------------------------------------------------ #
    # Post-process time vector T
    # ------------------------------------------------------------------ #
    fast_chunks, jump_flags, samp_ints = _process_time_vectors(
        fast_chunks, file_versions, jump_flags, samp_ints, selected_times
    )

    # ------------------------------------------------------------------ #
    # Concatenate
    # ------------------------------------------------------------------ #
    logger.info("Concatenating CAP data")
    bundle = DataBundle()
    if fast_chunks:
        fast_keys = [k for k in fast_chunks[0] if k != 'info']
        for key in fast_keys:
            bundle.fast[key] = np.concatenate([c[key] for c in fast_chunks])
    if slow_chunks:
        slow_keys = list(slow_chunks[0].keys())
        for key in slow_keys:
            bundle.slow[key] = np.concatenate([c[key] for c in slow_chunks])

    # ------------------------------------------------------------------ #
    # Build time vector
    # ------------------------------------------------------------------ #
    if 'T' in bundle.fast:
        bundle.T = bundle.fast['T']
        bundle.t = np.cumsum(bundle.T / FPGA_RATE) + fast_chunks[0]['info']['startTime']#bundle.info['startTime']
        _check_time_coverage(bundle.t, tstart, tstop)

    # ------------------------------------------------------------------ #
    # Interpolate to constant sample rate
    # ------------------------------------------------------------------ #
    if 'T' in bundle.fast and interp:
        logger.info("Interpolating CAP data")
        bundle = _interpolate(bundle, fast_keys, samp_ints)

    # ------------------------------------------------------------------ #
    # Trim to requested interval
    # ------------------------------------------------------------------ #
    if trim:
        logger.info("Trimming data")
        if not slow_only: # if fast data exists
            bundle = _trim(bundle, fast_keys, tstart, tstop)
            logger.info("CAP fast data trimmed to range [%d, %d]", bundle.t[0], bundle.t[-1])
            logger.info("CAP slow data trimmed to range [%d, %d]", bundle.slow['Time(s)'][0], bundle.slow['Time(s)'][-1])
        else: # for slow_only
            bundle = _trim(bundle, [], tstart, tstop)
            logger.info("CAP slow data trimmed to range [%d, %d]", bundle.slow['Time(s)'][0], bundle.slow['Time(s)'][-1])

    # ------------------------------------------------------------------ #
    # Update bundle information
    # ------------------------------------------------------------------ #
    if not slow_only:
        bundle.info = {'startTime_fast': bundle.t[0],
                       'stopTime_fast': bundle.t[-1],
                       'nsamples_fast': len(bundle.t),
                       'startTime_slow': bundle.slow['Time(s)'][0],
                       'stopTime_slow': bundle.slow['Time(s)'][-1],
                       'nsamples_slow': len(bundle.slow['Time(s)'])}
    else:
        bundle.info = {'startTime': bundle.slow['Time(s)'][0],
                       'nsamples': len(bundle.slow['Time(s)'])}
    logger.info("loadCAP complete — CAP data info: %r", bundle.info)
    return bundle


# ======================================================================= #
# Private helpers
# ======================================================================= #
def _get_sorted_bin_files(filepath: str) -> tuple[list[str], list[float]]:
    """
    Scans filepath for 'Time Series' .bin files,
    extracts start times from filenames, and returns
    both sorted in ascending time order.
    """
    logger.debug("Scanning for bin files in: %s", filepath)
    all_files = os.listdir(filepath)
    bin_files = [f for f in all_files if re.search(r'.*Time Series.*\.bin', f)]
    logger.debug("Found %d matching bin files", len(bin_files))

    # Parsing times in the filenames
    stimes = []
    for fname in bin_files:
        ms_str = fname.split('_')[0]
        stimes.append(float(ms_str) / 1000.0)

    sorted_pairs = sorted(zip(stimes, bin_files))
    stimes, filenames = zip(*sorted_pairs) if sorted_pairs else ([], [])
    logger.debug("Sorted %d files by start time (%.3f s to %.3f s)", len(filenames), stimes[0] if stimes else 0, stimes[-1] if stimes else 0)#→

    return list(filenames), list(stimes)

def _select_cap_files(filenames: list[str], stimes: np.ndarray, tstart: float, tstop: float) -> tuple[list[str], np.ndarray]:
    """
    Select files whose data overlaps with [tstart, tstop].
 
    A file overlaps if:
        - Its start time is before tstop (file starts before interval ends)
        - Its end time (approximated by the next file's start) is after tstart
 
    The last file has no known end time, so it's included if it starts before tstop.
 
    Raises:
        ValueError : if no files overlap the requested interval at all
        ValueError : if tstart >= tstop
    """
    if tstart >= tstop:
        raise ValueError(f"tstart ({tstart}) must be less than tstop ({tstop})")
 
    n = len(stimes)
 
    # A file i overlaps [tstart, tstop] if:
    # file starts before tstop  AND  next file starts after tstart
    mask = np.zeros(n, dtype=bool)
    for i in range(1,n):
        starts_before_tstop = stimes[i-1] < tstop
        ends_after_tstart   = stimes[i] > tstart
        mask[i] = starts_before_tstop and ends_after_tstart
 
    if not np.any(mask):
        raise ValueError("No CAP files found overlapping [%.3f, %.3f]. Available range: [%.3f, %.3f]" % (tstart, tstop, stimes[0], stimes[-1]))
    else:
        mask[np.argmax(mask)-1] = True
 
    selected_files = [filenames[i] for i in range(n) if mask[i]]
    selected_times = stimes[mask]
 
    logger.info("Selected %d file(s) covering [%.3f, %.3f]", len(selected_files), selected_times[0], selected_times[-1])
    logger.debug("Selected files: %s", selected_files)
 
    return selected_files, selected_times

def _mode(arr: np.ndarray) -> float:
    """Return the modal value of an array (fast, no scipy dependency)."""
    values, counts = np.unique(arr, return_counts=True)
    return float(values[np.argmax(counts)])


def _fill_missing_fields(chunks: list[dict]) -> list[dict]:
    """Fill missing fields across file chunks with NaN arrays."""
    all_keys = set().union(*[c.keys() for c in chunks])
    for kk, chunk in enumerate(chunks):
        nsamples = chunk.get('info', {}).get('nsamples', 0)
        for key in all_keys:
            if key not in chunk or chunk[key] is None:
                chunk[key] = np.full(nsamples, np.nan)
                logger.info(f"Field '{key}' missing in file {kk+1} — filling with NaN")
    return chunks


def _process_time_vectors( fast_chunks, file_versions, jump_flags, samp_ints, selected_times) -> tuple:
    """Correct T vectors: handle jumps, zero intervals, v1 vs v2+ format."""
    if not any('T' in c for c in fast_chunks):
        return fast_chunks, jump_flags, samp_ints

    logger.debug("Processing time vectors")
    fdurs  = np.zeros(len(fast_chunks))
    last_T = None

    for kk, chunk in enumerate(fast_chunks):
        T = chunk['T']

        # v>1: convert absolute timestamps → differences
        if file_versions[kk] > 1.0:
            T0     = (T[0] - samp_ints[kk]) if kk == 0 else last_T
            last_T = T[-1]
            T      = np.mod(np.diff(np.concatenate([[T0], T])), 2**32)
            chunk['T'] = T

            if np.any(T < 0):
                logger.warning(f"Negative time step in file {kk+1}")
            if np.any(T > 100 * FPGA_RATE):
                logger.warning(f"Time steps > 100 s detected in file {kk+1}")

        fdurs[kk] = T.sum()

        if kk == 0:
            logger.debug("File %d: duration = %.4f s", kk+1, fdurs[kk] / FPGA_RATE)
        if kk > 0:
            gap = ((chunk['info']['startTime'] - fast_chunks[kk-1]['info']['startTime'])* FPGA_RATE - fdurs[kk-1])
            logger.debug("File %d: duration = %.4f s| gap = %.4f s | jump_flag = %s", kk+1, fdurs[kk] / FPGA_RATE, (gap / FPGA_RATE), jump_flags[kk])

            needs_correction = ((file_versions[kk] <= 1.0 and jump_flags[kk]) or (file_versions[kk] >  1.0 and gap > 100 * FPGA_RATE))
            if needs_correction:
                if gap > 0:
                    T[0] = gap
                    logger.info(f"File {kk+1}: inserted {(gap / FPGA_RATE):.4f}  s gap into T[0]")
                else:
                    T[0] = samp_ints[kk]
                    logger.info(f"File {kk+1}: inserted nominal sampling interval into T[0]")

        # Fix zero intervals (v<=1 only)
        bad = np.where(T[1:] == 0)[0]
        if len(bad) > 0:
            if file_versions[kk] <= 1.0:
                T[bad + 1] = samp_ints[kk]
                logger.info(f"{len(bad)} zero intervals in file {kk+1} — replaced with nominal interval")
            else:
                logger.warning(f"{len(bad)} zero intervals in v{file_versions[kk]} file {kk+1} — duplicated timestamps!")

        chunk['T'] = T

    return fast_chunks, jump_flags, samp_ints


def _check_time_coverage(t: np.ndarray, tstart: float, tstop: float) -> None:
    if t[-1] < tstart:
        logger.warning("Last available data is older than requested interval — returning last file content")
    else:
        if t[0] > tstart:
            logger.info("Requested tstart is earlier than first available data — partial data returned")
        if t[-1] < tstop:
            logger.info("Requested tstop is later than last available data — partial data returned")


def _interpolate(bundle: DataBundle, fast_keys: list, samp_ints: np.ndarray) -> DataBundle:
    """Interpolate fast data to constant sample rate (mode of samp_ints)."""
    dominant = _mode(samp_ints)
    if not np.all(samp_ints == dominant):
        logger.info("Multiple sampling frequencies detected — using fastest for interpolation")
    dt   = float(np.min(samp_ints)) / FPGA_RATE

    t_re = np.arange(bundle.t[0], bundle.t[-1], dt)

    for key in fast_keys:
        if key == 'T':
            continue
        bundle.fast[key] = np.interp(t_re, bundle.t, bundle.fast[key].astype(float))

    bundle.t = t_re
    return bundle


def _trim(bundle: DataBundle, fast_keys: list, tstart: float, tstop: float) -> DataBundle:
    """Trim all data fields to [tstart, tstop]."""

    if fast_keys:
        mask = (bundle.t >= tstart) & (bundle.t <= tstop)
        if not np.any(mask):
            logger.warning("Trim would result in empty data — skipping")
            return bundle

        bundle.t = bundle.t[mask]
        for key in fast_keys:
            if key != 'T':
                bundle.fast[key] = bundle.fast[key][mask]
    else:
        logger.warning("No CAP fast data found - nothing to trim")

    if bundle.slow:
        slow_keys = list(bundle.slow.keys())
        slow_t    = bundle.slow[slow_keys[0]]
        slow_mask = (slow_t >= tstart) & (slow_t <= tstop)
        for key in slow_keys:
            bundle.slow[key] = bundle.slow[key][slow_mask]
    else:
        logger.warning("No CAP slow data found - nothing to trim")

    return bundle