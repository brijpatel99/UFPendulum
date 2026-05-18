import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def loadIFO(filepath: str, tstart: float, tstop: float, trim: bool = False,
            interp_param = None, tcorr: bool = False) -> np.ndarray:
    """
    Load and stitch IFO data files covering [tstart, tstop].

    Files are plain tab/space separated .txt with pattern *IFODataFile*.txt.
    Filename starts with a 12-digit millisecond timestamp.

    Columns in each file: [time_s, phi, ch2, ch3]
    Only time and phi (col 0, col 1) are used downstream.

    Args:
        filepath : folder containing IFO .txt files
        tstart   : interval start (pendulum time, seconds)
        tstop    : interval end (pendulum time, seconds)
        trim     : trim output to exactly [tstart, tstop]
        interp_param  : target sample rate / time vector for interpolation onto uniform grid.
                        - None      : no interpolation
                        - 0         : use modal fs of original data
                        - float     : specifies sample rate (fs) to use
                        - np.ndarray: specifies uniform time vector to use
        tcorr    : apply time vector correction for known DAQ bug

    Returns:
        np.ndarray shape (N, 4) — columns: [time, phi, ch2, ch3]
        or (N, 2) if only time+phi retained after interp
    """
    # ------------------------------------------------------------------ #
    # Check and standardize inputs
    # ------------------------------------------------------------------ #
    filepath = Path(filepath.rstrip('/\\'))

    if tstart > tstop:
        logger.warning("tstart > tstop — swapping values")
        tstart, tstop = tstop, tstart

    # ------------------------------------------------------------------ #
    # Select files
    # ------------------------------------------------------------------ #
    logger.debug("Selecting IFO files to load")
    filenames, stimes = _get_sorted_ifo_files(filepath)
    stimes = np.array(stimes)

    selected_files, _ = _select_ifo_files(filenames, stimes, tstart, tstop)
    logger.info("Selected %d IFO file(s) to load", len(selected_files))
    logger.info("Files: %s", selected_files)

    # ------------------------------------------------------------------ #
    # Load files
    # ------------------------------------------------------------------ #
    chunks = []
    for fname in selected_files:
        logger.debug("Loading: %s", fname)
        chunk = _load_ifo_txt(filepath / fname)
        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)
        else:
            logger.warning("Empty or unreadable file: %s", fname)

    if not chunks:
        raise ValueError("No data loaded for IFO interval [%.3f, %.3f]" % (tstart, tstop))

    # ------------------------------------------------------------------ #
    # Time correction (tcorr)
    # ------------------------------------------------------------------ #
    if tcorr:
        logger.info("Applying time vector correction (tcorr)")
        chunks = _apply_tcorr(chunks)

    # ------------------------------------------------------------------ #
    # Concatenate
    # ------------------------------------------------------------------ #
    data = np.vstack(chunks)
    logger.debug("Concatenated %d rows", len(data))

    # ------------------------------------------------------------------ #
    # Interpolate
    # ------------------------------------------------------------------ #
    if interp_param or interp_param==0:
        logger.info("Interpolating IFO data")
        data = _interpolate_ifo(data, interp_param)

    # ------------------------------------------------------------------ #
    # Trim
    # ------------------------------------------------------------------ #
    if trim:
        mask = (data[:, 0] >= tstart) & (data[:, 0] <= tstop)
        if not np.any(mask):
            logger.warning("Trim would result in empty data — skipping")
        else:
            data = data[mask]
            logger.info("IFO data trimmed to range [%d, %d]", data[0,0], data[-1,0])

    logger.info("loadIFO complete — %d samples", len(data))
    return data

# ======================================================================= #
# Private helpers
# ======================================================================= #

def _get_sorted_ifo_files(filepath: Path) -> tuple[list[str], list[float]]:
    """
    Find *IFODataFile*.txt files and sort by timestamp in filename.
    """
    logger.debug("Scanning for ifo files in: %s", filepath.__str__())
    all_files  = [f.name for f in filepath.iterdir() if f.is_file()]
    ifo_files  = [f for f in all_files if 'IFODataFile' in f and f.endswith('.txt')]
    logger.debug("Found %d matching ifo files", len(ifo_files))
    
    if not ifo_files:
        raise FileNotFoundError("No IFODataFile*.txt files found in %s" % filepath)

    stimes = []
    for fname in ifo_files:
        try:
            stimes.append(float(fname[:12]) / 1000.0) #first 12 char is the time
        except ValueError:
            logger.warning("Could not parse timestamp from filename: %s", fname)
            stimes.append(0.0)

    sorted_pairs = sorted(zip(stimes, ifo_files))
    stimes, filenames = zip(*sorted_pairs)

    logger.debug("Sorted %d files by start time (%.3f s to %.3f s)", len(filenames), stimes[0] if stimes else 0, stimes[-1] if stimes else 0)#→

    return list(filenames), list(stimes)


def _select_ifo_files(filenames: list[str], stimes: np.ndarray, tstart: float, tstop: float) -> tuple[list[str], np.ndarray]:
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
        raise ValueError("No IFO files found overlapping [%.3f, %.3f]. Available range: [%.3f, %.3f]" % (tstart, tstop, stimes[0], stimes[-1]))
    else:
        mask[np.argmax(mask)-1] = True
    
    selected_files = [filenames[i] for i in range(n) if mask[i]]
    selected_times = stimes[mask]

    logger.info("Selected %d file(s) covering [%.3f, %.3f]", len(selected_files), selected_times[0], selected_times[-1])
    logger.debug("Selected files: %s", selected_files)
 
    return selected_files, selected_times


def _load_ifo_txt(path: Path) -> np.ndarray | None:
    """Load a single IFO text file into a numpy array."""
    try:
        return np.loadtxt(path)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return None


def _apply_tcorr(chunks: list[np.ndarray]) -> list[np.ndarray]:
    """
    Fix corrupted initial time vectors (known DAQ bug from 2016-11-21).

    Reconstructs the time vector of each chunk by linearly spacing from
    the inferred end of the previous chunk to the last timestamp of the
    current chunk.
    """
    last_time = 0.0
    corrected = []

    for chunk in chunks:
        if len(chunk) == 0:
            corrected.append(chunk)
            continue

        if last_time == 0.0:
            # Estimate true start from end time and mean interval of stable tail
            mean_dt  = float(np.mean(np.diff(chunk[999:, 0])))
            last_time = chunk[-1, 0] - len(chunk) * mean_dt

        n    = len(chunk)
        t    = last_time + np.arange(1, n + 1) * (chunk[-1, 0] - last_time) / n
        fixed = np.column_stack([t, chunk[:, 1:]])
        corrected.append(fixed)
        last_time = t[-1]

    return corrected


def _interpolate_ifo(data: np.ndarray, interp_param = None) -> np.ndarray:
    """
    Interpolate IFO data onto a uniform time grid.
    Removes non-monotonic timestamps and all-zero rows before interpolating.

    Args:
        data          : (N, 4) array with time in column 0
        interp_param  : target sample rate / time vector for interpolation.
                        - None      : no interpolation
                        - 0.0         : use modal fs of original data
                        - float     : specifies sample rate (fs) to use
                        - np.ndarray: specifies uniform time vector to use

    Returns:
        (M, 4) interpolated array
    """
    t_orig = data[:, 0]
    fs = None
    # Check type of interp parameter
    if interp_param == 0:
        diffs = np.diff(t_orig)
        vals, counts = np.unique(np.round(diffs, 6), return_counts=True)
        fs = 1.0 / float(vals[np.argmax(counts)])
        logger.debug("fs not specified — using modal fs: %.6f Hz", fs)
        pass

    elif isinstance(interp_param, float) or isinstance(interp_param, int):
        fs = interp_param
        mean_fs_orig = 1.0 / float(np.mean(np.diff(t_orig)))
        if fs < mean_fs_orig * 0.99:
            logger.warning("Requested fs (%.4f Hz) is lower than original (%.4f Hz) — aliasing will occur!", fs, mean_fs_orig)

    elif isinstance(interp_param, np.ndarray):
        t_new = interp_param
        fs = 1 / ((max(t_new) - min(t_new)) / len(t_new)) # this seems wrong but its how its done in MATLAB - need to check!
        logger.debug("Using given time vector to interpolate ifo corresponding to %.3f sample rate", fs)

    else:
        logger.info("No interpolation parameter provided for IFO data")

    # Build target time vector
    n_samples = int(np.floor((t_orig[-1] - t_orig[0])*fs)) + 1
    t_new = t_orig[0] + np.arange(n_samples) / fs

    # Remove bad points: non-monotonic time or all-zero data row
    good   = np.ones(len(data), dtype=bool)
    t_max  = 0
    for i in range(len(data)):
        if data[i, 0] <= t_max or data[i, 1] == 0:
            good[i] = False
        else:
            t_max = data[i, 0]

    n_removed = int(np.sum(~good))
    if n_removed:
        logger.info("Removed %d bad points (%.2f%%) before interpolation", n_removed, 100.0 * n_removed / len(data))

    data_clean = data[good]
    interp_cols = [np.interp(t_new, data_clean[:, 0], data_clean[:, col]) for col in range(1, data.shape[1])]

    return np.column_stack([t_new] + interp_cols)