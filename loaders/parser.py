import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps type strings in file header -> (numpy dtype, byte count)
TYPE_MAP = {
    'I16': ('>i2', 2),   # big-endian int16
    'U32': ('>u4', 4),   # big-endian uint32
    'DBL': ('>f8', 8),   # big-endian float64
}


def parsedata(filename: str, slow_only: bool = False) -> dict:
    """
    Load a UF 4-mass pendulum binary file.

    File format:
        - 4-byte big-endian int   : header length
        - <headerLen> ASCII bytes : tab-separated header
        - remaining bytes         : interleaved column data (big-endian)

    Header fields (tab-separated):
        start_time | COL1 (TYPE) | COL2 (TYPE) | ... | vN

    Args:
        filename  : path to .bin file (extension added if omitted)
        slow_only : skip fast binary data, only load paired .txt slow data

    Returns:
        dict with keys:
            'info'     : startTime, nsamples, headers, version, firstfile
            '<col>'    : np.ndarray per fast data column
            'slowData' : dict of slow data arrays (if .txt sidecar exists)
    """
    path = Path(filename)
    if path.suffix != '.bin':
        path = path.with_suffix('.bin')

    first_file = path.stem[-1] == 'e'   # filename ends with 'e' before extension
    out = {}

    # ------------------------------------------------------------------ #
    # Fast data
    # ------------------------------------------------------------------ #
    if not slow_only:
        logger.debug("Opening: %s", path)
        raw_bytes = path.read_bytes()

        # Parse header
        header_len = int(np.frombuffer(raw_bytes[:4], dtype='>i4')[0])
        header_str = raw_bytes[4 : 4 + header_len].decode('ascii')
        data_bytes  = raw_bytes[4 + header_len:]
        logger.debug("Header length: %d bytes", header_len)

        # Header tokens: [start_time, COL, TYPE, COL, TYPE, ..., vN]
        tokens     = re.split(r'\t|\(|\)', header_str)
        tokens     = [t.strip() for t in tokens if t.strip()]
        start_time = float(tokens[0])
        version    = float(tokens[-1][1:])  # strip leading 'v'

        # Pair up column names and types (tokens[1:-1] alternates name/type)
        col_names_raw = tokens[1:-1:2]
        col_types_raw = tokens[2:-1:2]
        logger.debug("Columns: %s", list(zip(col_names_raw, col_types_raw)))

        col_info = _parse_column_types(col_names_raw, col_types_raw)

        # Read & reshape data
        row_len   = sum(c['byte_count'] for c in col_info)
        remainder = len(data_bytes) % row_len
        if remainder:
            logger.warning("Non-integer number of rows — discarding %d trailing bytes", remainder)
            data_bytes = data_bytes[:-remainder]

        data_array = np.frombuffer(data_bytes, dtype=np.uint8)
        nsamples   = len(data_array) // row_len
        data_2d    = data_array.reshape(nsamples, row_len)   # (nsamples, bytes_per_row)
        logger.debug("Loaded %d samples, %d column(s)", nsamples, len(col_info))

        out['info'] = {
            'startTime': start_time,
            'nsamples':  nsamples,
            'headers':   [c['name'] for c in col_info],
            'version':   version,
            'firstfile': first_file,
        }

        # Extract and typecast each column
        col_headers = _make_valid_headers([c['name'] for c in col_info])
        offset = 0
        for col, header in zip(col_info, col_headers):
            col_bytes = data_2d[:, offset : offset + col['byte_count']]
            out[header] = _typecast_column(col_bytes, col['dtype'], col['byte_count'])
            offset += col['byte_count']

    # ------------------------------------------------------------------ #
    # Slow data sidecar (.txt)
    # ------------------------------------------------------------------ #
    slow_path = path.with_suffix('.txt')
    if slow_path.exists():
        logger.debug("Loading slow data: %s", slow_path)
        out['slowData'] = _load_slow_data(slow_path)
    else:
        logger.warning("No slow data file found for: %s", path.name)

    return out


# ======================================================================= #
# Private helpers
# ======================================================================= #

def _parse_column_types(names: list[str], types: list[str]) -> list[dict]:
    """Map raw header type strings to numpy dtypes and byte counts."""
    col_info = []
    for name, type_str in zip(names, types):
        key = type_str.strip().upper()
        if key not in TYPE_MAP:
            raise ValueError(f"Unknown column type '{type_str}' for column '{name}'")
        dtype, byte_count = TYPE_MAP[key]
        col_info.append({'name': name, 'dtype': dtype, 'byte_count': byte_count})
    return col_info


def _typecast_column(col_bytes: np.ndarray, dtype: str, byte_count: int) -> np.ndarray:
    """
    Reconstruct typed values from raw big-endian bytes.
    col_bytes shape: (nsamples, byte_count)
    """
    # Pack bytes back into a contiguous buffer and reinterpret
    contiguous = np.ascontiguousarray(col_bytes).tobytes()
    return np.frombuffer(contiguous, dtype=dtype).copy()


def _make_valid_headers(headers: list[str]) -> list[str]:
    """
    Sanitise column names to be valid Python identifiers.
    Replaces spaces with underscores, deduplicates with trailing underscore.
    """
    cleaned   = [re.sub(r'\s+', '_', h) for h in headers]
    validated = []
    seen      = set()
    adjusted  = False

    for name in cleaned:
        # strip any non-identifier characters
        name = re.sub(r'[^\w]', '_', name)
        if not name[0].isalpha() and name[0] != '_':
            name = '_' + name
        # deduplicate
        original = name
        while name in seen:
            name    += '_'
            adjusted = True
        seen.add(name)
        validated.append(name)

    if adjusted:
        logger.warning("Duplicate column names found — underscore(s) appended to duplicates")

    return validated


def _load_slow_data(path: Path) -> dict:
    """Load tab-separated slow data sidecar file into a dict of arrays."""
    df = pd.read_csv(path, sep='\t', header=0)

    # Strip trailing underscores from column names (MATLAB artifact)
    df.columns = [c.rstrip('_') for c in df.columns]

    return {col: df[col].to_numpy() for col in df.columns}