"""
ao_handler/add_ptime.py

Adds pendulum time ao to any dict of aos
"""

import logging
import numpy as np
from ltpda.ao import ao

logger = logging.getLogger(__name__)


# ======================================================================= #
# Public API
# ======================================================================= #
def add_pendTime_ao(data_aos: dict[str, ao], start_time: float, match: str='PHI') -> ao:
    """
    Build a pendulum time ao matching the time grid of the corresponding 'match' ao in data_aos
    
    Args:
        data_aos   : dict aos we want to append with pendulum time
        start_time : pendulum time of the first sample (seconds)
        match      : time grid we want to match with (default='PHI')
    
    Returns:
        appended aos with pendulum time in seconds
    """
    result = {}
    
    if match in data_aos:
        n      = len(data_aos[match].ydata())
        fs     = data_aos[match].fs
        t0     = data_aos[match].t0
        t_pend = start_time + np.arange(n) / fs
        result['ptime'] = ao(yvals=t_pend, fs=fs, yunits='s', name='Pendulum time', type='tsdata')
        result['ptime'].t0 = t0
        logger.info("Adding ptime to given dict of aos by matching time grid of %s", match)
    else:
        logger.warning('Unable to create pendulum time ao: %s does not exist in given aos', match)

    for name, ao_temp in data_aos.items():
        result[name] = ao_temp
    return result
