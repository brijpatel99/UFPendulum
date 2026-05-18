import logging
import sys

def setup_logging(level: int = logging.DEBUG, log_file: str = None):
    """
    Call once in main.py to configure logging for the entire pipeline.
    
    Args:
        level    : logging.DEBUG / INFO / WARNING / ERROR / CRITICAL
        log_file : optional path to also write logs to a file
    """
    fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers
    )