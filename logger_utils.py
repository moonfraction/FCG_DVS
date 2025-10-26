"""
Logger utility for FCG project.
Creates a log file in /res with timestamped filename and returns a configured logger.
"""
import logging
import os
import datetime


def setup_logger(name: str = 'fcg', log_dir: str = 'res'):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(log_dir, f'file_{ts}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers when re-importing
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Stream handler (console)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    # attach filename for callers that may want to display it
    try:
        logger._logfile = filename
    except Exception:
        pass

    logger.info(f'Logger initialized. Writing to {filename}')
    return logger


# Initialize default logger on import
LOGGER = setup_logger()


def get_logger():
    return LOGGER
