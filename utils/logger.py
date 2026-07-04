import logging
import sys
from pathlib import Path

from config.settings import LOG_FILE, LOG_FORMAT, LOG_LEVEL

_loggers: dict = {}


def get_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            pass

    _loggers[name] = logger
    return logger
