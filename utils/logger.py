"""Structured logging for ARGUS."""

import logging
import sys
from pathlib import Path

from config.settings import LOG_FILE, LOG_FORMAT, LOG_LEVEL

_logger_cache = {}

_formatter = logging.Formatter(LOG_FORMAT)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

_file_handler = None


def _get_file_handler():
    global _file_handler
    if _file_handler is None:
        _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        _file_handler.setFormatter(_formatter)
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    logger.addHandler(_stream_handler)
    try:
        logger.addHandler(_get_file_handler())
    except Exception:
        pass

    _logger_cache[name] = logger
    return logger
