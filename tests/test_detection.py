"""Tests for person detection module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_import():
    from pipeline.detection import PersonDetector
    assert PersonDetector is not None
