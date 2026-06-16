"""Tests for motion capture module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_import():
    from pipeline.capture import MotionCapture
    assert MotionCapture is not None


def test_mog2_creation():
    import cv2
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
    assert bg is not None
