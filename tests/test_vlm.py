"""Tests for VLM engine module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_import():
    from pipeline.vlm_engine import VLMEngine
    assert VLMEngine is not None
