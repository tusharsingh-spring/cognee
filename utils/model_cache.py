"""Model download and cache management."""

import os
from pathlib import Path
from typing import Optional

from config.settings import MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: dict = {}


def ensure_model(primary_path: str, download_url: Optional[str] = None) -> Optional[Path]:
    if primary_path in _MODEL_CACHE:
        return _MODEL_CACHE[primary_path]

    path = Path(primary_path)
    if path.is_file():
        _MODEL_CACHE[primary_path] = path
        return path

    path = MODEL_DIR / path.name
    if path.is_file():
        _MODEL_CACHE[primary_path] = path
        return path

    if download_url:
        try:
            logger.info(f"Downloading model: {path.name}")
            _download_file(download_url, path)
            if path.is_file():
                _MODEL_CACHE[primary_path] = path
                return path
        except Exception as e:
            logger.warning(f"Model download failed: {e}")

    return None


def _download_file(url: str, dest: Path) -> None:
    import requests

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def get_available_models() -> list:
    models = []
    for f in MODEL_DIR.glob("*"):
        if f.is_file():
            models.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
            })
    return models
