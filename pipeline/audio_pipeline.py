"""Audio pipeline: Whisper transcription + optional pyannote diarization."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.settings import (
    AUDIO_DEVICE,
    AUDIO_ENABLED,
    AUDIO_SAMPLE_RATE,
    AUDIO_WHISPER_MODEL,
    DATA_DIR,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioPipeline:
    def __init__(self) -> None:
        self._enabled = AUDIO_ENABLED
        self._model_size = AUDIO_WHISPER_MODEL
        self._device = AUDIO_DEVICE
        self._sample_rate = AUDIO_SAMPLE_RATE
        self._model = None
        self._loaded = False
        self._transcripts: List[Dict] = []

        if self._enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import whisper

            logger.info(f"[Audio] Loading Whisper {self._model_size}...")
            self._model = whisper.load_model(self._model_size, device=self._device)
            self._loaded = True
            logger.info("[Audio] Whisper model loaded")
        except ImportError:
            logger.warning("[Audio] openai-whisper not installed. Run: pip install openai-whisper")
            self._loaded = False
        except Exception as e:
            logger.warning(f"[Audio] Load failed: {e}")
            self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        if output_path is None:
            output_path = str(DATA_DIR / f"audio_{int(time.time())}.wav")

        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self._sample_rate),
                "-ac", "1", output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning(f"[Audio] ffmpeg failed: {result.stderr[:200]}")
                return None
            logger.info(f"[Audio] Extracted audio to {output_path}")
            return output_path
        except FileNotFoundError:
            logger.warning("[Audio] ffmpeg not found. Audio extraction requires ffmpeg.")
            return None
        except Exception as e:
            logger.warning(f"[Audio] Extraction failed: {e}")
            return None

    def transcribe(self, audio_path: str) -> List[Dict]:
        if not self.is_ready:
            return []

        logger.info(f"[Audio] Transcribing: {audio_path}")
        try:
            result = self._model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                verbose=False,
            )
            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": round(seg.get("start", 0), 2),
                    "end": round(seg.get("end", 0), 2),
                    "text": seg.get("text", "").strip(),
                    "confidence": round(seg.get("confidence", 0), 2) if seg.get("confidence") else None,
                })

            self._transcripts.extend(segments)
            logger.info(f"[Audio] Transcription complete: {len(segments)} segments, {len(result.get('text', ''))} chars")
            return segments
        except Exception as e:
            logger.warning(f"[Audio] Transcription failed: {e}")
            return []

    def transcribe_from_video(self, video_path: str) -> List[Dict]:
        audio_path = self.extract_audio(video_path)
        if audio_path is None:
            return []
        segments = self.transcribe(audio_path)
        try:
            os.remove(audio_path)
        except OSError:
            pass
        return segments

    def get_full_text(self) -> str:
        return " ".join(s["text"] for s in sorted(self._transcripts, key=lambda x: x["start"]))

    def search(self, query: str) -> List[Dict]:
        q = query.lower()
        return [s for s in self._transcripts if q in s["text"].lower()]

    def get_segment_at_time(self, timestamp: float) -> Optional[Dict]:
        for seg in self._transcripts:
            if seg["start"] <= timestamp <= seg["end"]:
                return seg
        return None

    def export(self, output_path: Optional[str] = None) -> str:
        path = output_path or str(DATA_DIR / f"transcript_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump({
                "full_text": self.get_full_text(),
                "segments": self._transcripts,
            }, f, indent=2)
        logger.info(f"[Audio] Exported transcript to {path}")
        return path

    def get_transcripts(self) -> List[Dict]:
        return list(self._transcripts)
