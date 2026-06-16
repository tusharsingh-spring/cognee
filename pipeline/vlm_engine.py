"""VLM inference engine using Florence-2 with async queue.
Uses a single combined dense prompt per person for minimal latency."""

import queue
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from config.settings import (
    VLM_CACHE_TTL,
    VLM_DEVICE,
    VLM_MAX_SIZE,
    VLM_MODEL,
    VLM_QUEUE_MAXSIZE,
    VLM_MAX_CALLS_PER_MINUTE,
    VLM_TASK_MAP,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

VLMResult = Dict[str, Any]

PERSON_PROMPTS = OrderedDict([
    ("dense", "<MORE_DETAILED_CAPTION>"),
    ("scene", "<MORE_DETAILED_CAPTION>"),
])


class VLMEngine:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._task_queue: queue.Queue = queue.Queue(maxsize=VLM_QUEUE_MAXSIZE)
        self._result_cache: Dict[str, VLMResult] = {}
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._call_times: list = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(f"[VLM] Loading model: {VLM_MODEL}")
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL, trust_remote_code=True, attn_implementation="eager"
        ).to(VLM_DEVICE)
        self.model = self.model.float()
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            VLM_MODEL, trust_remote_code=True
        )
        self._loaded = True
        logger.info("[VLM] Model loaded successfully")

    def start_worker(self) -> None:
        if self._running:
            return
        if not self._loaded:
            self.load()
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        logger.info("[VLM] Async worker started")

    def stop_worker(self) -> None:
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        logger.info("[VLM] Worker stopped")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                track_id, task_type, image, callback = task
                t_start = time.perf_counter()
                result = self._run_inference(image, task_type)
                elapsed = (time.perf_counter() - t_start) * 1000
                cache_key = f"{track_id}:{task_type}"
                self._result_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time(),
                }
                if callback:
                    callback(result)
                logger.info(f"[VLM] Person_{track_id} caption ready ({elapsed:.0f}ms, {len(result)} chars):")
                logger.info(f"[VLM]   {result}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[VLM] Worker error: {e}")

    def _run_inference(
        self, image: np.ndarray, task_type: str
    ) -> str:
        if image is None or image.size == 0:
            return ""

        prober = profiler.get("vlm")
        prober.start()

        h, w = image.shape[:2]
        max_dim = VLM_MAX_SIZE
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        prompt = PERSON_PROMPTS.get(task_type) or VLM_TASK_MAP.get(task_type, task_type)

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(VLM_DEVICE) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                return_dict_in_generate=False,
            )
            if hasattr(generated_ids, "sequences"):
                generated_ids = generated_ids.sequences

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(pil_image.width, pil_image.height)
        )
        result = parsed.get(prompt, generated_text) if parsed else generated_text
        if isinstance(result, str):
            result = result.strip()
        elif isinstance(result, dict):
            result = str(result)

        elapsed = (time.perf_counter() - prober._start) * 1000 if prober._start else 0
        prober.stop()
        logger.debug(f"[VLM] Inference: {elapsed:.0f}ms")

        return result if result else ""

    def submit_task(
        self,
        track_id: int,
        task_type: str,
        image: np.ndarray,
        callback: Optional[callable] = None,
    ) -> bool:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60.0]
        if len(self._call_times) >= VLM_MAX_CALLS_PER_MINUTE:
            logger.debug("[VLM] Rate limit reached, dropping task")
            return False

        self._call_times.append(now)

        try:
            self._task_queue.put_nowait((track_id, task_type, image, callback))
            logger.debug(f"[VLM] Task submitted: {task_type} for Person_{track_id}")
            return True
        except queue.Full:
            logger.debug("[VLM] Queue full, dropping task")
            return False

    def get_result(self, track_id: int, task_type: str) -> Optional[str]:
        cache_key = f"{track_id}:{task_type}"
        entry = self._result_cache.get(cache_key)
        if entry and time.time() - entry["timestamp"] < VLM_CACHE_TTL:
            return entry["result"]
        return None

    def get_cache_age(self, track_id: int, task_type: str) -> float:
        cache_key = f"{track_id}:{task_type}"
        entry = self._result_cache.get(cache_key)
        if entry:
            return time.time() - entry["timestamp"]
        return float("inf")

    def purge_cache(self) -> None:
        now = time.time()
        expired = [
            k for k, v in self._result_cache.items()
            if now - v["timestamp"] > VLM_CACHE_TTL
        ]
        for k in expired:
            del self._result_cache[k]

    def get_person_details(self, track_id: int) -> Dict[str, str]:
        details = {}
        for key in PERSON_PROMPTS:
            result = self.get_result(track_id, key)
            if result:
                details[key] = result
        return details

    def submit_person_dense(self, track_id: int, image: np.ndarray) -> bool:
        key = "dense"
        cached = self.get_result(track_id, key)
        if cached:
            return False
        return self.submit_task(track_id, key, image)

    def submit_scene_dense(
        self, track_id: int, full_frame: np.ndarray, bbox: tuple, objects: list, task: str = "dense"
    ) -> bool:
        key = task
        cached = self.get_result(track_id, key)
        if cached:
            return False
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(full_frame.shape[1], x2 + 10)
        y2 = min(full_frame.shape[0], y2 + 10)
        crop = full_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        return self.submit_task(track_id, key, crop)

    def submit_person_prompts(self, track_id: int, image: np.ndarray) -> int:
        return int(self.submit_person_dense(track_id, image))
