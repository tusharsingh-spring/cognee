"""VLM inference engine using Florence-2-large with async worker queue.

Receives PerceptionPacket from Layer 1 + keyframes.
Task: add visual context that specialized models missed.
Prompts tell the VLM exactly what was already detected so it focuses on gaps.
"""

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
    VLM_MAX_CALLS_PER_MINUTE,
    VLM_MAX_SIZE,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLM_QUEUE_MAXSIZE,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

PERSON_PROMPTS = OrderedDict([
    ("dense", "<MORE_DETAILED_CAPTION>"),
    ("scene", "<DETAILED_CAPTION>"),
    ("od", "<OD>"),
    ("scene_full", "<MORE_DETAILED_CAPTION>"),
])


class VLMEngine:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._task_queue = queue.Queue(maxsize=VLM_QUEUE_MAXSIZE)
        self._result_cache: Dict[str, Dict] = {}
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._call_times: list = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        logger.info(f"[VLM] Loading model: {VLM_MODEL}")
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info("[VLM] Downloading/loading model weights (this may take 30-60s on first run)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                VLM_MODEL, trust_remote_code=True,
            )
            logger.info("[VLM] Model weights loaded, moving to device...")
            self.model = self.model.to(VLM_DEVICE)
            self.model.eval()
            logger.info("[VLM] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(VLM_MODEL, trust_remote_code=True)
            self._loaded = True
            logger.info("[VLM] Model loaded successfully")
        except Exception as e:
            logger.error(f"[VLM] Model load failed: {e}")
            self.model = None
            self.processor = None
            self._loaded = False
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

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

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                track_id, task_type, image, callback, prompt_override = task
                t_start = time.perf_counter()
                result = self._run_inference(image, task_type, prompt_override)
                elapsed = (time.perf_counter() - t_start) * 1000
                cache_key = f"{track_id}:{task_type}"
                self._result_cache[cache_key] = {"result": result, "timestamp": time.time()}
                if callback:
                    callback(result)
                result_preview = result[:120] if result else "(empty)"
                logger.info(f"[VLM] Person_{track_id} {task_type} ready ({elapsed:.0f}ms): {result_preview}...")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[VLM] Worker error: {e}")

    def _run_inference(self, image: np.ndarray, task_type: str, prompt_override: Optional[str] = None) -> str:
        if image is None or image.size == 0:
            return ""

        prober = profiler.get("vlm")
        prober.start()

        h, w = image.shape[:2]
        if max(h, w) > VLM_MAX_SIZE:
            scale = VLM_MAX_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        prompt = prompt_override or PERSON_PROMPTS.get(task_type, task_type)
        task_token = PERSON_PROMPTS.get(task_type, task_type)

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(VLM_DEVICE) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=VLM_MAX_TOKENS,
                num_beams=3,
                do_sample=False,
            )
            if hasattr(generated_ids, "sequences"):
                generated_ids = generated_ids.sequences

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task_token, image_size=(pil_image.width, pil_image.height)
        )
        result = parsed.get(task_token, generated_text) if parsed else generated_text
        if isinstance(result, dict):
            result = str(result)
        result = str(result).strip()

        prober.stop()
        return result

    def submit_task(self, track_id, task_type, image, callback=None, prompt_override=None) -> bool:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60.0]
        if len(self._call_times) >= VLM_MAX_CALLS_PER_MINUTE:
            return False
        self._call_times.append(now)
        try:
            self._task_queue.put_nowait((track_id, task_type, image, callback, prompt_override))
            return True
        except queue.Full:
            return False

    def get_result(self, track_id, task_type: str) -> Optional[str]:
        cache_key = f"{track_id}:{task_type}"
        entry = self._result_cache.get(cache_key)
        if entry and time.time() - entry["timestamp"] < VLM_CACHE_TTL:
            return entry["result"]
        return None

    def get_cache_age(self, track_id, task_type: str) -> float:
        cache_key = f"{track_id}:{task_type}"
        entry = self._result_cache.get(cache_key)
        return time.time() - entry["timestamp"] if entry else float("inf")

    def submit_with_perception(
        self, track_id, crop, perception_context: str, task: str = "dense"
    ) -> bool:
        cached = self.get_result(track_id, task)
        if cached:
            return False

        return self.submit_task(track_id, task, crop)

    def submit_full_scene(self, scene_id: str, frame: np.ndarray, perception_context: str = "") -> bool:
        key = "scene_full"
        cached = self.get_result(scene_id, key)
        if cached:
            return False
        resized = cv2.resize(frame, (VLM_MAX_SIZE, VLM_MAX_SIZE))

        return self.submit_task(scene_id, key, resized)

    def get_person_details(self, track_id) -> Dict[str, str]:
        details = {}
        for key in PERSON_PROMPTS:
            result = self.get_result(track_id, key)
            if result:
                details[key] = result
        return details

    def get_combined_result(self, track_id) -> Dict[str, str]:
        results = {}
        for key in ["dense", "scene", "od"]:
            r = self.get_result(track_id, key)
            if r:
                results[key] = r
        return results

    def purge_cache(self) -> None:
        now = time.time()
        expired = [k for k, v in self._result_cache.items()
                    if now - v["timestamp"] > VLM_CACHE_TTL]
        for k in expired:
            del self._result_cache[k]
