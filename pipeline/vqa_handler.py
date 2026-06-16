"""VQA (Video Question Answering) handler."""

from typing import Dict, List, Optional

import numpy as np

from config.settings import VQA_DEFAULT_QUESTIONS
from utils.logger import get_logger

logger = get_logger(__name__)


class VQAHandler:
    def __init__(self, vlm_engine) -> None:
        self.vlm = vlm_engine
        self._vqa_history: Dict[int, List[Dict]] = {}

    def ask_questions(
        self, track_id: int, crop: np.ndarray
    ) -> List[Dict]:
        results = []
        if track_id not in self._vqa_history:
            self._vqa_history[track_id] = []

        for question in VQA_DEFAULT_QUESTIONS:
            task = f"<VQA>{question}"
            cached = self.vlm.get_result(track_id, task)
            if cached:
                results.append({"question": question, "answer": cached, "cached": True})
                continue

            self.vlm.submit_task(track_id, task, crop)

        return results

    def get_all_answers(self, track_id: int) -> List[Dict]:
        answers = []
        if track_id not in self._vqa_history:
            return answers
        for entry in self._vqa_history[track_id]:
            q = entry["question"]
            task = f"<VQA>{q}"
            cached = self.vlm.get_result(track_id, task)
            if cached:
                answers.append({"question": q, "answer": cached})
        return answers
