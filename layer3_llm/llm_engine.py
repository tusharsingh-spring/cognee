"""Local LLM engine using Qwen2.5-3B GGUF via llama-cpp-python.

Falls back to Groq cloud API if local model unavailable.
Does ALL the reasoning: narrative, intent, anomaly, notify decision.
"""

import json
import os
import time
from typing import Dict, List, Optional

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_CONTEXT_LENGTH,
    LLM_ENABLED,
    LLM_GPU_LAYERS,
    LLM_MODEL_PATH,
    LLM_THREADS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the reasoning brain of a CCTV surveillance system. Output ONLY valid JSON, no markdown.

Given structured perception data, build a narrative, infer intent, rate anomaly (0.0-1.0), and decide if notification is needed."""




class LLMEngine:
    def __init__(self) -> None:
        self.enabled = LLM_ENABLED
        self._model = None
        self._local_available = False
        self._groq_available = bool(GROQ_API_KEY)
        self._groq_client = None
        self._history: List[Dict] = []

        if self.enabled:
            self._load()

    def _load(self) -> None:
        if self._try_local():
            return
        if self._try_groq():
            return
        logger.warning("[LLM] No LLM backends available — reasoning disabled")
        self.enabled = False

    def _try_local(self) -> bool:
        try:
            from llama_cpp import Llama

            model_path = LLM_MODEL_PATH
            if not model_path:
                candidates = [
                    "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                    "qwen2.5-3b-instruct-q4_k_m.gguf",
                    "llama-3.2-3b-instruct-q4_k_m.gguf",
                ]
                from config.settings import MODEL_DIR
                for cand in candidates:
                    p = MODEL_DIR / cand
                    if p.is_file():
                        model_path = str(p)
                        break

            if model_path and os.path.isfile(model_path):
                self._model = Llama(
                    model_path=model_path,
                    n_ctx=LLM_CONTEXT_LENGTH,
                    n_threads=LLM_THREADS,
                    n_gpu_layers=LLM_GPU_LAYERS,
                    verbose=False,
                )
                self._local_available = True
                logger.info(f"[LLM] Local model loaded: {model_path}")
                return True
            else:
                logger.info("[LLM] No local model found. Put a GGUF file in data/models/")
                return False
        except ImportError:
            logger.info("[LLM] llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.warning(f"[LLM] Local model failed: {e}")
            return False

    def _try_groq(self) -> bool:
        if not GROQ_API_KEY:
            return False
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info(f"[LLM] Groq API ready (model={GROQ_MODEL})")
            return True
        except ImportError:
            logger.info("[LLM] groq package not installed. Run: pip install groq")
            return False
        except Exception as e:
            logger.warning(f"[LLM] Groq init failed: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        if self._local_available and self._model is not None:
            try:
                result = self._model(prompt, max_tokens=max_tokens, temperature=temperature,
                                     stop=["</s>", "<|im_end|>", "<|endoftext|>"])
                return result["choices"][0]["text"].strip()
            except Exception as e:
                logger.error(f"[LLM] Local inference error: {e}")

        if self._groq_client is not None:
            for attempt in range(4):
                try:
                    response = self._groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    err_str = str(e)
                    if "rate_limit" in err_str.lower() or "429" in err_str:
                        delay = min(30, 2 ** attempt)
                        logger.warning(f"[LLM] Rate limited, retrying in {delay}s (attempt {attempt + 1}/4)")
                        time.sleep(delay)
                    else:
                        logger.error(f"[LLM] Groq error: {e}")
                        break

        return "{}"

    def reason(
        self,
        perception_context: str,
        vlm_context: str,
        short_term: str,
        medium_term: str,
        long_term: str,
        identity_profiles: str,
        frame_time: float = 0.0,
    ) -> dict:
        """Full reasoning: takes all context, returns structured output."""
        if not self.enabled:
            return self._empty_reasoning()

        if frame_time > 0:
            ts = time.localtime(frame_time)
            time_str = time.strftime('%H:%M:%S', ts)
            day_str = time.strftime('%A', ts)
        else:
            time_str = time.strftime('%H:%M:%S')
            day_str = time.strftime('%A')

        prompt = f"""{SYSTEM_PROMPT}

Time: {time_str} ({day_str})

PERCEPTION:
{perception_context[:600]}

VLM CONTEXT:
{vlm_context[:300]}

SHORT_TERM:
{short_term[:400]}

IDENTITY:
{identity_profiles[:300]}

Return JSON:
{{
  "narrative": "...",
  "intent": "...",
  "is_normal": true,
  "anomaly_score": 0.1,
  "notify": false,
  "urgency": "none",
  "notification_text": null
}}
"""

        raw = self.generate(prompt, max_tokens=512)
        return self._parse_response(raw)

    def answer_question(self, question: str, context: str) -> str:
        if not self.enabled:
            return "LLM not available. Install llama-cpp-python or set GROQ_API_KEY."

        prompt = f"""{SYSTEM_PROMPT}

Available context from CCTV data and knowledge graph:
{context}

User question: {question}

Answer the question using ONLY the context provided. Be concise (2-4 sentences).
If the context doesn't contain the answer, say so clearly."""
        return self.generate(prompt, max_tokens=512, temperature=0.3)

    def _parse_response(self, raw: str) -> dict:
        try:
            if "{" in raw and "}" in raw:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return self._empty_reasoning()

    def _empty_reasoning(self) -> dict:
        return {
            "narrative": "LLM reasoning unavailable",
            "intent": "",
            "is_normal": True,
            "anomaly_score": 0.0,
            "notify": False,
            "urgency": "none",
            "notification_text": None,
        }

    @property
    def available(self) -> bool:
        return self.enabled and (self._local_available or self._groq_available)
