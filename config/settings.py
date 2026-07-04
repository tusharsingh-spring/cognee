"""ARGUS V2 — Central configuration with typed defaults and env overrides.

Supports BOTH the new 3-layer architecture AND legacy pipeline modules.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
DB_DIR = DATA_DIR / "db"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = DATA_DIR / "logs"
SESSION_DIR = DATA_DIR / "sessions"
VIDEOS_DIR = DATA_DIR / "videos"
CAUSAL_DIR = DATA_DIR / "causal_outputs"
INPUT_VID_DIR = ROOT_DIR / "input_vid"

for d in [DATA_DIR, MODEL_DIR, DB_DIR, CHROMA_DIR, LOG_DIR, SESSION_DIR, VIDEOS_DIR, CAUSAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_FORMATS: Set[str] = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}

# ── Camera ──
CAMERA_URL: str = os.getenv("ARGUS_CAMERA_URL", "0")
CAMERA_INDEX: int = int(os.getenv("ARGUS_CAMERA_INDEX", "0"))
CAMERA_WIDTH: int = int(os.getenv("ARGUS_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT: int = int(os.getenv("ARGUS_CAMERA_HEIGHT", "720"))
CAMERA_FPS: int = int(os.getenv("ARGUS_CAMERA_FPS", "30"))

# ── Video File / Motion Capture ──
VIDEO_FILE: Optional[str] = os.getenv("ARGUS_VIDEO_FILE") or None
MOTION_FRAME_SKIP: int = int(os.getenv("ARGUS_MOTION_FRAME_SKIP", "0"))
MOTION_HISTORY: int = int(os.getenv("ARGUS_MOTION_HISTORY", "500"))
MOTION_LEARNING_RATE: float = float(os.getenv("ARGUS_MOTION_LR", "0.005"))
MOTION_MIN_AREA: int = int(os.getenv("ARGUS_MOTION_MIN_AREA", "2500"))
MOTION_THRESHOLD: int = int(os.getenv("ARGUS_MOTION_THRESHOLD", "25"))

# ── Gating (YOLOv8n) ──
GATE_ENABLED: bool = os.getenv("ARGUS_GATE_ENABLED", "true").lower() == "true"
GATE_MODEL: str = os.getenv("ARGUS_GATE_MODEL", "yolov8n.pt")
GATE_IMG_SIZE: int = int(os.getenv("ARGUS_GATE_IMG_SIZE", "320"))
GATE_CONFIDENCE: float = float(os.getenv("ARGUS_GATE_CONFIDENCE", "0.3"))
GATE_PERSON_CLASS: int = 0
GATE_NEW_PERSON_FRAMES: int = int(os.getenv("ARGUS_GATE_NEW_PERSON_FRAMES", "10"))
GATE_MOTION_THRESHOLD: float = float(os.getenv("ARGUS_GATE_MOTION_THRESHOLD", "0.05"))
GATE_MIN_INTERVAL: float = float(os.getenv("ARGUS_GATE_MIN_INTERVAL", "0.2"))
GATE_MAX_SKIP: int = int(os.getenv("ARGUS_GATE_MAX_SKIP", "30"))
GATE_POSE_DELTA: float = float(os.getenv("ARGUS_GATE_POSE_DELTA", "15.0"))
GATE_CONTACT_EVENT_FORCE: bool = os.getenv("ARGUS_GATE_CONTACT_FORCE", "true").lower() == "true"

# ── Detection (YOLOv11 / v9 / v8) ──
YOLO_MODEL: str = os.getenv("ARGUS_YOLO_MODEL", "yolo11n.pt")
YOLO_IMAGE_SIZE: int = int(os.getenv("ARGUS_YOLO_IMG_SIZE", "640"))
YOLO_IMG_SIZE: int = YOLO_IMAGE_SIZE  # alias for compatibility
YOLO_CONFIDENCE: float = float(os.getenv("ARGUS_YOLO_CONFIDENCE", "0.35"))
YOLO_IOU: float = float(os.getenv("ARGUS_YOLO_IOU", "0.45"))
YOLO_PERSON_CLASS: int = 0
DETECT_ALL_OBJECTS: bool = os.getenv("ARGUS_DETECT_ALL_OBJECTS", "true").lower() == "true"
OBJECT_CONFIDENCE: float = float(os.getenv("ARGUS_OBJECT_CONFIDENCE", "0.25"))
OBJECT_CLASSES_OF_INTEREST: Set[int] = {
    int(c) for c in os.getenv("ARGUS_OBJECT_CLASSES", "0,24,25,26,27,28,34,36,38,39,41,43,44,46,47,56,57,59,62,63,64,65,66,67,73,76,79").split(",")
}

# ── Tracking (ByteTrack) ──
TRACK_PERSIST: int = int(os.getenv("ARGUS_TRACK_PERSIST", "30"))
TRACKER_CONFIG: str = os.getenv("ARGUS_TRACKER_CONFIG", "bytetrack.yaml")

# ── CV/Device ──
CV_DEVICE: str = os.getenv("ARGUS_CV_DEVICE", "cpu")
CV_USE_FP16: bool = os.getenv("ARGUS_CV_FP16", "false").lower() == "true"

# ── Pose (RTMPose / MediaPipe) ──
POSE_ENABLED: bool = os.getenv("ARGUS_POSE_ENABLED", "true").lower() == "true"
POSE_MODEL_NAME: str = os.getenv("ARGUS_POSE_MODEL", "rtmpose_s")
POSE_DEVICE: str = os.getenv("ARGUS_POSE_DEVICE", "cpu")
POSE_CONFIDENCE: float = float(os.getenv("ARGUS_POSE_CONFIDENCE", "0.5"))
POSE_IMG_SIZE: Tuple[int, int] = (256, 192)
POSE_ONNX_PATH: Path = MODEL_DIR / os.getenv("ARGUS_POSE_ONNX", "rtmpose_s.onnx")
POSE_DET_ONNX_PATH: Path = MODEL_DIR / os.getenv("ARGUS_POSE_DET_ONNX", "rtmdet_n.onnx")
POSE_COCO_KEYPOINTS: int = 17

# ── Action (ST-GCN) ──
ACTION_ENABLED: bool = os.getenv("ARGUS_ACTION_ENABLED", "true").lower() == "true"
ACTION_MODEL: str = os.getenv("ARGUS_ACTION_MODEL", "stgcn")
ACTION_WINDOW: int = int(os.getenv("ARGUS_ACTION_WINDOW", "32"))
ACTION_STRIDE: int = int(os.getenv("ARGUS_ACTION_STRIDE", "8"))
ACTION_CONFIDENCE: float = float(os.getenv("ARGUS_ACTION_CONFIDENCE", "0.6"))
# Legacy aliases for pipeline/action_recognizer.py
ACTION_RECOG_ENABLED: bool = ACTION_ENABLED
ACTION_RECOG_DEVICE: str = CV_DEVICE
ACTION_RECOG_MODEL: str = ACTION_MODEL
ACTION_RECOG_WINDOW: int = ACTION_WINDOW
ACTION_RECOG_STRIDE: int = ACTION_STRIDE
ACTION_RECOG_CONFIDENCE: float = ACTION_CONFIDENCE

# ── Optical Flow (RAFT/Farneback) ──
FLOW_ENABLED: bool = os.getenv("ARGUS_FLOW_ENABLED", "true").lower() == "true"
FLOW_EVERY_N: int = int(os.getenv("ARGUS_FLOW_EVERY_N", "2"))
FLOW_RESIZE: Tuple[int, int] = (384, 256)

# ── Depth (MiDaS / Depth Anything V2) ──
DEPTH_ENABLED: bool = os.getenv("ARGUS_DEPTH_ENABLED", "true").lower() == "true"
DEPTH_DEVICE: str = CV_DEVICE
DEPTH_EVERY_N: int = int(os.getenv("ARGUS_DEPTH_EVERY_N", "2"))
DEPTH_RESIZE: Tuple[int, int] = (384, 384)
DEPTH_CONTACT_Z_THRESHOLD: float = float(os.getenv("ARGUS_DEPTH_CONTACT_Z_THRESHOLD", "30.0"))
DEPTH_ONNX_PATH: Path = MODEL_DIR / os.getenv("ARGUS_DEPTH_ONNX", "midas_v2_small.onnx")

# ── Gaze (MediaPipe FaceMesh) ──
GAZE_ENABLED: bool = os.getenv("ARGUS_GAZE_ENABLED", "true").lower() == "true"
GAZE_EVERY_N: int = int(os.getenv("ARGUS_GAZE_EVERY_N", "3"))
GAZE_CONFIDENCE: float = float(os.getenv("ARGUS_GAZE_CONFIDENCE", "0.5"))
GAZE_ANGLE_THRESHOLD: float = float(os.getenv("ARGUS_GAZE_ANGLE_THRESHOLD", "25.0"))

# ── Hand Tracking (MediaPipe Hands) ──
HAND_ENABLED: bool = os.getenv("ARGUS_HAND_ENABLED", "true").lower() == "true"
HAND_EVERY_N: int = int(os.getenv("ARGUS_HAND_EVERY_N", "3"))
HAND_CONFIDENCE: float = float(os.getenv("ARGUS_HAND_CONFIDENCE", "0.5"))
HAND_MAX_HANDS: int = int(os.getenv("ARGUS_HAND_MAX_HANDS", "4"))

# ── Segmentation (MobileSAM / SAM2) ──
SEG_ENABLED: bool = os.getenv("ARGUS_SEG_ENABLED", "true").lower() == "true"
SEG_DEVICE: str = CV_DEVICE
SEG_MODEL: str = os.getenv("ARGUS_SEG_MODEL", "mobile_sam.pt")
SEG_EVERY_N: int = int(os.getenv("ARGUS_SEG_EVERY_N", "5"))
SEG_ALPHA: float = float(os.getenv("ARGUS_SEG_ALPHA", "0.35"))

# ── Contact Detection (derived) ──
CONTACT_ENABLED: bool = os.getenv("ARGUS_CONTACT_ENABLED", "true").lower() == "true"
CONTACT_DISTANCE_THRESHOLD_MM: float = float(os.getenv("ARGUS_CONTACT_DIST_MM", "30.0"))
CONTACT_FLOW_THRESHOLD: float = float(os.getenv("ARGUS_CONTACT_FLOW_THRESHOLD", "0.15"))
CONTACT_IOU_OVERLAP: float = float(os.getenv("ARGUS_CONTACT_IOU", "0.08"))

# ── Face Recognition ──
FACE_DETECTION_ENABLED: bool = os.getenv("ARGUS_FACE_ENABLED", "true").lower() == "true"
FACE_MIN_CONFIDENCE: float = float(os.getenv("ARGUS_FACE_CONFIDENCE", "0.5"))

# ── Re-ID (CLIP embeddings) ──
REID_ENABLED: bool = os.getenv("ARGUS_REID_ENABLED", "true").lower() == "true"
REID_MATCH_THRESHOLD: float = float(os.getenv("ARGUS_REID_MATCH_THRESHOLD", "0.65"))

# ── VLM (Florence-2-large) ──
VLM_ENABLED: bool = os.getenv("ARGUS_VLM_ENABLED", "true").lower() == "true"
VLM_TURBO_MODE: bool = os.getenv("ARGUS_VLM_TURBO", "false").lower() == "true"
VLM_MODEL: str = os.getenv("ARGUS_VLM_MODEL", "microsoft/Florence-2-base")
VLM_DEVICE: str = os.getenv("ARGUS_VLM_DEVICE", "cpu")
VLM_MAX_SIZE: int = int(os.getenv("ARGUS_VLM_MAX_SIZE", "512"))
VLM_MAX_TOKENS: int = int(os.getenv("ARGUS_VLM_MAX_TOKENS", "1500"))
VLM_MAX_NEW_TOKENS: int = VLM_MAX_TOKENS  # alias for pipeline/vlm_engine.py
VLM_QUEUE_MAXSIZE: int = int(os.getenv("ARGUS_VLM_QUEUE_MAXSIZE", "100"))
VLM_MAX_CALLS_PER_MINUTE: int = int(os.getenv("ARGUS_VLM_MAX_CALLS", "30"))
VLM_CACHE_TTL: float = float(os.getenv("ARGUS_VLM_CACHE_TTL", "30.0"))
VLM_REFRESH_INTERVAL: float = float(os.getenv("ARGUS_VLM_REFRESH_INTERVAL", "3.0"))
VLM_CROP_PADDING: float = float(os.getenv("ARGUS_VLM_CROP_PADDING", "0.10"))
VLM_CAPTION_DELAY_NEW: float = float(os.getenv("ARGUS_VLM_CAPTION_DELAY", "2.0"))
VLM_STATE_CHANGE_THRESHOLD: float = float(os.getenv("ARGUS_VLM_STATE_CHANGE", "0.6"))
VQA_REFRESH_INTERVAL: float = float(os.getenv("ARGUS_VQA_REFRESH", "10.0"))
VQA_DEFAULT_QUESTIONS: List[str] = [
    "What is this person doing?",
    "Describe their appearance.",
    "What objects are nearby?",
    "What is their emotional state?",
]
VLM_DENSE_PROMPT: str = "<MORE_DETAILED_CAPTION>"
VLM_TASK_MAP: Dict[str, str] = {
    "dense": "<MORE_DETAILED_CAPTION>",
    "scene": "<DETAILED_CAPTION>",
    "od": "<OD>",
    "scene_full": "<MORE_DETAILED_CAPTION>",
}

# ── Dense Captioning ──
DENSE_CAPTIONING: bool = os.getenv("ARGUS_DENSE_CAPTIONING", "true").lower() == "true"
DENSE_FRAME_INTERVAL: int = int(os.getenv("ARGUS_DENSE_FRAME_INTERVAL", "1"))

# ── LLM (Qwen2.5-7B / 3B GGUF + Groq fallback) ──
LLM_ENABLED: bool = os.getenv("ARGUS_LLM_ENABLED", "true").lower() == "true"
LLM_MODEL_PATH: Optional[str] = os.getenv("ARGUS_LLM_MODEL_PATH", None)
LLM_CONTEXT_LENGTH: int = int(os.getenv("ARGUS_LLM_CONTEXT", "8192"))
LLM_THREADS: int = int(os.getenv("ARGUS_LLM_THREADS", "4"))
LLM_GPU_LAYERS: int = int(os.getenv("ARGUS_LLM_GPU_LAYERS", "0"))
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)

# ── Cognee Graph RAG ──
COGNEE_ENABLED: bool = os.getenv("ARGUS_COGNEE_ENABLED", "true").lower() == "true"
# Groq via openai provider + litellm routing. Valid LLMProvider: openai, ollama, anthropic, gemini, mistral, azure, bedrock, llama_cpp
COGNEE_LLM_PROVIDER: str = os.getenv("COGNEE_LLM_PROVIDER", "openai")
COGNEE_LLM_MODEL: str = os.getenv("COGNEE_LLM_MODEL", "groq/llama-3.1-8b-instant")
# Cognee v1.0+ reads its own env vars (set in cognee_bridge._configure_cognee_env):
#   LLM_PROVIDER=openai, LLM_MODEL=groq/llama-3.1-8b-instant, LLM_API_KEY, LLM_ENDPOINT

# ── ChromaDB + BGE-M3 ──
CHROMA_COLLECTION: str = os.getenv("ARGUS_CHROMA_COLLECTION", "argus_events")
VSS_MODEL: str = os.getenv("ARGUS_VSS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VSS_SIMILARITY_THRESHOLD: float = float(os.getenv("ARGUS_VSS_THRESHOLD", "0.75"))
VSS_MAX_EMBEDDINGS: int = int(os.getenv("ARGUS_VSS_MAX_EMBEDDINGS", "10000"))

# ── SQLite ──
SQLITE_PATH: Path = DB_DIR / os.getenv("ARGUS_SQLITE_FILE", "events.db")

# ── Knowledge Graph ──
GRAPH_PURGE_AGE: float = float(os.getenv("ARGUS_GRAPH_PURGE_AGE", "3600.0"))
GRAPH_SAVE_INTERVAL: int = int(os.getenv("ARGUS_GRAPH_SAVE_INTERVAL", "50"))

# ── Audio ──
AUDIO_ENABLED: bool = os.getenv("ARGUS_AUDIO_ENABLED", "false").lower() == "true"
AUDIO_DEVICE: str = os.getenv("ARGUS_AUDIO_DEVICE", "0")
AUDIO_SAMPLE_RATE: int = int(os.getenv("ARGUS_AUDIO_SAMPLE_RATE", "16000"))
AUDIO_WHISPER_MODEL: str = os.getenv("ARGUS_WHISPER_MODEL", "base")

# ── Causal ──
CAUSAL_ENABLED: bool = os.getenv("ARGUS_CAUSAL_ENABLED", "true").lower() == "true"
CAUSAL_OUTPUT_DIR: Path = CAUSAL_DIR
CAUSAL_WINDOW_SECONDS: float = float(os.getenv("ARGUS_CAUSAL_WINDOW", "300.0"))

# ── Alerts ──
ALERT_THROTTLE_SECONDS: float = float(os.getenv("ARGUS_ALERT_THROTTLE", "10.0"))
ALERT_DEDUP_SECONDS: float = float(os.getenv("ARGUS_ALERT_DEDUP", "30.0"))
SLACK_WEBHOOK_URL: str = os.getenv("ARGUS_SLACK_WEBHOOK", "")
DISCORD_WEBHOOK_URL: str = os.getenv("ARGUS_DISCORD_WEBHOOK", "")

# ── Display ──
DISPLAY_ENABLED: bool = os.getenv("ARGUS_DISPLAY_ENABLED", "true").lower() == "true"
DISPLAY_WINDOW: str = os.getenv("ARGUS_DISPLAY_WINDOW", "ARGUS V2 Live")
DISPLAY_FONT_SCALE: float = float(os.getenv("ARGUS_DISPLAY_FONT_SCALE", "0.6"))
DISPLAY_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
DISPLAY_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
DISPLAY_OVERLAY_ALPHA: float = float(os.getenv("ARGUS_DISPLAY_OVERLAY_ALPHA", "0.6"))

# ── Session ──
SESSION_ENABLED: bool = os.getenv("ARGUS_SESSION_ENABLED", "true").lower() == "true"
SUMMARY_INTERVAL: float = float(os.getenv("ARGUS_SUMMARY_INTERVAL", "60.0"))

# ── Logging ──
LOG_LEVEL: str = os.getenv("ARGUS_LOG_LEVEL", "INFO")
LOG_FILE: Path = LOG_DIR / "argus.log"
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ── ONNX Runtime ──
ONNX_PROVIDERS: list = ["CPUExecutionProvider"]
ONNX_INTER_THREADS: int = int(os.getenv("ARGUS_ONNX_THREADS", "4"))
ONNX_INTRA_THREADS: int = int(os.getenv("ARGUS_ONNX_INTRA_THREADS", "1"))
ONNX_GRAPH_OPT_LEVEL: int = 2
