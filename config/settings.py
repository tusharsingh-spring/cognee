"""ARGUS Video Intelligence System - Configuration"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
DB_DIR = DATA_DIR / "db"
CHROMA_DIR = DATA_DIR / "chroma"
MODEL_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
VIDEOS_DIR = DATA_DIR / "videos"
INPUT_VID_DIR = ROOT_DIR / "input_vid"

for d in [DATA_DIR, DB_DIR, CHROMA_DIR, MODEL_DIR, LOG_DIR, VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}

# --- Camera ---
CAMERA_URL = os.getenv("ARGUS_CAMERA_URL", "rtsp://192.168.0.100:554/stream")
CAMERA_INDEX = int(os.getenv("ARGUS_CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("ARGUS_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("ARGUS_CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("ARGUS_CAMERA_FPS", "30"))
VIDEO_FILE = os.getenv("ARGUS_VIDEO_FILE", "")

# --- Motion Detection (MOG2) ---
MOTION_HISTORY = int(os.getenv("ARGUS_MOTION_HISTORY", "500"))
MOTION_THRESHOLD = int(os.getenv("ARGUS_MOTION_THRESHOLD", "16"))
MOTION_MIN_AREA = int(os.getenv("ARGUS_MOTION_MIN_AREA", "500"))
MOTION_FRAME_SKIP = int(os.getenv("ARGUS_MOTION_FRAME_SKIP", "3"))
MOTION_LEARNING_RATE = float(os.getenv("ARGUS_MOTION_LR", "0.005"))

# --- Dense Captioning (video-file / offline analysis mode) ---
DENSE_CAPTIONING = os.getenv("ARGUS_DENSE_CAPTIONING", "true").lower() == "true"
DENSE_FRAME_INTERVAL = int(os.getenv("ARGUS_DENSE_FRAME_INTERVAL", "3"))
DENSE_VLM_QUEUE_MAXSIZE = int(os.getenv("ARGUS_DENSE_VLM_QUEUE_MAXSIZE", "200"))
DENSE_VLM_MAX_CALLS = int(os.getenv("ARGUS_DENSE_VLM_MAX_CALLS", "60"))

# --- Person Detection (YOLOv8) ---
YOLO_MODEL = os.getenv("ARGUS_YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("ARGUS_YOLO_CONFIDENCE", "0.5"))
YOLO_IOU = float(os.getenv("ARGUS_YOLO_IOU", "0.5"))
YOLO_PERSON_CLASS = int(os.getenv("ARGUS_YOLO_PERSON_CLASS", "0"))
YOLO_IMAGE_SIZE = int(os.getenv("ARGUS_YOLO_IMAGE_SIZE", "320"))

# --- Tracking (ByteTrack) ---
TRACK_PERSIST = int(os.getenv("ARGUS_TRACK_PERSIST", "30"))
TRACKER_CONFIG = os.getenv("ARGUS_TRACKER_CONFIG", "bytetrack.yaml")

# --- VLM (Florence-2) ---
# Use microsoft/Florence-2-base (0.23B, fast, cached) or Florence-2-large (0.77B, stronger)
VLM_MODEL = os.getenv("ARGUS_VLM_MODEL", "microsoft/Florence-2-base")
VLM_DEVICE = os.getenv("ARGUS_VLM_DEVICE", "cpu")
VLM_MAX_SIZE = int(os.getenv("ARGUS_VLM_MAX_SIZE", "512"))
VLM_CROP_PADDING = float(os.getenv("ARGUS_VLM_CROP_PADDING", "0.10"))
VLM_TURBO_MODE = os.getenv("ARGUS_TURBO", "false").lower() == "true"
VLM_TASK_MAP = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "od": "<OD>",
    "ocr": "<OCR>",
    "vqa": "<VQA>",
}

# --- VLM Trigger Rules ---
VLM_CAPTION_DELAY_NEW = float(os.getenv("ARGUS_VLM_DELAY_NEW", "0.3"))
VLM_REFRESH_INTERVAL = float(os.getenv("ARGUS_VLM_REFRESH_INTERVAL", "3.0"))
VLM_STATE_CHANGE_THRESHOLD = float(os.getenv("ARGUS_VLM_STATE_CHANGE", "0.10"))
VLM_CACHE_TTL = float(os.getenv("ARGUS_VLM_CACHE_TTL", "30.0"))
VLM_QUEUE_MAXSIZE = int(os.getenv("ARGUS_VLM_QUEUE_MAXSIZE", "100"))
VLM_MAX_CALLS_PER_MINUTE = int(os.getenv("ARGUS_VLM_MAX_CALLS", "30"))

# --- VQA ---
VQA_REFRESH_INTERVAL = float(os.getenv("ARGUS_VQA_REFRESH", "3.0"))
VQA_DEFAULT_QUESTIONS = [
    "What is the person wearing?",
    "What is the person holding?",
    "What action is the person performing?",
]

# --- Visual Similarity Search ---
VSS_MODEL = os.getenv("ARGUS_VSS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VSS_SIMILARITY_THRESHOLD = float(os.getenv("ARGUS_VSS_THRESHOLD", "0.75"))
VSS_MAX_EMBEDDINGS = int(os.getenv("ARGUS_VSS_MAX_EMBEDDINGS", "1000"))

# --- Object Detection (YOLO all classes) ---
DETECT_ALL_OBJECTS = os.getenv("ARGUS_DETECT_ALL_OBJECTS", "true").lower() == "true"
OBJECT_CONFIDENCE = float(os.getenv("ARGUS_OBJECT_CONFIDENCE", "0.35"))
OBJECT_CLASSES_OF_INTEREST = set(
    int(c) for c in os.getenv("ARGUS_OBJECT_CLASSES", "39,41,44,46,47,62,63,64,67,73,75,76,77").split(",")
)

# --- Face Recognition (InsightFace) ---
FACE_DETECTION_ENABLED = os.getenv("ARGUS_FACE_ENABLED", "true").lower() == "true"
FACE_RECOGNITION_MODEL = os.getenv("ARGUS_FACE_MODEL", "buffalo_l")
FACE_MIN_CONFIDENCE = float(os.getenv("ARGUS_FACE_MIN_CONF", "0.5"))

# --- Person Re-Identification (OSNet) ---
REID_ENABLED = os.getenv("ARGUS_REID_ENABLED", "true").lower() == "true"
REID_MODEL = os.getenv("ARGUS_REID_MODEL", "osnet_x0_25")
REID_MATCH_THRESHOLD = float(os.getenv("ARGUS_REID_MATCH_THRESHOLD", "0.65"))

# --- Session Management ---
SESSION_ENABLED = os.getenv("ARGUS_SESSION_ENABLED", "true").lower() == "true"
SESSION_DIR = DATA_DIR / "sessions"

# --- Knowledge Graph ---
GRAPH_PURGE_AGE = float(os.getenv("ARGUS_GRAPH_PURGE_AGE", "3600.0"))
GRAPH_SAVE_INTERVAL = int(os.getenv("ARGUS_GRAPH_SAVE_INTERVAL", "100"))

# --- Vector Store (ChromaDB) ---
CHROMA_COLLECTION = os.getenv("ARGUS_CHROMA_COLLECTION", "person_embeddings")

# --- SQLite ---
SQLITE_PATH = DB_DIR / os.getenv("ARGUS_SQLITE_FILE", "events.db")

# --- Alerts ---
ALERT_THROTTLE_SECONDS = float(os.getenv("ARGUS_ALERT_THROTTLE", "10.0"))
ALERT_DEDUP_SECONDS = float(os.getenv("ARGUS_ALERT_DEDUP", "30.0"))
SLACK_WEBHOOK_URL = os.getenv("ARGUS_SLACK_WEBHOOK", "")
DISCORD_WEBHOOK_URL = os.getenv("ARGUS_DISCORD_WEBHOOK", "")

# --- Dashboard ---
DASHBOARD_HOST = os.getenv("ARGUS_DASHBOARD_HOST", "localhost")
DASHBOARD_PORT = int(os.getenv("ARGUS_DASHBOARD_PORT", "8501"))

# --- Summary ---
SUMMARY_INTERVAL = float(os.getenv("ARGUS_SUMMARY_INTERVAL", "60.0"))

# --- Display ---
DISPLAY_WINDOW = os.getenv("ARGUS_DISPLAY_WINDOW", "ARGUS Live")
DISPLAY_FONT_SCALE = float(os.getenv("ARGUS_DISPLAY_FONT_SCALE", "0.6"))
DISPLAY_BOX_COLOR = (0, 255, 0)
DISPLAY_TEXT_COLOR = (255, 255, 255)
DISPLAY_OVERLAY_ALPHA = float(os.getenv("ARGUS_DISPLAY_OVERLAY_ALPHA", "0.7"))

# --- Smart Frame Gate ---
GATE_ENABLED = os.getenv("ARGUS_GATE_ENABLED", "true").lower() == "true"
GATE_MOTION_THRESHOLD = float(os.getenv("ARGUS_GATE_MOTION_THRESHOLD", "0.05"))
GATE_POSE_DELTA = float(os.getenv("ARGUS_GATE_POSE_DELTA", "15.0"))
GATE_MIN_INTERVAL = float(os.getenv("ARGUS_GATE_MIN_INTERVAL", "0.1"))
GATE_MAX_SKIP = int(os.getenv("ARGUS_GATE_MAX_SKIP", "10"))
GATE_CONTACT_EVENT_FORCE = os.getenv("ARGUS_GATE_CONTACT_FORCE", "true").lower() == "true"
GATE_NEW_PERSON_FRAMES = int(os.getenv("ARGUS_GATE_NEW_PERSON_FRAMES", "15"))

# --- Pose Estimation (RTMPose-S via ONNX) ---
POSE_ENABLED = os.getenv("ARGUS_POSE_ENABLED", "true").lower() == "true"
POSE_MODEL = os.getenv("ARGUS_POSE_MODEL", "rtmpose_s")
POSE_DEVICE = os.getenv("ARGUS_POSE_DEVICE", "cpu")
POSE_CONFIDENCE = float(os.getenv("ARGUS_POSE_CONFIDENCE", "0.5"))
POSE_ONNX_PATH = MODEL_DIR / os.getenv("ARGUS_POSE_ONNX", "rtmpose_s.onnx")
POSE_DET_ONNX_PATH = MODEL_DIR / os.getenv("ARGUS_POSE_DET_ONNX", "rtmdet_nano.onnx")
POSE_IMG_SIZE = tuple(int(v) for v in os.getenv("ARGUS_POSE_IMG_SIZE", "256,192").split(","))
POSE_COCO_KEYPOINTS = 17

# --- Optical Flow (RAFT-small) ---
FLOW_ENABLED = os.getenv("ARGUS_FLOW_ENABLED", "true").lower() == "true"
FLOW_MODEL = os.getenv("ARGUS_FLOW_MODEL", "raft_small")
FLOW_DEVICE = os.getenv("ARGUS_FLOW_DEVICE", "cpu")
FLOW_EVERY_N = int(os.getenv("ARGUS_FLOW_EVERY_N", "2"))
FLOW_RESIZE = tuple(int(v) for v in os.getenv("ARGUS_FLOW_RESIZE", "384,256").split(","))

# --- Depth Estimation (Depth Anything V2 Small) ---
DEPTH_ENABLED = os.getenv("ARGUS_DEPTH_ENABLED", "true").lower() == "true"
DEPTH_MODEL = os.getenv("ARGUS_DEPTH_MODEL", "depth_anything_v2_small")
DEPTH_DEVICE = os.getenv("ARGUS_DEPTH_DEVICE", "cpu")
DEPTH_EVERY_N = int(os.getenv("ARGUS_DEPTH_EVERY_N", "2"))
DEPTH_RESIZE = tuple(int(v) for v in os.getenv("ARGUS_DEPTH_RESIZE", "384,384").split(","))
DEPTH_MM_SCALE = float(os.getenv("ARGUS_DEPTH_MM_SCALE", "1.0"))
DEPTH_CONTACT_Z_THRESHOLD = float(os.getenv("ARGUS_DEPTH_CONTACT_Z_THRESHOLD", "30.0"))

# --- Contact Detection (derived, no ML) ---
CONTACT_ENABLED = os.getenv("ARGUS_CONTACT_ENABLED", "true").lower() == "true"
CONTACT_DISTANCE_THRESHOLD_MM = float(os.getenv("ARGUS_CONTACT_DIST_MM", "30.0"))
CONTACT_FLOW_THRESHOLD = float(os.getenv("ARGUS_CONTACT_FLOW_THRESHOLD", "0.15"))
CONTACT_IOU_OVERLAP = float(os.getenv("ARGUS_CONTACT_IOU", "0.08"))

# --- Segmentation (SAM2 / MobileSAM) ---
SEG_ENABLED = os.getenv("ARGUS_SEG_ENABLED", "true").lower() == "true"
SEG_MODEL = os.getenv("ARGUS_SEG_MODEL", "mobile_sam")
SEG_DEVICE = os.getenv("ARGUS_SEG_DEVICE", "cpu")
SEG_EVERY_N = int(os.getenv("ARGUS_SEG_EVERY_N", "5"))
SEG_ALPHA = float(os.getenv("ARGUS_SEG_ALPHA", "0.35"))

# --- Hand Tracking (MediaPipe Hands) ---
HAND_ENABLED = os.getenv("ARGUS_HAND_ENABLED", "true").lower() == "true"
HAND_CONFIDENCE = float(os.getenv("ARGUS_HAND_CONFIDENCE", "0.5"))
HAND_EVERY_N = int(os.getenv("ARGUS_HAND_EVERY_N", "3"))
HAND_MAX_HANDS = int(os.getenv("ARGUS_HAND_MAX_HANDS", "4"))

# --- Gaze Estimation (MediaPipe FaceMesh iris) ---
GAZE_ENABLED = os.getenv("ARGUS_GAZE_ENABLED", "true").lower() == "true"
GAZE_MODEL = os.getenv("ARGUS_GAZE_MODEL", "mediapipe_facemesh")
GAZE_EVERY_N = int(os.getenv("ARGUS_GAZE_EVERY_N", "3"))
GAZE_CONFIDENCE = float(os.getenv("ARGUS_GAZE_CONFIDENCE", "0.5"))
GAZE_ANGLE_THRESHOLD = float(os.getenv("ARGUS_GAZE_ANGLE_THRESHOLD", "25.0"))

# --- Action Recognition (ST-GCN / CTR-GCN on pose sequences) ---
ACTION_RECOG_ENABLED = os.getenv("ARGUS_ACTION_RECOG_ENABLED", "true").lower() == "true"
ACTION_RECOG_MODEL = os.getenv("ARGUS_ACTION_RECOG_MODEL", "stgcn")
ACTION_RECOG_DEVICE = os.getenv("ARGUS_ACTION_RECOG_DEVICE", "cpu")
ACTION_RECOG_WINDOW = int(os.getenv("ARGUS_ACTION_RECOG_WINDOW", "32"))
ACTION_RECOG_STRIDE = int(os.getenv("ARGUS_ACTION_RECOG_STRIDE", "8"))
ACTION_RECOG_CONFIDENCE = float(os.getenv("ARGUS_ACTION_RECOG_CONFIDENCE", "0.6"))

# --- Causal Variable Extraction (PCMCI+ feed) ---
CAUSAL_ENABLED = os.getenv("ARGUS_CAUSAL_ENABLED", "true").lower() == "true"
CAUSAL_OUTPUT_DIR = DATA_DIR / os.getenv("ARGUS_CAUSAL_DIR", "causal")
CAUSAL_WINDOW_SECONDS = float(os.getenv("ARGUS_CAUSAL_WINDOW", "30.0"))

# --- VLM Dense Captioning ---
VLM_MAX_NEW_TOKENS = int(os.getenv("ARGUS_VLM_MAX_TOKENS", "1500"))
VLM_DENSE_PROMPT = os.getenv("ARGUS_VLM_DENSE_PROMPT", "true").lower() == "true"

# --- Audio Pipeline (Whisper) ---
AUDIO_ENABLED = os.getenv("ARGUS_AUDIO_ENABLED", "false").lower() == "true"
AUDIO_WHISPER_MODEL = os.getenv("ARGUS_AUDIO_WHISPER_MODEL", "small")
AUDIO_DEVICE = os.getenv("ARGUS_AUDIO_DEVICE", "cpu")
AUDIO_SAMPLE_RATE = int(os.getenv("ARGUS_AUDIO_SAMPLE_RATE", "16000"))

# --- Unified CV Device ---
CV_DEVICE = os.getenv("ARGUS_CV_DEVICE", "cpu")
CV_USE_FP16 = os.getenv("ARGUS_CV_FP16", "true").lower() == "true"
CV_ONNX_DIR = MODEL_DIR

# --- Profiling ---
PROFILE_EVERY_N_FRAMES = int(os.getenv("ARGUS_PROFILE_FRAMES", "100"))
PROFILE_ENABLED = os.getenv("ARGUS_PROFILE_ENABLED", "true").lower() == "true"

# --- Logging ---
LOG_LEVEL = os.getenv("ARGUS_LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "argus.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
