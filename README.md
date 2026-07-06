<p align="center">
  <img src="https://raw.githubusercontent.com/tusharsingh-spring/cognee/main/assets/argus-banner.png" alt="ARGUS V3" width="800" onerror="this.style.display='none'"/>
</p>

<h1 align="center">ARGUS V3 — Cognee-Powered Video Intelligence</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5-red?logo=pytorch" alt="PyTorch"/></a>
  <a href="https://github.com/tusharsingh-spring/cognee/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"/></a>
  <a href="https://cognee.ai"><img src="https://img.shields.io/badge/cognee-1.0%2B-purple?logo=graphql" alt="Cognee"/></a>
  <a href="https://huggingface.co/microsoft/Florence-2-base"><img src="https://img.shields.io/badge/VLM-Florence--2-orange?logo=huggingface" alt="Florence-2"/></a>
  <a href="https://platform.openai.com"><img src="https://img.shields.io/badge/LLM-Qwen2.5%20%7C%20Groq-lightgrey" alt="LLM"/></a>
</p>

<p align="center">
  <b>A 3-layer video intelligence pipeline that sees, understands, and remembers.</b><br/>
  Computer Vision → Visual Language Model → Large Language Model → <b>Knowledge Graph (Cognee Graph RAG)</b>
</p>

---

## Overview

**ARGUS V3** is a real-time CCTV intelligence system that ingests video streams and builds a rich, queryable **knowledge graph** of everything it observes. Unlike traditional surveillance systems that only detect objects, ARGUS creates a **semantic timeline** — who did what, when, to whom, with what objects, and what happened next.

At its core, ARGUS integrates the **cognee** Graph RAG framework to construct a temporal knowledge graph from structured perception data, VLM visual reasoning, and LLM narrative generation. You can then ask natural language questions about your footage and get answers grounded in the graph, vectors, and event logs.

> "What was Person_3 doing between 14:00 and 14:30?"  
> "Show me all interactions between people in the last hour."  
> "Has anyone entered the restricted zone?"

---

## Features

- ⚡ **Real-time 3-Layer Pipeline** — Perception, VLM, and LLM run concurrently with smart gating for maximum throughput
- 🧠 **Cognee Graph RAG** — Temporal knowledge graph (NetworkX) enriched with entities, actions, relationships, and daily patterns
- 🔍 **Multi-Modal Retrieval** — Combine graph queries, vector similarity (ChromaDB + BGE-M3), and SQL timeline queries
- 👁️ **YOLOv11n Detection** — Person + 80-class COCO object detection with ByteTrack multi-object tracking
- 🕺 **Pose + Action Recognition** — 17-point COCO pose via RTMPose; ST-GCN action classification (walking, running, reaching, grabbing, falling...)
- 👀 **Gaze Estimation** — MediaPipe FaceMesh-based gaze direction + person-person gaze targeting
- 📏 **Monocular Depth** — MiDaS small ONNX for per-person depth and 3D positioning
- 🌊 **Optical Flow** — Farneback dense flow for motion magnitude per bounding-box
- ✋ **Hand Tracking** — MediaPipe Hands with 21-landmark detection and grip/open classification
- ✂️ **Segmentation** — MobileSAM for pixel-accurate person boundaries
- 🤝 **Contact Detection** — IoU + depth + flow fusion to detect physical contact events
- 🖼️ **VLM Layer (Florence-2)** — Dense captioning, object detection, scene description on gated keyframes
- 🧾 **LLM Reasoning** — Local Qwen2.5-3B (GGUF) or Groq cloud API for narrative, intent, anomaly scoring
- 🔗 **Face Recognition** — DeepFace for person identity matching across frames
- 🔄 **Re-Identification** — CLIP-embedding person re-identification for persistent identity
- 🚨 **Alert System** — Threat keyword detection, unusual behavior alerts, configurable Slack/Discord webhooks
- 🔬 **Causal Extraction** — Causal event chaining from pose, depth, flow, and contact data
- 📊 **Streamlit Dashboard** — Live metrics, event log viewer, graph stats, and alert panel
- 💬 **Streamlit Chat UI** — Natural language Q&A over your CCTV footage with context from all three memory stores
- 📝 **Session + Periodic Summaries** — Auto-generated event summaries every N seconds
- 🧩 **Fully modular** — Every component can be enabled/disabled via environment variables
- 🎯 **Smart Gating** — YOLOv8n fast gate skips idle frames, VLM triggered only on state changes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARGUS V3                                        │
│                    3-Layer Video Intelligence System                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   LAYER 1        │    │   LAYER 2        │    │   LAYER 3        │       │
│  │   PERCEPTION     │───▶│   VLM            │───▶│   LLM            │       │
│  │                  │    │                  │    │                  │       │
│  │  ┌────────────┐  │    │  ┌────────────┐  │    │  ┌────────────┐  │       │
│  │  │ YOLOv11n   │  │    │  │ Florence-2 │  │    │  │ Qwen2.5-3B │  │       │
│  │  │ Detection  │  │    │  │ Dense Capt │  │    │  │ Narrative  │  │       │
│  │  └────────────┘  │    │  └────────────┘  │    │  └────────────┘  │       │
│  │  ┌────────────┐  │    │  ┌────────────┐  │    │  ┌────────────┐  │       │
│  │  │ RTMPose    │  │    │  │ Scene Desc │  │    │  │ Intent     │  │       │
│  │  │ Pose (17)  │  │    │  └────────────┘  │    │  └────────────┘  │       │
│  │  └────────────┘  │    │  ┌────────────┐  │    │  ┌────────────┐  │       │
│  │  ┌────────────┐  │    │  │ Object Det │  │    │  │ Anomaly    │  │       │
│  │  │ ST-GCN     │  │    │  └────────────┘  │    │  └────────────┘  │       │
│  │  │ Action Rec │  │    │                  │    │  ┌────────────┐  │       │
│  │  └────────────┘  │    │  Gated trigger   │    │  │ Notify     │  │       │
│  │  ┌────────────┐  │    │  (state change,  │    │  └────────────┘  │       │
│  │  │ MediaPipe  │  │    │   new person,    │    │                  │       │
│  │  │ Gaze+Hand  │  │    │   contact,       │    │  Groq fallback   │       │
│  │  └────────────┘  │    │   periodic)      │    │  available       │       │
│  │  ┌────────────┐  │    │                  │    │                  │       │
│  │  │ MiDaS      │  │    └──────────────────┘    └──────────────────┘       │
│  │  │ Depth      │  │              │                      │                 │
│  │  └────────────┘  │              │                      │                 │
│  │  ┌────────────┐  │              ▼                      ▼                 │
│  │  │ Farneback  │  │    ┌─────────────────────────────────────────┐        │
│  │  │ Flow       │  │    │          KNOWLEDGE LAYER                │        │
│  │  └────────────┘  │    │                                         │        │
│  │  ┌────────────┐  │    │  ┌───────────┐  ┌───────────┐  ┌──────┐│        │
│  │  │ MobileSAM  │  │    │  │  Cognee   │  │ ChromaDB  │  │SQLite││        │
│  │  │ Segment    │  │    │  │ Graph RAG │  │  Vector   │  │Event ││        │
│  │  └────────────┘  │    │  │ (NetworkX)│  │  Store    │  │ Log  ││        │
│  │  ┌────────────┐  │    │  └─────┬─────┘  └─────┬─────┘  └──┬───┘│        │
│  │  │ Contact    │  │    │        │              │           │    │        │
│  │  │ Detector   │  │    │        ▼              ▼           ▼    │        │
│  │  └────────────┘  │    │  ┌──────────────────────────────────┐  │        │
│  │                  │    │  │      Multi-Modal Retrieval       │  │        │
│  │  Parallel via    │    │  │  Graph Query + Vector + SQL      │  │        │
│  │  ThreadPool (6)  │    │  └──────────────────────────────────┘  │        │
│  │                  │    │                                         │        │
│  │  Gated by        │    │  ┌──────────┐  ┌──────────┐  ┌───────┐ │        │
│  │  YOLOv8n (fast)  │    │  │ Session  │  │  Alert   │  │Webhook│ │        │
│  └──────────────────┘    │  │ Manager  │  │  Engine  │  │Notif. │ │        │
│                          │  └──────────┘  └──────────┘  └───────┘ │        │
│                          └─────────────────────────────────────────┘        │
│                                          │                                  │
│                    ┌─────────────────────┼─────────────────────┐            │
│                    ▼                     ▼                     ▼            │
│             ┌───────────┐        ┌───────────┐        ┌───────────┐        │
│             │  OpenCV   │        │ Streamlit │        │ Streamlit │        │
│             │  Display  │        │ Chat UI   │        │ Dashboard │        │
│             │  (live)   │        │ :8501     │        │ :8502     │        │
│             └───────────┘        └───────────┘        └───────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Video Frame
    │
    ├── YOLOv8n Gate (skip idle frames)
    │
    ▼
Layer 1 ──▶ PersonEntry[track_id, bbox, confidence]
    │        ObjectEntry[class_id, name, confidence]
    │        PoseResult[17 keypoints]
    │        ActionResult[action, confidence]
    │        GazeResult[direction, target_person_id]
    │        DepthInfo[torso_depth, body_depth]
    │        FlowInfo[mean_magnitude]
    │        HandInfo[21 landmarks, handedness]
    │        ContactInfo[person_a, person_b, score]
    │        PerceptionPacket ──────────────────────────┐
    │                                                    │
    ▼                                                    ▼
Layer 2 ──▶ dense_caption + scene_caption + OD ──▶ Cognee Graph RAG (NetworkX)
    │         (Florence-2, gated)                      │  Nodes: Person, Object,
    │                                                    │  Event, Action, Scene
    ▼                                                    │  Edges: INTERACTS, HOLDS,
Layer 3 ──▶ narrative + intent + anomaly ─────────────▶│  NEAR, LOOKS_AT, PERFORMS
    │         (Qwen2.5 / Groq)                          │
    │                                                    ▼
    ▼                                              ┌─────────────┐
ChromaDB + SQLite + CogneeBridge ─────────────────▶│  Chat UI    │
    (vector, timeline, graph search)               │  Dashboard  │
                                                   └─────────────┘
```

---

## Project Structure

```
cognee/
├── main.py                          # ARGUS main class: pipeline orchestration
├── run.py                           # Unified launcher (pipeline + chat + dashboard)
├── requirements.txt                 # Python dependencies
├── integration_test.py              # Quick integration smoke test
│
├── config/
│   └── settings.py                  # Central configuration (250+ env-controllable settings)
│
├── layer1_perception/               # Perception models (CV)
│   ├── perception_pipeline.py       # Orchestrator: runs all models in parallel
│   ├── perception_schema.py         # Pydantic data models (PerceptionPacket)
│   ├── detector.py                  # YOLOv11n person + object detection
│   ├── pose.py                      # RTMPose 17-keypoint estimation
│   ├── action_stgcn.py              # ST-GCN action recognition
│   ├── gaze.py                      # MediaPipe FaceMesh gaze estimation
│   ├── depth.py                     # MiDaS v2 monocular depth
│   ├── flow.py                      # Farneback optical flow
│   ├── hand_tracker.py              # MediaPipe Hands (21 landmarks)
│   ├── segmentation.py              # MobileSAM instance segmentation
│   ├── contact.py                   # IoU + depth + flow contact detection
│   ├── gating.py                    # YOLOv8n fast gate (skip idle frames)
│   └── fast_actions.py              # Heuristic fast action detector
│
├── layer2_vlm/                      # Visual Language Model (Florence-2)
│   ├── vlm_engine.py                # VLM worker queue + model loading
│   ├── vlm_prompt.py                # Task-aware prompt templates
│   └── vlm_trigger.py               # Smart gating: trigger VLM on state changes
│
├── layer3_llm/                      # Large Language Model reasoning
│   └── llm_engine.py                # Qwen2.5-3B GGUF local / Groq cloud fallback
│
├── graph_rag/                       # Cognee Graph RAG knowledge layer
│   ├── knowledge_graph.py           # NetworkX graph: nodes, edges, ingestion, query
│   └── cognee_bridge.py             # JSONL event store bridging perception to graph
│
├── storage/                         # Vector + Relational storage
│   ├── vector_store.py              # ChromaDB: 3 collections (events, frames, daily)
│   └── sqlite_store.py              # SQLite: events, nodes, edges, daily summaries
│
├── knowledge/                       # Higher-level knowledge modules
│   ├── groq_chat.py                 # Graph RAG + Vector RAG chatbot (Groq API)
│   ├── summary_engine.py            # Periodic session summaries
│   ├── session_manager.py           # Session start/end, stats tracking
│   ├── cctv_qa.py                   # CCTV-specific question answering
│   ├── project_tracker.py           # Project metadata tracking
│   └── graph_store.py               # Graph serialization/deserialization
│
├── pipeline/                        # Legacy + supplementary pipeline modules
│   ├── capture.py                   # Video/webcam frame capture
│   ├── display.py                   # OpenCV annotated display
│   ├── face_recognition.py          # DeepFace identity recognition
│   ├── reid_handler.py              # CLIP-based person re-identification
│   ├── action_engine.py             # Temporal action log
│   ├── causal_extractor.py          # Causal event chain extraction
│   ├── scene_analyzer.py            # Scene-level analysis
│   ├── video_chunker.py             # Video chunking for processing
│   └── ...                          # (vlm_engine, vss_handler, vqa_handler, etc.)
│
├── notifications/                   # Alerting & notifications
│   ├── alert_engine.py              # Rule-based alert evaluation + dedup
│   └── webhook.py                   # Slack/Discord webhook sender
│
├── dashboard/                       # Streamlit dashboard
│   └── app.py                       # Live metrics, events, graph stats
│
├── chat_ui/                         # Streamlit chat interface
│   └── app.py                       # Natural language CCTV query UI
│
├── utils/                           # Shared utilities
│   ├── logger.py                    # Structured logging
│   ├── profiler.py                  # Frame-level performance profiler
│   └── model_cache.py               # ONNX model download + cache
│
└── tests/                           # Unit & integration tests
    ├── test_capture.py
    ├── test_detection.py
    └── test_vlm.py
```

---

## Installation / Getting Started

### Prerequisites

- **Python 3.10+**
- **PyTorch 2.5+** (CUDA optional; CPU-only supported)
- **MediaPipe** (CPU)
- **ONNX Runtime** (CPU)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/tusharsingh-spring/cognee.git
cd cognee
```

### 2. Set Up Python Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First launch downloads ~2-4 GB of model weights (YOLO, RTMPose, Florence-2, MiDaS, MobileSAM, DeepFace). Ensure a stable internet connection. Models are cached in `data/models/`.

### 4. Configure Environment (Optional)

Copy the example environment file and edit as needed:

```bash
# All settings have sensible defaults.
# Create a .env file only if you want to customize:
cp .env.example .env  # (if available)
```

Key variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `ARGUS_VLM_MODEL` | `microsoft/Florence-2-base` | VLM model for Layer 2 |
| `ARGUS_YOLO_MODEL` | `yolo11n.pt` | Detection backbone |
| `GROQ_API_KEY` | — | Enable Groq cloud fallback for chat/LLM |
| `ARGUS_SLACK_WEBHOOK` | — | Slack alert webhook URL |
| `ARGUS_DISCORD_WEBHOOK` | — | Discord alert webhook URL |
| `ARGUS_GATE_ENABLED` | `true` | Toggle YOLOv8n smart gating |
| `ARGUS_VLM_ENABLED` | `true` | Toggle VLM layer |
| `ARGUS_TURBO` | `false` | Skip VLM + LLM (heuristic only) |

### 5. Place Input Videos

Drop your CCTV footage into either directory:

```
input_vid/      # Root-level video input
data/videos/    # Alternative video location
```

Supported formats: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.flv`, `.wmv`, `.m4v`

---

## Usage

### Quick Start (Video File)

```bash
# Auto-detect first video in input_vid/ or data/videos/
python run.py --video auto

# Process a specific video
python run.py --video path/to/cctv_footage.mp4

# List all discovered videos
python run.py --video list
```

### Webcam / Live Camera

```bash
# Local webcam (index 0)
python run.py --webcam

# RTSP / HTTP camera stream
python run.py --camera "rtsp://192.168.1.100:554/stream"

# Camera by index
python run.py --camera 1
```

### Turbo Mode (Skip VLM + LLM)

```bash
# Heuristic-only mode for maximum speed
python run.py --video auto --turbo
```

### Headless Mode (No Display Window)

```bash
# Server / background processing
python run.py --video auto --headless --no-chat
```

### Chat UI Only (No Pipeline)

```bash
# Launch chat interface to query stored data
python run.py --chat-only
# Open: http://localhost:8501
```

### Dashboard Only

```bash
# Launch monitoring dashboard
python run.py --dashboard-only
# Open: http://localhost:8502
```

### Full System (Pipeline + Chat + Dashboard)

```bash
# All three components simultaneously
python run.py --video auto
# Pipeline:      processing in terminal
# Chat UI:       http://localhost:8501
# Dashboard:     http://localhost:8502
```

### Using `main.py` Directly

```bash
# Full pipeline only
python main.py --video auto

# Pipeline + dashboard
python main.py --video auto --dashboard

# Pipeline + chat
python main.py --video auto --chat

# Webcam with headless display
python main.py --webcam --headless
```

---

## How It Works

### The 3-Layer Loop

Each frame goes through a gating check (YOLOv8n) before the pipeline activates:

1. **YOLOv8n Gate** — A lightweight YOLOv8n model scans every frame at 320× resolution. If no motion or person is detected, the frame is skipped. This typically filters out 60-80% of frames.

2. **Layer 1 — Perception** — All 9 CV models run in parallel via `ThreadPoolExecutor(max_workers=6)`:
   - Person detection (YOLOv11n) + multi-object tracking (ByteTrack)
   - Object detection (80 COCO classes: backpack, cell phone, knife, chair...)
   - Pose estimation (17 keypoints via RTMPose ONNX)
   - Action recognition (ST-GCN on 32-frame pose windows)
   - Gaze estimation (MediaPipe FaceMesh → direction vector)
   - Depth estimation (MiDaS v2 ONNX → per-person torso depth)
   - Optical flow (Farneback → per-bbox motion magnitude)
   - Hand tracking (MediaPipe Hands → 21 landmarks per hand)
   - Segmentation (MobileSAM → per-person pixel masks)
   - Contact detection (IoU + depth proximity + flow correlation)

3. **Layer 2 — VLM (Gated)** — Florence-2 provides visual context that structured models miss. Not run on every frame — the `VLMTriggerManager` gates it on:
   - New person appears
   - Action state changes (standing→walking, etc.)
   - Contact events
   - Periodic full-scene descriptors (every 50 frames)
   
   Tasks: dense captioning (`<MORE_DETAILED_CAPTION>`), object detection (`<OD>`), scene description.

4. **Layer 3 — LLM (Gated)** — The LLM receives perception + VLM data and produces:
   - Narrative (natural language description)
   - Intent inference
   - Anomaly score (0.0–1.0)
   - Notification decision + urgency
   - Store tags for retrieval

5. **Knowledge Ingestion** — Every frame's perception packet, VLM output, and LLM reasoning is written to:
   - **Cognee Graph RAG** — NetworkX directed graph with Person/Object/Action/Event/Scene nodes and INTERACTS_WITH/HOLDS/LOOKS_AT/PERFORMS edges
   - **ChromaDB** — Vector embeddings (sentence-transformers/all-MiniLM-L6-v2) across 3 collections for semantic similarity search
   - **SQLite** — Structured event log (events, nodes, edges tables) with timestamps
   - **CogneeBridge** — JSONL file-based event store for timeline reconstruction

6. **Alerting** — Every caption is checked against threat keywords. Alerts are deduplicated and throttled, then sent via configurable Slack/Discord webhooks.

### Multi-Modal Retrieval (Chat UI)

When you ask a question in the chat:

```
User: "What was Person_5 doing?"
```

The system performs a **3-way retrieval**:

1. **Graph Query** — Search the NetworkX knowledge graph for `Person_5` node and traverse connected Action/Scene/Object nodes
2. **Vector Search** — ChromaDB semantic similarity across all event embeddings (finds related captions even if Person_5 wasn't directly mentioned in that chunk)
3. **SQL Timeline** — SQLite chronological event log for temporal queries ("before", "after", "during the last hour")

Combined context is fed to the LLM (Groq `llama-3.1-8b-instant`) with a structured system prompt that grounds answers only in provided data.

---

## Dependencies

### Core (required)

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.5.0 | ML framework backbone |
| `torchvision` | ≥0.20.0 | Vision models + transforms |
| `ultralytics` | ≥8.2.0 | YOLO detection + tracking |
| `opencv-python` | ≥4.9.0 | Video I/O + display |
| `numpy` | ≥1.26.0 | Numerical arrays |
| `Pillow` | ≥10.0.0 | Image processing |
| `mediapipe` | ≥0.10.0 | Face mesh, hands, pose |
| `onnxruntime` | ≥1.17.0 | ONNX model inference |
| `networkx` | ≥3.3 | Knowledge graph engine |
| `pydantic` | ≥2.0.0 | Data validation (PerceptionPacket) |
| `chromadb` | ≥0.5.0 | Vector database |
| `sentence-transformers` | ≥2.7.0 | Text embeddings (BGE-M3/All-MiniLM) |
| `deepface` | ≥0.0.79 | Face recognition |
| `pyyaml` | ≥6.0 | Configuration parsing |

### VLM / LLM

| Package | Version | Purpose |
|---|---|---|
| `transformers` | ≥4.41.0 | Florence-2 VLM loading |
| `accelerate` | ≥0.30.0 | Optimized VLM inference |
| `einops` | ≥0.8.0 | Tensor operations |
| `timm` | ≥0.9.0 | Vision model utilities |
| `tokenizers` | ≥0.19.0 | LLM tokenization |
| `groq` | ≥0.9.0 | Groq cloud LLM API |
| `open_clip_torch` | ≥2.24.0 | CLIP for re-ID |

### Cognee / Knowledge

| Package | Version | Purpose |
|---|---|---|
| `cognee` | ≥1.0.0 | Graph RAG framework |
| `python-louvain` | ≥0.16 | Graph community detection |
| `asyncio-throttle` | ≥1.0 | API rate limiting |

### UI

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.35.0 | Chat UI + Dashboard |
| `requests` | ≥2.32.0 | HTTP client (webhooks) |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |

---

## Configuration

All settings are managed in `config/settings.py` and can be overridden via environment variables (`.env` file). Here are the main categories:

<details>
<summary><b>Camera & Input</b> (click to expand)</summary>

```bash
ARGUS_CAMERA_URL=0              # Webcam index or RTSP URL
ARGUS_CAMERA_INDEX=0            # Alternate camera index
ARGUS_CAMERA_WIDTH=1280
ARGUS_CAMERA_HEIGHT=720
ARGUS_CAMERA_FPS=30
ARGUS_VIDEO_FILE=               # Path to specific video file
```

</details>

<details>
<summary><b>Detection & Gating</b></summary>

```bash
ARGUS_YOLO_MODEL=yolo11n.pt
ARGUS_YOLO_CONFIDENCE=0.35
ARGUS_GATE_ENABLED=true
ARGUS_GATE_MODEL=yolov8n.pt
ARGUS_GATE_CONFIDENCE=0.3
ARGUS_TRACK_PERSIST=30          # ByteTrack persistence frames
ARGUS_DETECT_ALL_OBJECTS=true
```

</details>

<details>
<summary><b>Perception Modules</b></summary>

```bash
ARGUS_POSE_ENABLED=true
ARGUS_ACTION_ENABLED=true
ARGUS_GAZE_ENABLED=true
ARGUS_DEPTH_ENABLED=true
ARGUS_FLOW_ENABLED=true
ARGUS_HAND_ENABLED=true
ARGUS_SEG_ENABLED=true
ARGUS_CONTACT_ENABLED=true
ARGUS_AUDIO_ENABLED=false        # Whisper audio transcription
ARGUS_CAUSAL_ENABLED=true        # Causal event extraction
```

</details>

<details>
<summary><b>VLM (Layer 2)</b></summary>

```bash
ARGUS_VLM_ENABLED=true
ARGUS_VLM_MODEL=microsoft/Florence-2-base
ARGUS_VLM_DEVICE=cpu
ARGUS_VLM_MAX_SIZE=512
ARGUS_VLM_MAX_CALLS=30           # Rate limit per minute
ARGUS_VLM_REFRESH_INTERVAL=3.0   # Seconds between VLM requests per person
```

</details>

<details>
<summary><b>LLM (Layer 3)</b></summary>

```bash
ARGUS_LLM_ENABLED=true
ARGUS_LLM_MODEL_PATH=            # Path to GGUF model (auto-search if empty)
ARGUS_LLM_CONTEXT=8192
ARGUS_LLM_THREADS=4
ARGUS_LLM_GPU_LAYERS=0           # CPU-only by default
GROQ_API_KEY=                    # Set for Groq cloud fallback
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_API_KEY=                  # Optional Gemini fallback
```

</details>

<details>
<summary><b>Cognee Graph RAG</b></summary>

```bash
ARGUS_COGNEE_ENABLED=true
COGNEE_LLM_PROVIDER=openai       # Provider for cognee's own LLM calls
COGNEE_LLM_MODEL=groq/llama-3.1-8b-instant
```

</details>

<details>
<summary><b>Vector Store</b></summary>

```bash
ARGUS_CHROMA_COLLECTION=argus_events
ARGUS_VSS_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARGUS_VSS_THRESHOLD=0.75
ARGUS_VSS_MAX_EMBEDDINGS=10000
```

</details>

<details>
<summary><b>Alerts & Notifications</b></summary>

```bash
ARGUS_ALERT_THROTTLE=10.0        # Seconds between same-type alerts
ARGUS_ALERT_DEDUP=30.0           # Seconds to suppress duplicate alerts
ARGUS_SLACK_WEBHOOK=             # Slack webhook URL
ARGUS_DISCORD_WEBHOOK=           # Discord webhook URL
```

</details>

---

## Memory Architecture

ARGUS implements a **three-tier memory system** inspired by human cognition:

```
┌──────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                    │
├───────────────┬──────────────────┬───────────────────────┤
│  SHORT-TERM   │   MEDIUM-TERM    │     LONG-TERM         │
│  (SQLite)     │   (CogneeBridge) │    (Graph + ChromaDB) │
├───────────────┼──────────────────┼───────────────────────┤
│ Recent 10     │ Event stream     │ Full knowledge graph  │
│ events        │ (JSONL)          │ + vector embeddings   │
│               │                  │ + daily summaries     │
│ Temporal      │ Structured       │ Semantic search       │
│ recency       │ timeline         │ + graph patterns      │
├───────────────┼──────────────────┼───────────────────────┤
│ "What just    │ "What happened   │ "What have we learned │
│  happened?"   │  in sequence?"   │  about Person_3?"     │
└───────────────┴──────────────────┴───────────────────────┘
```

---

## Testing

```bash
# Run integration smoke test
python integration_test.py

# Expected output:
#   OK: ARGUS initialized
#   OK: Graph stats: ...
#   OK: Vector count: ...
#   OK: Events: ...
#   OK: Alert type: ...
#   OK: VSS loaded, matches: ...
#   OK: Double-stop idempotent
#   ALL CHECKS PASSED
```

---

## Performance Notes

- **CPU-only mode** works out of the box. A modern 8-core CPU can process ~5-15 FPS depending on enabled modules.
- **Gating is critical**: YOLOv8n gate typically skips 60-80% of frames, dramatically improving throughput.
- **Turbo mode** (`--turbo`) disables VLM + LLM entirely and runs purely on heuristics — ideal for high-FPS streams or resource-constrained environments.
- **VLM is the bottleneck**: Florence-2 runs at ~2-8 seconds per inference on CPU. The `VLMTriggerManager` aggressively gates it to maintain real-time performance.
- **Multithreading**: Layer 1 runs 6 workers in parallel. VLM has its own async worker thread with a bounded queue.
- **CUDA support**: Set `ARGUS_CV_DEVICE=cuda` and `ARGUS_VLM_DEVICE=cuda` for GPU acceleration (requires CUDA PyTorch).

---

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/tusharsingh-spring/cognee).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with PyTorch, cognee, Florence-2, and Streamlit</sub>
</p>
