"""Pydantic models for structured perception data flowing between layers.

Layer 1 outputs → Layer 2 receives as context → Layer 3 uses for reasoning.
All models are JSON-serializable for seamless inter-layer communication.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


def _ndarray_to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, dict):
        return {k: _ndarray_to_list(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_ndarray_to_list(i) for i in v]
    return v


def _serialize(obj):
    return json.dumps(obj, default=str, indent=2)


# ── Perception Results ──


class PersonEntry(BaseModel):
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"

    @field_validator("bbox", mode="before")
    @classmethod
    def _coerce_bbox(cls, v):
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return tuple(float(x) for x in v)


class ObjectEntry(BaseModel):
    class_id: int
    name: str
    confidence: float
    bbox: Tuple[float, float, float, float]


class PoseResult(BaseModel):
    track_id: int
    keypoints: List[List[float]]  # [[x, y, conf], ...] 17 joints
    visible_count: int = 0

    @classmethod
    def from_ndarray(cls, tid: int, kpts: np.ndarray) -> "PoseResult":
        kpts_list = kpts.tolist()
        visible = sum(1 for k in kpts_list if len(k) > 2 and k[2] > 0.5)
        return cls(track_id=tid, keypoints=kpts_list, visible_count=visible)


class ActionResult(BaseModel):
    track_id: int
    action: str
    confidence: float
    top3: List[Tuple[str, float]] = []
    source: str = "stgcn"
    timestamp: float = 0.0


class GazeResult(BaseModel):
    track_id: int
    yaw: float = 0.0
    pitch: float = 0.0
    direction: str = "center"
    target_person_id: Optional[int] = None


class DepthInfo(BaseModel):
    track_id: int
    mean_depth: float = 0.0
    torso_depth: float = 0.0
    min_depth: float = 0.0
    max_depth: float = 0.0


class FlowInfo(BaseModel):
    track_id: int
    mean_magnitude: float = 0.0
    max_magnitude: float = 0.0
    direction_degrees: float = 0.0


class ContactInfo(BaseModel):
    person_a: int
    person_b: int
    contact: bool = False
    score: float = 0.0
    evidence: List[str] = []
    iou_score: float = 0.0
    joint_distance_px: float = 999.0
    depth_diff_mm: float = 999.0
    flow_correlation: float = 0.0


class HandInfo(BaseModel):
    track_id: int
    handedness: str = "unknown"
    state: str = "neutral"
    landmarks: List[List[float]] = []


# ── The Perception Packet (passed between all layers) ──


class PerceptionPacket(BaseModel):
    """Unified perception output from Layer 1. Passed to VLM (Layer 2) and LLM (Layer 3)."""

    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    frame_number: int = 0
    frame_width: int = 0
    frame_height: int = 0

    persons: List[PersonEntry] = []
    objects: List[ObjectEntry] = []

    poses: Dict[int, PoseResult] = {}
    actions: Dict[int, ActionResult] = {}
    gaze: Dict[int, GazeResult] = {}
    depth: Dict[int, DepthInfo] = {}
    flow: Dict[int, FlowInfo] = {}
    contacts: List[ContactInfo] = []
    hands: Dict[int, List[HandInfo]] = {}

    person_count: int = 0
    object_count: int = 0
    contact_count: int = 0

    model_errors: Dict[str, str] = {}

    def model_post_init(self, __context: Any) -> None:
        self.person_count = len(self.persons)
        self.object_count = len(self.objects)
        self.contact_count = sum(1 for c in self.contacts if c.contact)

    def to_json(self) -> str:
        return _serialize(self.model_dump())

    def to_context_string(self) -> str:
        """Format as compact context string for VLM/LLM prompts."""
        lines = []

        lines.append(f"Frame {self.frame_number} | Time: {datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')}")

        if self.persons:
            lines.append(f"\nPERSONS ({self.person_count}):")
            for p in self.persons:
                parts = [f"  Person_{p.track_id} (conf={p.confidence:.2f})"]
                if p.track_id in self.actions:
                    act = self.actions[p.track_id]
                    parts.append(f"→ action: {act.action} ({act.confidence:.2f})")
                if p.track_id in self.poses:
                    parts.append(f"→ pose: {self.poses[p.track_id].visible_count}/17 joints visible")
                if p.track_id in self.gaze:
                    g = self.gaze[p.track_id]
                    gaze_str = f"→ gaze: {g.direction} (yaw={g.yaw:.0f})"
                    if g.target_person_id is not None:
                        gaze_str += f" looking at Person_{g.target_person_id}"
                    parts.append(gaze_str)
                if p.track_id in self.depth:
                    parts.append(f"→ depth: {self.depth[p.track_id].torso_depth:.2f}")
                if p.track_id in self.flow:
                    parts.append(f"→ flow: {self.flow[p.track_id].mean_magnitude:.2f}px/frame")
                lines.append(" | ".join(parts))

        if self.objects:
            obj_names = [o.name for o in self.objects]
            lines.append(f"\nOBJECTS ({self.object_count}): {', '.join(set(obj_names))}")

        if self.contacts:
            contacts = [c for c in self.contacts if c.contact]
            if contacts:
                lines.append(f"\nCONTACTS ({len(contacts)}):")
                for c in contacts:
                    lines.append(f"  Person_{c.person_a} ↔ Person_{c.person_b} (score={c.score:.2f}, {', '.join(c.evidence)})")

        if self.hands:
            for tid, hands_list in self.hands.items():
                for h in hands_list:
                    lines.append(f"  Person_{tid} {h.handedness} hand: {h.state}")

        return "\n".join(lines)

    def get_person_summary(self, track_id: int) -> str:
        """Get a single person's full data as text."""
        parts = [f"Person_{track_id}:"]
        if track_id in self.actions:
            parts.append(f"  action: {self.actions[track_id].action} (conf={self.actions[track_id].confidence:.2f})")
        if track_id in self.poses:
            parts.append(f"  pose: {self.poses[track_id].visible_count}/17 joints")
        if track_id in self.gaze:
            parts.append(f"  gaze: {self.gaze[track_id].direction}")
        if track_id in self.depth:
            parts.append(f"  depth: {self.depth[track_id].torso_depth:.2f}")
        if track_id in self.flow:
            parts.append(f"  flow: {self.flow[track_id].mean_magnitude:.2f}px/frame")
        return "\n".join(parts)


# ── VLM Output ──


class VLMOutput(BaseModel):
    track_id: int
    task: str = "dense"
    visual_context: str = ""
    emotional_cues: str = ""
    additional_objects: str = ""
    scene_description: str = ""
    raw_caption: str = ""
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


# ── LLM Reasoning Output ──


class LLMReasoning(BaseModel):
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    narrative: str = ""
    intent: str = ""
    is_normal: bool = True
    anomaly_score: float = 0.0
    linked_events: List[str] = []
    pattern_detected: Optional[str] = None
    notify: bool = False
    urgency: str = "none"
    notification_text: Optional[str] = None
    store_tags: List[str] = []
    reasoning_raw: str = ""


# ── Enriched Context Packet (Layer 2 + Layer 1 combined) ──


class EnrichedContext(BaseModel):
    """Combined Layer 1 + Layer 2 output, ready for Layer 3 reasoning."""

    perception: PerceptionPacket
    vlm_outputs: Dict[int, VLMOutput] = {}
    full_scene_vlm: str = ""
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    def to_context_string(self) -> str:
        lines = [self.perception.to_context_string()]
        if self.vlm_outputs:
            lines.append("\nVLM VISUAL CONTEXT:")
            for tid, vlm in self.vlm_outputs.items():
                lines.append(f"  Person_{tid}: {vlm.visual_context[:200]}")
                if vlm.emotional_cues:
                    lines.append(f"    Emotional: {vlm.emotional_cues}")
        if self.full_scene_vlm:
            lines.append(f"\nFULL SCENE: {self.full_scene_vlm[:300]}")
        return "\n".join(lines)
