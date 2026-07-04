"""Action recognition using ST-GCN on buffered pose sequences.

Model runs on CPU via PyTorch (lightweight). Classifies: standing, walking,
running, sitting, falling, reaching, grabbing, pushing, pulling, waving,
crouching, jumping, turning, looking.

Uses heuristic fallback when ST-GCN is unavailable.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from config.settings import (
    ACTION_CONFIDENCE,
    ACTION_ENABLED,
    ACTION_MODEL,
    ACTION_STRIDE,
    ACTION_WINDOW,
)
from utils.logger import get_logger
from utils.profiler import profiler

logger = get_logger(__name__)

ACTION_LABELS = [
    "standing", "walking", "running", "sitting", "falling",
    "reaching", "grabbing", "pushing", "pulling", "waving",
    "crouching", "jumping", "turning", "looking",
]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class ActionRecognizer:
    def __init__(self) -> None:
        self.enabled = ACTION_ENABLED
        self.is_ready = False
        self._model = None
        self._window = ACTION_WINDOW
        self._stride = ACTION_STRIDE
        self._confidence = ACTION_CONFIDENCE
        self._labels = ACTION_LABELS
        self._use_heuristic = False

        self._pose_buffers: Dict[int, deque] = {}
        self._last_results: Dict[int, Dict] = {}
        self._frame_counter = 0

        if self.enabled:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            import torch.nn as nn

            self._model = STGCNLight(
                in_channels=3,
                num_class=len(self._labels),
                edge_importance_weighting=True,
            )
            self._model.eval()
            self.is_ready = True
            logger.info(f"[ACTION] ST-GCN ready ({len(self._labels)} classes)")
        except Exception as e:
            logger.warning(f"[ACTION] ST-GCN failed: {e}, using heuristic")
            self._use_heuristic = True
            self.is_ready = True

    def update(
        self, track_id: int, keypoints: np.ndarray, frame_time: float
    ) -> Optional[Dict]:
        if not self.is_ready:
            return None

        if track_id not in self._pose_buffers:
            self._pose_buffers[track_id] = deque(maxlen=self._window * 2)

        self._pose_buffers[track_id].append({
            "keypoints": keypoints.copy(),
            "timestamp": frame_time,
        })

        buf = self._pose_buffers[track_id]
        if len(buf) < self._window:
            return self._last_results.get(track_id)

        recent = list(buf)[-self._window:]
        sequence = np.stack([e["keypoints"] for e in recent], axis=0)

        if self._use_heuristic or self._model is None:
            result = self._heuristic_classify(track_id, sequence, frame_time)
        else:
            result = self._model_classify(track_id, sequence, frame_time)

        if result:
            self._last_results[track_id] = result
        return result

    def _model_classify(self, track_id, sequence, frame_time) -> Optional[Dict]:
        try:
            seq_t = torch.from_numpy(sequence).float().unsqueeze(0).permute(0, 3, 1, 2)
            with torch.no_grad():
                output = self._model(seq_t)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])

            if conf < self._confidence:
                return None

            top3_idx = np.argsort(probs)[::-1][:3]
            return {
                "track_id": track_id,
                "action": self._labels[best_idx],
                "confidence": round(conf, 3),
                "top3": [(self._labels[i], round(float(probs[i]), 3)) for i in top3_idx],
                "timestamp": frame_time,
                "source": "stgcn",
            }
        except Exception:
            return self._heuristic_classify(track_id, sequence, frame_time)

    def _heuristic_classify(self, track_id, sequence, frame_time) -> Dict:
        n_frames = sequence.shape[0]
        if n_frames < 2:
            return {"track_id": track_id, "action": "standing", "confidence": 0.5,
                    "timestamp": frame_time, "source": "heuristic"}

        torso_indices = [5, 6, 11, 12]
        valid_mask = sequence[:, torso_indices, 2] > 0.3
        if not valid_mask.any():
            return {"track_id": track_id, "action": "standing", "confidence": 0.3,
                    "timestamp": frame_time, "source": "heuristic"}

        torso_pts = sequence[:, torso_indices, :2]
        torso_valid = valid_mask[:, :, np.newaxis]
        masked = np.where(torso_valid, torso_pts, 0)
        center = masked.sum(axis=1) / (valid_mask.sum(axis=1, keepdims=True) + 1e-8)

        velocities = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1)
        v_recent = float(np.mean(velocities[-5:])) if len(velocities) >= 5 else \
                   float(np.mean(velocities)) if len(velocities) > 0 else 0.0

        head_idx = [0, 1, 2, 3, 4]
        head_valid = sequence[:, head_idx, 2] > 0.3
        if head_valid.any() and n_frames >= self._window // 2:
            head_y = sequence[:, head_idx, 1]
            early = float(head_y[:4][head_valid[:4]].mean()) if head_valid[:4].any() else 0
            late = float(head_y[-4:][head_valid[-4:]].mean()) if head_valid[-4:].any() else 0
            y_drop = early - late
        else:
            y_drop = 0.0

        if y_drop > 50 and v_recent > 5:
            action, conf = "falling", 0.75
        elif v_recent > 15:
            action, conf = "running", 0.7
        elif v_recent > 8:
            action, conf = "walking", 0.65
        elif v_recent > 3:
            action, conf = "walking", 0.5
        elif v_recent > 0.5:
            action, conf = "standing", 0.4
        else:
            action, conf = "standing", 0.5

        return {"track_id": track_id, "action": action,
                "confidence": round(conf, 3), "timestamp": frame_time,
                "source": "heuristic"}

    def get_all_actions(self) -> Dict[int, Dict]:
        return dict(self._last_results)


class STGCNLight(nn.Module):
    def __init__(self, in_channels=3, num_class=14, edge_importance_weighting=True):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, 17, stride=1, residual=False),
            STGCNBlock(64, 64, 17, stride=1),
            STGCNBlock(64, 128, 17, stride=2),
            STGCNBlock(128, 128, 17, stride=1),
            STGCNBlock(128, 256, 17, stride=2),
            STGCNBlock(256, 256, 17, stride=1),
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(g.A.size())) for g in self.st_gcn_networks
        ]) if edge_importance_weighting else [1] * len(self.st_gcn_networks)
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x).view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)
        for i, gcn in enumerate(self.st_gcn_networks):
            x, _ = gcn(x, self.edge_importance[i])
        x = nn.functional.avg_pool2d(x, (1, V))
        return self.fcn(x).view(N, -1)


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, num_nodes, stride=1, residual=True):
        super().__init__()
        self.gcn = GCN(in_c, out_c, num_nodes)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_c),
        )
        self.residual = nn.Identity() if residual and in_c == out_c and stride == 1 else \
                        nn.Sequential(nn.Conv2d(in_c, out_c, (1, 1), (stride, 1)),
                                      nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)
        A = np.zeros((num_nodes, num_nodes))
        for i, j in SKELETON_EDGES:
            A[i, j] = A[j, i] = 1
        np.fill_diagonal(A, 1)
        self.A = torch.tensor(A, dtype=torch.float32)

    def forward(self, x, ei):
        x_g = self.gcn(x, self.A.to(x.device) * ei.to(x.device))
        return self.relu(self.residual(x) + self.tcn(x_g)), self.A


class GCN(nn.Module):
    def __init__(self, in_c, out_c, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x, A):
        return self.conv(torch.einsum("nctv,vw->nctw", (x, A)))
