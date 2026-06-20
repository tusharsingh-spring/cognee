"""Action recognition using ST-GCN on buffered pose sequences (skeleton-based)."""

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
    ACTION_RECOG_CONFIDENCE,
    ACTION_RECOG_DEVICE,
    ACTION_RECOG_ENABLED,
    ACTION_RECOG_MODEL,
    ACTION_RECOG_STRIDE,
    ACTION_RECOG_WINDOW,
    CV_DEVICE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ACTION_LABELS = [
    "standing", "walking", "running", "sitting", "falling",
    "reaching", "grabbing", "pushing", "pulling", "waving",
    "crouching", "jumping", "turning", "looking",
]

SKELETON_EDGES_STGCN = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class ActionRecognizer:
    def __init__(self) -> None:
        self._enabled = ACTION_RECOG_ENABLED
        self._device = CV_DEVICE
        self._window = ACTION_RECOG_WINDOW
        self._stride = ACTION_RECOG_STRIDE
        self._confidence = ACTION_RECOG_CONFIDENCE
        self._model = None
        self._loaded = False
        self._labels = DEFAULT_ACTION_LABELS

        self._pose_buffers: Dict[int, deque] = {}
        self._last_results: Dict[int, Dict] = {}
        self._frame_counter = 0

        if self._enabled:
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
            if CV_DEVICE == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            self._model.eval()
            self._loaded = True
            logger.info(f"[Action] ST-GCN loaded ({len(self._labels)} classes)")
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False
            logger.warning(f"[Action] Model load failed: {e}")
            self._try_heuristic_fallback()

    def _try_heuristic_fallback(self) -> None:
        self._loaded = True
        self._use_heuristic = True
        logger.info("[Action] Using heuristic action classifier fallback")

    @property
    def is_ready(self) -> bool:
        return self._enabled and self._loaded

    @property
    def labels(self) -> List[str]:
        return self._labels

    def update(
        self, track_id: int, keypoints: np.ndarray, frame_time: float
    ) -> Optional[Dict]:
        if not self.is_ready:
            return None

        if track_id not in self._pose_buffers:
            self._pose_buffers[track_id] = deque(maxlen=self._window * 2)

        entry = {
            "keypoints": keypoints.copy(),
            "timestamp": frame_time,
        }
        self._pose_buffers[track_id].append(entry)

        buf = self._pose_buffers[track_id]
        if len(buf) < self._window:
            return self._last_results.get(track_id)

        recent = list(buf)[-self._window:]
        sequence = np.stack([e["keypoints"] for e in recent], axis=0)

        if getattr(self, "_use_heuristic", False):
            result = self._heuristic_classify(track_id, sequence, frame_time)
        else:
            result = self._model_classify(track_id, sequence, frame_time)

        if result:
            self._last_results[track_id] = result
        return result if result else self._last_results.get(track_id)

    def _model_classify(
        self, track_id: int, sequence: np.ndarray, frame_time: float
    ) -> Optional[Dict]:
        try:
            import torch

            seq_t = torch.from_numpy(sequence).float().unsqueeze(0).permute(0, 3, 1, 2)
            if CV_DEVICE == "cuda" and torch.cuda.is_available():
                seq_t = seq_t.cuda()

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
        except Exception as e:
            logger.debug(f"[Action] Model classify failed: {e}")
            return self._heuristic_classify(track_id, sequence, frame_time)

    def _heuristic_classify(
        self, track_id: int, sequence: np.ndarray, frame_time: float
    ) -> Dict:
        n_frames = sequence.shape[0]
        if n_frames < 2:
            return {
                "track_id": track_id,
                "action": "standing",
                "confidence": 0.5,
                "timestamp": frame_time,
                "source": "heuristic",
            }

        torso_indices = [5, 6, 11, 12]
        valid_mask = sequence[:, torso_indices, 2] > 0.3
        if not valid_mask.any():
            return {
                "track_id": track_id,
                "action": "standing",
                "confidence": 0.3,
                "timestamp": frame_time,
                "source": "heuristic",
            }

        torso_pts = sequence[:, torso_indices, :2]
        torso_valid = valid_mask[:, :, np.newaxis]
        masked = np.where(torso_valid, torso_pts, 0)
        center = masked.sum(axis=1) / (valid_mask.sum(axis=1, keepdims=True) + 1e-8)

        velocities = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1)
        mean_vel = float(np.mean(velocities)) if len(velocities) > 0 else 0.0

        v_recent = float(np.mean(velocities[-5:])) if len(velocities) >= 5 else mean_vel

        head_idx = [0, 1, 2, 3, 4]
        head_valid = sequence[:, head_idx, 2] > 0.3
        head_y = sequence[:, head_idx, 1]

        if head_valid.any():
            head_mean_y = float(head_y[head_valid].mean())
            if n_frames >= self._window // 2:
                early_head_y = float(head_y[:4][head_valid[:4]].mean()) if head_valid[:4].any() else head_mean_y
                late_head_y = float(head_y[-4:][head_valid[-4:]].mean()) if head_valid[-4:].any() else head_mean_y
                y_drop = early_head_y - late_head_y
            else:
                y_drop = 0.0
        else:
            head_mean_y = 0.0
            y_drop = 0.0

        max_frame_dim = max(sequence.shape[1], 1)
        rel_vel = v_recent / max_frame_dim if max_frame_dim > 0 else 0.0

        if y_drop > 50 and v_recent > 5:
            action = "falling"
            conf = 0.75
        elif v_recent > 15:
            action = "running"
            conf = 0.7
        elif v_recent > 8:
            action = "walking"
            conf = 0.65
        elif v_recent > 3:
            action = "walking"
            conf = 0.5
        elif v_recent > 0.5:
            action = "standing"
            conf = 0.4
        else:
            action = "standing"
            conf = 0.5

        return {
            "track_id": track_id,
            "action": action,
            "confidence": round(conf, 3),
            "velocity": round(mean_vel, 2),
            "timestamp": frame_time,
            "source": "heuristic",
        }

    def get_action_history(self, track_id: int, n: int = 10) -> List[Dict]:
        buf = self._pose_buffers.get(track_id, deque())
        if not buf:
            return []

        history = []
        step = max(1, len(buf) // n)
        for i in range(0, len(buf), step):
            entry = list(buf)[i]
            history.append({
                "timestamp": entry["timestamp"],
                "keypoints_shape": entry["keypoints"].shape,
            })
        return history[-n:]

    def get_all_actions(self) -> Dict[int, Dict]:
        return dict(self._last_results)


class STGCNLight(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_class: int = 14,
        edge_importance_weighting: bool = True,
    ):
        super(STGCNLight, self).__init__()

        self.data_bn = nn.BatchNorm1d(in_channels * 17)

        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, 17, stride=1, residual=False),
            STGCNBlock(64, 64, 17, stride=1),
            STGCNBlock(64, 64, 17, stride=1),
            STGCNBlock(64, 128, 17, stride=2),
            STGCNBlock(128, 128, 17, stride=1),
            STGCNBlock(128, 256, 17, stride=2),
            STGCNBlock(256, 256, 17, stride=1),
        ])

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.st_gcn_networks[i].A.size()))
                for i in range(len(self.st_gcn_networks))
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        for i, gcn in enumerate(self.st_gcn_networks):
            x, _ = gcn(x, self.edge_importance[i])

        x = nn.functional.avg_pool2d(x, (1, V))
        x = self.fcn(x)
        x = x.view(N, -1)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1, residual=True):
        super(STGCNBlock, self).__init__()

        self.gcn = GCN(in_channels, out_channels, num_nodes)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
        )

        if not residual or in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.A = self._get_adjacency(num_nodes)

    def _get_adjacency(self, num_nodes):
        import torch
        A = np.zeros((num_nodes, num_nodes))
        for i, j in SKELETON_EDGES_STGCN:
            A[i, j] = 1
            A[j, i] = 1
        np.fill_diagonal(A, 1)
        return torch.tensor(A, dtype=torch.float32)

    def forward(self, x, edge_importance):
        x_gcn = self.gcn(x, self.A.to(x.device) * edge_importance.to(x.device))
        x_tcn = self.tcn(x_gcn)
        x_out = self.residual(x) + x_tcn
        return self.relu(x_out), self.A


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, A):
        x = torch.einsum("nctv,vw->nctw", (x, A))
        return self.conv(x)
