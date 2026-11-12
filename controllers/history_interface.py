"""Controller C helpers: history-only observation views over Controller B/A.

Does not change physics. Defines what a history-inference model may see at
test time versus privileged training labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque
from collections import deque

import numpy as np


@dataclass
class HistoryFrame:
    x: np.ndarray
    a: np.ndarray
    c_obs: np.ndarray | None = None
    exec_feedback: np.ndarray | None = None  # optional measured command if logged


class HistoryBuffer:
    def __init__(self, length: int = 8):
        self.length = int(length)
        self._buf: Deque[HistoryFrame] = deque(maxlen=self.length)

    def push(self, frame: HistoryFrame) -> None:
        self._buf.append(frame)

    def as_arrays(self) -> dict[str, np.ndarray]:
        if not self._buf:
            raise RuntimeError("empty history")
        xs = np.stack([f.x for f in self._buf])
        acts = np.stack([f.a for f in self._buf])
        out = {"x_hist": xs, "a_hist": acts}
        if self._buf[0].c_obs is not None:
            out["c_obs_hist"] = np.stack([f.c_obs for f in self._buf])
        if self._buf[0].exec_feedback is not None:
            out["u_fb_hist"] = np.stack([f.exec_feedback for f in self._buf])
        return out

    def inference_dict(self) -> dict[str, np.ndarray]:
        """Strip privileged fields — only x/a/(optional feedback)."""
        data = self.as_arrays()
        keep = {k: data[k] for k in ("x_hist", "a_hist") if k in data}
        if "u_fb_hist" in data:
            keep["u_fb_hist"] = data["u_fb_hist"]
        return keep
