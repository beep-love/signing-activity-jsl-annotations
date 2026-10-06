"""Exact keypoint/label functions extracted from prepare_dgs_slap.py.
No corpus download or original corpus content is included.
"""
import math
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

def _segments_to_frame_labels(
    segments_ms: List[Tuple[int, int]], duration_ms: int, fps: int
) -> np.ndarray:
    """Build binary labels at given fps from ms segments."""
    if duration_ms < 0:
        raise ValueError("duration_ms must be non-negative")
    T = int(math.ceil(duration_ms / 1000.0 * fps))
    y = np.zeros((T,), dtype=np.uint8)
    for s_ms, e_ms in segments_ms:
        if e_ms <= s_ms:
            continue
        s = int(math.floor(s_ms / 1000.0 * fps))
        e = int(math.ceil(e_ms / 1000.0 * fps))
        s = max(0, min(T, s))
        e = max(0, min(T, e))
        if e > s:
            y[s:e] = 1
    return y


def _extract_keypoints_2d(
    person: Dict,
    key: str,
    n_points: int,
    select_idxs: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (P,3) from OpenPose person dict.

    Returns:
        pts: (P,3) float64 with NaN for missing x/y when conf==0
        mask: (P,) bool, True if conf>0
    """
    out_points = len(select_idxs) if select_idxs is not None else n_points
    arr = person.get(key)
    if arr is None or arr == []:
        # anonymized model or missing
        pts = np.full((out_points, 3), np.nan, dtype=np.float64)
        mask = np.zeros((out_points,), dtype=bool)
        return pts, mask

    a = np.asarray(arr, dtype=np.float64)
    if a.size % 3 != 0:
        # unexpected
        pts = np.full((out_points, 3), np.nan, dtype=np.float64)
        mask = np.zeros((out_points,), dtype=bool)
        return pts, mask

    pts_all = a.reshape(-1, 3)
    # some OpenPose builds may output fewer points; pad
    if pts_all.shape[0] < n_points:
        pad = np.full((n_points - pts_all.shape[0], 3), np.nan, dtype=np.float64)
        pts_all = np.concatenate([pts_all, pad], axis=0)
    elif pts_all.shape[0] > n_points:
        pts_all = pts_all[:n_points]

    if select_idxs is not None:
        pts = pts_all[np.asarray(select_idxs, dtype=np.int64)]
    else:
        pts = pts_all

    conf = pts[:, 2]
    mask = conf > 0
    # treat conf==0 as missing
    pts[~mask, 0:2] = np.nan
    return pts, mask


def _labels_to_segments_rel_sec(
    y: np.ndarray,
    fps: int,
    start_f: int,
    end_f: int,
    prec: int = 3,
) -> List[List[float]]:
    """Convert frame-wise 0/1 labels into [start,end] segments (seconds) relative to start_f.

    The output is suitable for VAP/MaAI-style CSVs where segments are stored as a nested list.
    """
    start_f = int(max(0, start_f))
    end_f = int(min(int(y.shape[0]), end_f))
    if end_f <= start_f:
        return []
    sub = y[start_f:end_f]
    segs: List[List[float]] = []
    t = 0
    n = int(sub.shape[0])
    inv_fps = 1.0 / float(fps)
    fmt = "{:." + str(int(prec)) + "f}"

    while t < n:
        if sub[t] != 0:
            s0 = t
            t += 1
            while t < n and sub[t] != 0:
                t += 1
            e0 = t
            segs.append([
                float(fmt.format(s0 * inv_fps)),
                float(fmt.format(e0 * inv_fps)),
            ])
        else:
            t += 1
    return segs

