from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import SIDES, TIERS, sha256, write_json
from ..source import ModelSource
from .media import index_json_frames


def read_annotation_intervals(path: Path) -> tuple[list, list]:
    data = {t: [] for t in TIERS}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['tier', 'start_ms', 'end_ms', 'label']:
            raise ValueError(f'Unexpected annotation CSV header: {reader.fieldnames}')
        for row in reader:
            if row['tier'] not in data or row['label'] != 'SIGNING':
                raise ValueError('Only public HUMAN tiers and the exact SIGNING label are supported.')
            s, e = int(row['start_ms']), int(row['end_ms'])
            if s < 0 or e <= s:
                raise ValueError(f'Invalid annotation interval {s}, {e}')
            data[row['tier']].append((s, e))
    for tier in TIERS:
        data[tier].sort()
        if any(b[0] < a[1] for a, b in zip(data[tier], data[tier][1:])):
            raise ValueError(f'Overlapping intervals inside {tier}; cross-speaker overlap IS allowed.')
    return data[TIERS[0]], data[TIERS[1]]


def resample_indices(pts: np.ndarray, n_out: int, hz: int, mode: str,
                     nominal_rate: str | None = None) -> np.ndarray:
    q = np.arange(n_out, dtype=np.float64) / hz
    if mode == 'previous':
        return np.maximum(np.searchsorted(pts, q, side='right') - 1, 0)
    if mode != 'nearest_legacy':
        raise ValueError('resampling must be nearest_legacy or previous.')
    # Reproduce np.rint(i * fps_in / fps_out) exactly for constant-rate data.
    # ffprobe may quantize decimal PTS; use the explicit rational stream rate.
    if nominal_rate and nominal_rate not in ('0/0', '0'):
        from fractions import Fraction
        rate = float(Fraction(nominal_rate))
        if rate > 0 and np.max(np.abs(pts - np.arange(len(pts)) / rate)) <= 0.001:
            return np.clip(np.rint(q * rate).astype(np.int64), 0, len(pts)-1)
    right = np.clip(np.searchsorted(pts, q), 0, len(pts)-1)
    left = np.maximum(right-1, 0)
    dl, dr = np.abs(q - pts[left]), np.abs(pts[right] - q)
    # Ties use the even index, matching rint on a regular grid.
    use_right = (dr < dl) | (np.isclose(dl, dr, atol=1e-10, rtol=0) & ((right % 2) == 0))
    return np.where(use_right, right, left)


def pack_record(cfg: dict, rec: dict, legacy: ModelSource, *, resume: bool = False) -> dict:
    work = Path(cfg['work_dir'])
    root = work / 'openpose' / rec['recording_id']
    meta_path = root / 'extraction.json'
    ext = json.loads(meta_path.read_text(encoding='utf-8'))
    source_video = Path(rec['video_path'])
    if str(source_video) != ext['fingerprint']['source']:
        raise ValueError('Configured video path differs from the extraction source.')
    if source_video.exists():
        st = source_video.stat()
        if st.st_size != ext['fingerprint']['source_size'] or st.st_mtime_ns != ext['fingerprint']['source_mtime_ns']:
            raise ValueError('Source video changed after extraction; use a fresh work_dir.')
    if ext['fingerprint']['roi_xywh'] != rec['roi_xywh']:
        raise ValueError('ROIs changed since extraction; regenerate coordinates in a new work_dir.')
    pts = np.asarray(ext['timestamps_seconds'], dtype=np.float64)
    n = ext['num_frames']
    if pts.shape != (n,) or np.any(np.diff(pts) <= 0):
        raise ValueError('Invalid extraction timestamps.')
    hz = int(cfg['frame_hz'])
    if hz != 30:
        raise ValueError('The supplied experimental profile expects 30 Hz.')
    duration_ms = int(math.ceil(ext['duration_seconds'] * 1000))
    # Match the old label builder's output length, using actual source duration,
    # not the last annotation offset as the recording end.
    ya_segs, yb_segs = read_annotation_intervals(Path(rec['annotation_csv']))
    if not ext['partial_extraction']:
        final_end = max((e for segs in (ya_segs, yb_segs) for s, e in segs), default=0)
        if final_end > duration_ms + 2:
            raise ValueError(f'Annotation extends to {final_end} ms but video ends at {duration_ms} ms. '
                             'Confirm matching video and EAF time origin; no automatic time shift is applied.')
    labels = [legacy.labels(segs, duration_ms, hz, cfg['label_raster']) for segs in (ya_segs, yb_segs)]
    n_out = len(labels[0])
    indices = resample_indices(pts, n_out, hz, cfg['resampling'], ext['source_metadata'].get('r_frame_rate'))
    fingerprint = {'extraction_sha256': sha256(meta_path),
                   'annotation_sha256': sha256(Path(rec['annotation_csv'])),
                   'preparation_sha256': legacy.preparation_sha256,
                   'resampling': cfg['resampling'], 'label_raster': cfg['label_raster'], 'frame_hz': hz}
    out = work / 'features' / (rec['recording_id'] + '.npz')
    sidecar = out.with_suffix('.json')
    if out.exists() or sidecar.exists():
        if not resume or not out.is_file() or not sidecar.is_file():
            raise FileExistsError(f'Feature output exists/incomplete: {out}; use a new work_dir.')
        old = json.loads(sidecar.read_text(encoding='utf-8'))
        if old['fingerprint'] != fingerprint:
            raise ValueError('Feature inputs/settings changed; refusing stale cache.')
        if old['feature_sha256'] != sha256(out):
            raise ValueError('Feature cache was modified.')
        return old
    arrays, masks, qc = [], [], {}
    for side in SIDES:
        index = index_json_frames(root / side)
        if set(index) != set(range(n)):
            raise ValueError('Incomplete OpenPose output; refusing to treat missing files as missing detections.')
        x = np.empty((n, 411), np.float32)
        mask = np.empty((n, 137), np.uint8)
        no_person = 0
        for i in range(n):
            obj = json.loads(index[i].read_text(encoding='utf-8'))
            people = obj.get('people')
            if not isinstance(people, list) or len(people) > 1:
                raise ValueError('Expected 0/1 person per explicitly selected signer crop. '
                                 'Multi-person JSON must not be arbitrarily mapped by array order.')
            person = people[0] if people else None
            no_person += person is None
            sm = ext['sides'][side]
            x[i], mask[i] = legacy.feature_vector(person, sm['width'], sm['height'])
        arrays.append(x[indices]); masks.append(mask[indices])
        m = masks[-1]
        qc[side] = {'source_frames_without_person': int(no_person),
                    'source_fraction_without_person': no_person/n,
                    'observed_body_fraction': float(m[:, :25].mean()),
                    'observed_hands_fraction': float(m[:, 25:67].mean()),
                    'observed_eyes_fraction': float(m[:, list(range(103,115)) + [135,136]].mean()),
                    'observed_mouth_fraction': float(m[:,115:135].mean())}
    if not np.any(masks[0]) or not np.any(masks[1]):
        raise ValueError('An entire signer stream has no detected keypoints. Check ROI and OpenPose before evaluation.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as f:
        np.savez_compressed(f, X_a=arrays[0], X_b=arrays[1], M_a=masks[0], M_b=masks[1],
                            y_a=labels[0], y_b=labels[1],
                            duration_ms=np.array([duration_ms], np.int64),
                            fps_out=np.array([hz], np.int64),
                            source_frame_index=indices,
                            frame_time_ms=np.arange(n_out, dtype=np.float64)/hz*1000,
                            source_time_ms=pts[indices]*1000)
    delta = pts[indices] - np.arange(n_out)/hz
    report = {'schema_version': 1, 'recording_id': rec['recording_id'], 'fingerprint': fingerprint,
              'feature_sha256': sha256(out), 'feature_dim': 411, 'frames': n_out,
              'duration_ms': duration_ms, 'partial_extraction': ext['partial_extraction'],
              'source_video_path': rec['video_path'],
              'roi_xywh': rec['roi_xywh'], 'roi_reviewed': bool(rec.get('roi_reviewed', False)),
              'max_positive_resampling_lookahead_ms': float(max(0, delta.max()*1000)),
              'max_absolute_resampling_error_ms': float(np.abs(delta).max()*1000),
              'quality': qc, 'channel_mapping': {'X_a': 'LEFT_HUMAN_ANNOTATION',
                                              'X_b': 'RIGHT_HUMAN_ANNOTATION'}}
    write_json(sidecar, report)
    return report
