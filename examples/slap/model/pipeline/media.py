from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .common import SIDES, csv_rows, sha256, write_json


def probe_video(path: Path) -> dict:
    """Read per-frame timestamps, not pixels. Keep source frame indices intact."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not shutil.which('ffprobe'):
        raise RuntimeError('ffprobe is required (install FFmpeg).')
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
           '-show_frames', '-show_entries',
           'stream=width,height,avg_frame_rate,r_frame_rate,duration:stream_tags=rotate:'
           'stream_side_data=rotation:frame=best_effort_timestamp_time', '-of', 'json', str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if len(data.get('streams', [])) != 1:
        raise ValueError(f'Expected one selected video stream: {path}')
    stream = data['streams'][0]
    if any(abs(float(s.get('rotation', 0))) > 1e-5 for s in stream.get('side_data_list', [])):
        raise ValueError('Rotation-tagged video: prepare an explicitly oriented local copy and verify timing first.')
    if float(stream.get('tags', {}).get('rotate', 0)) != 0:
        raise ValueError('Rotation-tagged video is not supported without an explicit orientation conversion.')
    frames = data.get('frames', [])
    if len(frames) < 2 or any('best_effort_timestamp_time' not in f for f in frames):
        raise ValueError(f'Need at least two frames with timestamps: {path}')
    pts = np.asarray([float(f['best_effort_timestamp_time']) for f in frames], dtype=np.float64)
    pts -= pts[0]
    delta = np.diff(pts)
    if not np.isfinite(pts).all() or np.any(delta <= 0):
        raise ValueError('Video frame timestamps must be finite and strictly increasing.')
    duration = float(pts[-1] + np.median(delta))
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'num_frames': len(pts), 'timestamps_seconds': pts.tolist(),
            'duration_seconds': duration,
            'avg_frame_rate': stream.get('avg_frame_rate'),
            'r_frame_rate': stream.get('r_frame_rate'),
            'median_frame_interval_seconds': float(np.median(delta)),
            'max_frame_interval_seconds': float(delta.max()),
            'source_file_size': path.stat().st_size,
            'source_mtime_ns': path.stat().st_mtime_ns}


def init_config(annotations_root: Path, video_dir: Path,
                work_dir: Path, output: Path, recording_ids: list[str] | None) -> None:
    ar = annotations_root.expanduser().resolve()
    records = csv_rows(ar / 'metadata' / 'recordings.csv')
    if recording_ids:
        wanted = set(recording_ids)
        if wanted - {r['recording_id'] for r in records}:
            raise ValueError('Unknown recording ID in --recording.')
        records = [r for r in records if r['recording_id'] in wanted]
    out_records = []
    for row in records:
        rid = row['recording_id']
        video = video_dir.expanduser().resolve() / row['corpus_video_filename']
        ann = ar / 'annotations' / 'csv' / (rid + '.csv')
        if not ann.is_file():
            raise FileNotFoundError(ann)
        meta = probe_video(video)
        w, h = meta['width'], meta['height']
        mid = w // 2
        out_records.append({
            'recording_id': rid, 'video_path': str(video), 'annotation_csv': str(ann),
            'left_participant_id': row['left_participant_id'],
            'right_participant_id': row['right_participant_id'],
            'roi_xywh': {'left': [0, 0, mid, h], 'right': [mid, 0, w-mid, h]},
            'roi_reviewed': False,
            'analysis_ranges_ms': [[0, round(meta['duration_seconds'] * 1000)]],
            'analysis_ranges_reviewed': False,
        })
    cfg = {
        'schema_version': 1,
        'annotations_root': str(ar), 'work_dir': str(work_dir.expanduser().resolve()),
        'frame_hz': 30, 'resampling': 'nearest_legacy',
        'label_raster': 'legacy_floor_ceil', 'label_window': 'maai_roundtrip',
        'openpose': {'binary': '/REPLACE/openpose/build/examples/openpose/openpose.bin',
                     'model_folder': '/REPLACE/openpose/models',
                     'num_gpu': 1, 'num_gpu_start': 0, 'net_resolution': '-1x368'},
        'evaluation': {'window_seconds': 20, 'warmup_seconds': 0,
                       'batch_size': 1, 'device': 'cpu',
                       'persistence_confidence': 0.9},
        'records': out_records,
    }
    write_json(output, cfg)


def validate_roi(roi: list, w: int, h: int) -> tuple[int, int, int, int]:
    if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError('roi_xywh must contain four integers [x, y, width, height].')
    x, y, cw, ch = roi
    if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x+cw > w or y+ch > h:
        raise ValueError(f'ROI {roi} is outside image {w}x{h}.')
    return x, y, cw, ch


def index_json_frames(folder: Path) -> dict[int, Path]:
    indexed = {}
    for p in folder.glob('*_keypoints.json'):
        try:
            idx = int(p.stem.rsplit('_', 2)[-2])
        except (ValueError, IndexError) as e:
            raise ValueError(f'Unrecognized OpenPose JSON filename: {p.name}') from e
        if idx in indexed:
            raise ValueError(f'Duplicate OpenPose frame index {idx} in {folder}')
        indexed[idx] = p
    return indexed


def extract_record(cfg: dict, rec: dict, *, max_seconds: float | None = None,
                   resume: bool = False, remove_crops: bool = False) -> dict:
    op = cfg['openpose']
    binary = Path(op['binary']).expanduser().resolve()
    models = Path(op['model_folder']).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f'OpenPose binary is not installed/specified: {binary}')
    if not models.is_dir():
        raise FileNotFoundError(f'OpenPose model folder does not exist: {models}')
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required.')
    source = Path(rec['video_path'])
    src = probe_video(source)
    pts = np.asarray(src['timestamps_seconds'], dtype=np.float64)
    if max_seconds is not None:
        if max_seconds <= 0:
            raise ValueError('--max-seconds must be positive.')
        pts = pts[pts < max_seconds]
    n = len(pts)
    if n < 2:
        raise ValueError('The extraction range has fewer than two frames.')
    work = Path(cfg['work_dir']) / 'openpose' / rec['recording_id']
    fingerprint = {'source': str(source), 'source_size': src['source_file_size'],
                   'source_mtime_ns': src['source_mtime_ns'], 'roi_xywh': rec['roi_xywh'],
                   'frames': n, 'openpose_config': op, 'binary_sha256': sha256(binary)}
    done = work / 'extraction.json'
    if work.exists():
        if not resume or not done.is_file():
            raise FileExistsError(f'{work} exists. Completed runs may use --resume. For an interrupted '
                                  'run, rename/remove ONLY its work/openpose/<recording> directory first.')
        old = json.loads(done.read_text(encoding='utf-8'))
        if old['fingerprint'] != fingerprint:
            raise ValueError('Extraction configuration/input changed. Use a new work_dir (do not reuse stale coordinates).')
        for side in SIDES:
            if set(index_json_frames(work / side)) != set(range(n)):
                raise ValueError('Cached extraction is incomplete; resume refused.')
        return old
    work.mkdir(parents=True)
    report = {'schema_version': 1, 'fingerprint': fingerprint, 'recording_id': rec['recording_id'],
              'source_metadata': src, 'num_frames': n, 'timestamps_seconds': pts.tolist(),
              'duration_seconds': float(pts[-1] + np.median(np.diff(pts))),
              'partial_extraction': n < src['num_frames'],
              'roi_reviewed': bool(rec.get('roi_reviewed', False)), 'sides': {}}
    for side in SIDES:
        roi = validate_roi(rec['roi_xywh'][side], src['width'], src['height'])
        x, y, cw, ch = roi
        crop = work / f'{side}.mkv'
        ff_cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n', '-noautorotate',
                  '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
                  '-vf', f'crop={cw}:{ch}:{x}:{y}:exact=1,setpts=PTS-STARTPTS',
                  '-frames:v', str(n), '-fps_mode', 'passthrough', '-c:v', 'ffv1', str(crop)]
        subprocess.run(ff_cmd, check=True)
        crop_meta = probe_video(crop)
        cp = np.asarray(crop_meta['timestamps_seconds'])
        if len(cp) != n or crop_meta['width'] != cw or crop_meta['height'] != ch:
            raise ValueError('Crop changed frame count/dimensions; extraction stopped.')
        if np.max(np.abs(cp - pts)) > 0.0025:
            raise ValueError('Crop timestamps differ by >2.5 ms; do not assume alignment.')
        dest = work / side
        dest.mkdir()
        command = [str(binary), '--video', str(crop), '--model_folder', str(models) + '/',
                   '--model_pose', 'BODY_25', '--hand', '--face',
                   '--keypoint_scale', '0', '--write_json', str(dest),
                   '--display', '0', '--render_pose', '0', '--face_render', '0', '--hand_render', '0',
                   '--number_people_max', '1', '--hand_detector', '0', '--face_detector', '0',
                   '--num_gpu', str(op.get('num_gpu', 1)), '--num_gpu_start', str(op.get('num_gpu_start', 0)),
                   '--net_resolution', str(op.get('net_resolution', '-1x368'))]
        # OpenPose expects to run from its installation root. model_folder is absolute too.
        cwd = models.parent
        with (work / f'{side}.openpose.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        frame_map = index_json_frames(dest)
        if set(frame_map) != set(range(n)):
            raise ValueError(f'OpenPose output for {side} is incomplete/not zero-based: '
                             f'{len(frame_map)} JSON files for {n} frames. Missing JSON is not a missing detection.')
        report['sides'][side] = {'width': cw, 'height': ch, 'roi_xywh': list(roi),
                                'command': command, 'crop_command': ff_cmd,
                                'max_crop_timestamp_error_seconds': float(np.max(np.abs(cp-pts)))}
        if remove_crops:
            crop.unlink()
    write_json(done, report)
    return report
