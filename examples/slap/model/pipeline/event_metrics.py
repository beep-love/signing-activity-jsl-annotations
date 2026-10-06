#!/usr/bin/env python3
"""Saved SLAP probabilities → the established VAP-style proxy event metrics.
All windows are kept by default; source numerical functions are preserved.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Local evaluation package, supplied earlier. This file is an additive entry point.
try:
    from .common import sha256, write_json, write_csv
    from ..source import ModelSource, descriptor_path, PACKAGE_ROOT, VARIANTS, MODALITIES
    from .evaluate import plan_windows
except ImportError as exc:
    raise SystemExit('The bundled model/pipeline package is incomplete. '
                     f'Original import error: {exc}') from exc

TASKS = {
    'hs': ('SHIFT/HOLD', 'frame', 'hold', 'shift', 'p_now'),
    'pred_shift': ('SHIFT-pred', 'frame', 'no-shift', 'shift', 'p_future'),
    'hs2': ('SHIFT/HOLD', 'event_mean', 'hold', 'shift', 'p_now'),
    'pred_shift2': ('SHIFT-pred', 'event_mean', 'no-shift', 'shift', 'p_future'),
}
EVENT_KEYS = ('shift', 'hold', 'pred_shift', 'pred_shift_neg')
ALL_EVENT_KEYS = EVENT_KEYS + ('long', 'short', 'pred_backchannel', 'pred_backchannel_neg')
RECORDINGS = ('ani1', 'cur1', 'pro1', 'ani2', 'cur2', 'pro2')
EXPECTED_EVENT_SHA256 = '1c113b370ba508a1c84586d3d31dd22217a33d2c6be1fb9822976ee5f8afdfd2'  # Set when packaging; unmodified uploaded events.py.


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return data


def config_dict(value: Any, description: str) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise ValueError(f'Missing/unsupported {description}; it is not inferred from filenames.')


def binary_metrics(cm: np.ndarray) -> dict:
    """Rows=truth, columns=prediction. Undefined two-class results are NA."""
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (2, 2) or (cm < 0).any():
        raise ValueError('Expected a nonnegative 2x2 confusion matrix.')
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    n0, n1 = tn + fp, fn + tp
    n = n0 + n1
    recall0 = tn / n0 if n0 else None
    recall1 = tp / n1 if n1 else None
    f0 = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    status = 'ok' if n0 and n1 else ('no_events' if not n else 'one_class_only')
    # Raw compatibility fields show the older TorchMetrics zero-division convention.
    # Never use these fields to disguise missing classes as a valid two-class score.
    return {
        'status': status, 'n_class0': n0, 'n_class1': n1, 'n_samples': n,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'balanced_accuracy': (recall0 + recall1) / 2 if status == 'ok' else None,
        'macro_f1': (f0 + f1) / 2 if status == 'ok' else None,
        'weighted_f1': (n0 * f0 + n1 * f1) / n if status == 'ok' else None,
        'accuracy': (tn + tp) / n if n else None,
        'recall_class0': recall0, 'recall_class1': recall1,
        'legacy_zero_division_balanced_accuracy': ((recall0 or 0.0) + (recall1 or 0.0)) / 2 if n else None,
        'legacy_zero_division_macro_f1': (f0 + f1) / 2 if n else None,
        'legacy_zero_division_weighted_f1': (n0 * f0 + n1 * f1) / n if n else None,
    }


def confusion(scores: torch.Tensor | None, target: torch.Tensor | None) -> np.ndarray:
    if scores is None or target is None:
        if scores is not None or target is not None:
            raise ValueError('Only one of scores/targets is missing.')
        return np.zeros((2, 2), dtype=np.int64)
    p = scores.detach().cpu().reshape(-1)
    t = target.detach().cpu().long().reshape(-1)
    if p.shape != t.shape or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid original event scores.')
    if ((t < 0) | (t > 1)).any():
        raise ValueError('Event labels must be binary.')
    # Deliberately preserve train_hands.py / eval_macro_from_summary.py rounding.
    # torch.round(0.5) == 0, unlike a >=0.5 classifier.
    pred = p.round().long()
    return torch.bincount(2 * t + pred, minlength=4).reshape(2, 2).numpy()


def counts_for(rows: list[dict]) -> dict:
    out = {}
    for key in EVENT_KEYS:
        rs = [r for r in rows if r['event_type'] == key]
        out[f'{key}_n_selected_events'] = len(rs)
        out[f'{key}_n_nonempty_scored_events'] = sum(r['n_scored_frames'] > 0 for r in rs)
        out[f'{key}_n_scored_frames'] = sum(r['n_scored_frames'] for r in rs)
    return out


def original_sampled_plan(events_module, event_conf, windows: list[dict],
                          batch_size: int, sampling_seed: int, drop_last: bool):
    """Call the source extractor, including its carry-over and unused BC RNG draws."""
    usable = (len(windows) // batch_size) * batch_size if drop_last else len(windows)
    included, dropped = windows[:usable], windows[usable:]
    extractor = events_module.TurnTakingEvents(event_conf)
    state = random.getstate()
    random.seed(sampling_seed)
    result, rows, raw_rows = {}, [], []
    try:
        for offset in range(0, len(included), batch_size):
            block = included[offset:offset + batch_size]
            vad = torch.stack([w['vad'] for w in block])  # 20 s + TRUE 2 s labels
            raw = extractor.HS(vad)  # deterministic, before balancing
            events = extractor(vad)  # unchanged source code; samples HOLD/negatives
            for b, w in enumerate(block):
                key = (w['tag'], w['start'])
                result[key] = {name: [tuple(map(int, r)) for r in events[name][b]]
                               for name in EVENT_KEYS}
                raw_rows.append({
                    'tag': w['tag'], 'recording_id': w['recording_id'],
                    'window_start_frame': w['start'], 'batch_index': offset // batch_size,
                    'raw_shift': len(raw['shift'][b]), 'raw_hold': len(raw['hold'][b]),
                    'raw_pred_shift': len(raw['pred_shift'][b]), 'raw_pred_hold': len(raw['pred_hold'][b]),
                    **{f'selected_{name}': len(result[key][name]) for name in EVENT_KEYS},
                })
                for name in EVENT_KEYS:
                    for s, e, speaker in result[key][name]:
                        if not (0 <= s < e <= len(w['vad']) and speaker in (0, 1)):
                            raise ValueError(f'Invalid original event interval: {name}, {(s, e, speaker)}')
                        # Native tensor slices stop at available logits (20 s).
                        cs, ce = min(s, w['width']), min(e, w['width'])
                        rows.append({
                            'tag': w['tag'], 'recording_id': w['recording_id'],
                            'window_start_frame': w['start'], 'batch_index': offset // batch_size,
                            'event_type': name, 'target': int(name in ('shift', 'pred_shift')),
                            'next_speaker': speaker, 'next_speaker_screen_side': ('LEFT', 'RIGHT')[speaker],
                            'local_start_frame': s, 'local_end_frame': e,
                            'global_start_frame': w['start'] + s, 'global_end_frame': w['start'] + e,
                            'global_start_seconds': (w['start'] + s) / w['hz'],
                            'global_end_seconds': (w['start'] + e) / w['hz'],
                            'n_scored_frames': max(0, ce - cs), 'right_clipped_to_window': e > w['width'],
                        })
    finally:
        random.setstate(state)
    return result, rows, raw_rows, dropped, dict(extractor.add_extra)


def original_scores(objective, probability: np.ndarray, selection: dict):
    p = torch.from_numpy(np.ascontiguousarray(probability, dtype=np.float32))[None]
    # The saved array IS probabilities. Do not apply softmax a second time.
    now = objective.probs_next_speaker_aggregate(p, from_bin=0, to_bin=1)
    fut = objective.probs_next_speaker_aggregate(p, from_bin=2, to_bin=3)
    events = {name: [[]] for name in ALL_EVENT_KEYS}
    for name in EVENT_KEYS:
        events[name] = [selection[name]]
    # BC/long/short were included during sampling to preserve the random sequence,
    # but are not scored/reported here. HS/SP are passed unchanged.
    predictions, targets = objective.extract_prediction_and_targets(now, fut, events, device='cpu')
    return predictions, targets


def read_variant_checkpoint(source, path, trust, expected, events_module):
    from ..source import read_descriptor
    model, info = source.model_from_checkpoint(path)
    desc, weights = read_descriptor(path)
    config = desc['model_config']
    if config['pose_modalities'] != expected:
        raise ValueError('Descriptor modality differs from the requested variant.')
    fields = {f.name for f in dataclasses.fields(events_module.EventConfig)}
    if set(desc['event_config']) != fields:
        raise ValueError('Event configuration fields differ from the bundled implementation.')
    ev = events_module.EventConfig(**desc['event_config'])
    meta = {'path': str(weights), 'sha256': info['checkpoint_sha256'],
            'model_config': config, 'event_config': dataclasses.asdict(ev),
            'source_checkpoint': desc['source_checkpoint']}
    return model.objective, ev, meta


def validate_run(path: Path, model_info: dict, legacy: ModelSource, objective,
                 expected_id: str, cached: dict | None = None):
    summary = read_json(path / 'summary.json')
    if summary.get('schema_version') != 1 or summary.get('evaluation_type') != 'frozen_DGS_checkpoint_on_JSL':
        raise ValueError(f'Unsupported prediction summary: {path}')
    prior = summary['model']
    if prior['checkpoint_sha256'] != model_info['sha256'] or not prior['strict_state_dict_load']:
        raise ValueError(f'Checkpoint/strict-loading provenance differs: {path}')
    if prior['config'] != model_info['model_config']:
        raise ValueError(f'Model config differs: {path}')
    for source, digest in legacy.hashes.items():
        if prior['source_sha256'].get(source) != digest:
            raise ValueError(f'Original source changed since predictions: {source} ({path})')
    cfg = summary['configuration']
    opts = summary['actual_evaluation_options']
    if (opts['window_frames'], opts['horizon_frames'], opts['warmup_frames']) != (600, 60, 0):
        raise ValueError('Requires complete 600-frame windows, 60 future labels, warmup=0; '
                         'rerun normal evaluate with --save-predictions if necessary.')
    if opts.get('max_windows') is not None:
        raise ValueError(f'Smoke/limited predictions are not a full example evaluation: {path}')
    if len(summary['recordings']) != 1 or len(summary['diagnostics']) != 1:
        raise ValueError('Use one recording per saved run, as produced by scripts/evaluate_all.py.')
    result_row = summary['recordings'][0]
    rid = result_row['recording_id']
    if rid != expected_id:
        raise ValueError(f'Wrong recording: expected {expected_id}, got {rid}')
    recs = [r for r in cfg['records'] if r['recording_id'] == rid]
    if len(recs) != 1:
        raise ValueError('Missing/duplicate recording in saved run config.')
    rec = recs[0]
    diag = summary['diagnostics'][0]
    if diag['partial_extraction'] or diag['planned_windows'] != diag['evaluated_windows']:
        raise ValueError(f'Incomplete source extraction/evaluation: {path}')
    if cfg['frame_hz'] != 30:
        raise ValueError('Source path/frame rate differs from saved predictions.')
    feature_path = Path(cfg['work_dir']) / 'features' / f'{rid}.npz'
    fp = read_json(feature_path.with_suffix('.json'))
    if fp['feature_sha256'] != diag['features_sha256'] or fp['partial_extraction']:
        raise ValueError('Feature cache provenance differs/incomplete.')
    protocol = {
        'recording_id': rid, 'features_sha256': fp['feature_sha256'],
        'feature_path': str(feature_path.resolve()), 'frame_hz': cfg['frame_hz'],
        'resampling': cfg['resampling'], 'label_raster': cfg['label_raster'],
        'label_window': cfg['label_window'], 'analysis_ranges_ms': rec['analysis_ranges_ms'],
        'roi_xywh': rec['roi_xywh'], 'n_windows': result_row['n_windows'],
        'n_scored_frames': result_row['n_scored_frames'],
    }
    if cached is not None and protocol != cached['protocol']:
        raise ValueError(f'Different input/timeline across modalities: {rid}')
    if cached is None:
        if sha256(feature_path) != fp['feature_sha256']:
            raise ValueError('Modified feature cache.')
        fpr = fp['fingerprint']
        if (fpr['annotation_sha256'] != sha256(Path(rec['annotation_csv']))
                or fpr['resampling'] != cfg['resampling'] or fpr['label_raster'] != cfg['label_raster']
                or fpr['frame_hz'] != 30
                or not legacy.accepts_preparation(fpr['preparation_sha256'])
                or fp['roi_xywh'] != rec['roi_xywh'] or fp['source_video_path'] != rec['video_path']):
            raise ValueError('Stale annotation/ROI/preparation cache.')
        with np.load(feature_path, allow_pickle=False) as z:
            ya, yb = z['y_a'], z['y_b']
        if len(ya) != len(yb) or not np.isin(ya, [0, 1]).all() or not np.isin(yb, [0, 1]).all():
            raise ValueError('Malformed source activity labels.')
        starts = plan_windows(rec['analysis_ranges_ms'], len(ya), 30, 600, 60)
        if len(starts) != result_row['n_windows'] or len(starts) * 600 != result_row['n_scored_frames']:
            raise ValueError('Source labels do not reconstruct all saved windows.')
        vads = {s: legacy.window_labels(ya, yb, s, 660, 30, cfg['label_window']) for s in starts}
        targets = {s: objective.get_labels(vads[s][None])[0].numpy() for s in starts}
        cached = {'protocol': protocol, 'starts': starts, 'vads': vads, 'targets': targets,
                  'roi_reviewed': bool(rec.get('roi_reviewed', False)),
                  'analysis_ranges_reviewed': bool(rec.get('analysis_ranges_reviewed', False)),
                  'max_positive_resampling_lookahead_ms': fp.get('max_positive_resampling_lookahead_ms')}
    prediction_path = path / f'{rid}.predictions.npz'
    with np.load(prediction_path, allow_pickle=False) as z:
        keys = ['frame_index', 'window_start_frame', 'projection_probabilities', 'target_state', 'target_activity']
        if any(k not in z for k in keys):
            raise ValueError(f'Incomplete prediction NPZ: {prediction_path}')
        pred = {k: z[k] for k in keys}
    starts = cached['starts']
    expected_frames = np.concatenate([np.arange(s, s + 600, dtype=np.int64) for s in starts])
    expected_starts = np.repeat(np.asarray(starts, np.int64), 600)
    if not np.array_equal(pred['frame_index'], expected_frames) or not np.array_equal(pred['window_start_frame'], expected_starts):
        raise ValueError('Missing/duplicate/reordered prediction frames.')
    p = pred['projection_probabilities']
    if p.shape != (len(starts) * 600, 256) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid projection probabilities.')
    if not np.allclose(p.sum(-1), 1, atol=2e-5, rtol=0):
        raise ValueError('Saved arrays are not normalized probabilities.')
    expected_target = np.concatenate([cached['targets'][s] for s in starts])
    expected_activity = np.concatenate([cached['vads'][s][:600].numpy() for s in starts])
    if not np.array_equal(pred['target_state'], expected_target) or not np.array_equal(pred['target_activity'], expected_activity):
        raise ValueError(f'Reconstructed labels differ from saved evaluation targets: {rid}')
    meta = {'prediction_file': str(prediction_path.resolve()), 'prediction_sha256': sha256(prediction_path),
            'summary_file': str((path / 'summary.json').resolve()), 'summary_sha256': sha256(path / 'summary.json')}
    return cached, p, result_row, meta


def run(args) -> dict:
    outdir = Path(args.output_dir).expanduser().resolve()
    if outdir.exists():
        raise FileExistsError(f'Output already exists: {outdir}. Choose a fresh name; nothing is overwritten.')
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Batch size and CPU threads must be positive.')
    torch.set_num_threads(args.threads)
    root = Path(args.root).expanduser().resolve()
    if args.manifest:
        spec = read_json(Path(args.manifest).expanduser().resolve())
    else:
        protocol = read_json(PACKAGE_ROOT / 'results/protocol.json')
        seen = set(); tags = []
        for w in protocol['sampling']['window_order']:
            if w['tag'] not in seen:
                seen.add(w['tag']); tags.append({'tag': w['tag'], 'recording_id': w['recording_id']})
        spec = {'schema_version': 1,
                'variants': [{'name': v, 'pose_modalities': MODALITIES[v],
                              'checkpoint': str(descriptor_path(v, args.weights_dir)),
                              'prediction_root': v} for v in args.variants],
                'recordings': tags}
    if spec.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version.')
    variants = spec['variants']
    tags = spec['recordings']
    if not variants or not tags:
        raise ValueError('Empty variant/recording manifest.')
    if len({v['name'] for v in variants}) != len(variants) or len({r['tag'] for r in tags}) != len(tags):
        raise ValueError('Duplicate variants or recording tags.')
    for obj in variants + tags:
        name = obj.get('name', obj.get('tag'))
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid name/tag in manifest.')
    legacy = ModelSource()
    from .. import events as em
    metas, objectives, reference_event = {}, {}, None
    for item in variants:
        cp = Path(item['checkpoint']).expanduser()
        cp = (root / cp).resolve() if not cp.is_absolute() else cp.resolve()
        obj, event_conf, meta = read_variant_checkpoint(legacy, cp, False,
                                                      item['pose_modalities'], em)
        if reference_event is None:
            reference_event = event_conf
        elif dataclasses.asdict(event_conf) != dataclasses.asdict(reference_event):
            raise ValueError('EventConfig differs across checkpoints. Do not compare different task definitions.')
        metas[item['name']], objectives[item['name']] = meta, obj
    cached, predictions_meta, base_metrics = {}, {}, {}
    reference = variants[0]
    windows = []
    # Build exactly one event plan, before looking at any model probabilities.
    for rec in tags:
        path = root / reference['prediction_root'] / rec['tag']
        data, _, row, pm = validate_run(path, metas[reference['name']], legacy,
                                       objectives[reference['name']], rec['recording_id'])
        cached[rec['tag']] = data
        for s in data['starts']:
            windows.append({'tag': rec['tag'], 'recording_id': rec['recording_id'], 'start': s,
                            'width': 600, 'hz': 30, 'vad': data['vads'][s]})
    selection, event_rows, raw_rows, dropped, debt = original_sampled_plan(
        em, reference_event, windows, args.batch_size, args.sampling_seed, not args.keep_last_batch)
    if not selection:
        raise ValueError('No full event batch; use enough saved windows or omit --drop-last.')
    plan_hash = canonical_hash(event_rows)
    print(f'Event plan: {len(selection)}/{len(windows)} windows, dropped={len(dropped)}, '
          f'event rows={len(event_rows)}, SHA256={plan_hash}', flush=True)
    metrics_rows, cm_map = [], {}
    for item in variants:
        name = item['name']
        print(f'=== {name} ===', flush=True)
        predictions_meta[name], base_metrics[name] = [], []
        cm_map[name] = {}
        for rec in tags:
            tag = rec['tag']
            path = root / item['prediction_root'] / tag
            data, p, base, pm = validate_run(path, metas[name], legacy, objectives[name],
                                           rec['recording_id'], cached[tag])
            predictions_meta[name].append(pm)
            base_metrics[name].append(base)
            cms = {task: np.zeros((2, 2), dtype=np.int64) for task in TASKS}
            for j, s in enumerate(data['starts']):
                if (tag, s) not in selection:
                    continue
                pred, target = original_scores(objectives[name], p[j*600:(j+1)*600], selection[(tag, s)])
                for task in TASKS:
                    cms[task] += confusion(pred[task], target[task])
            cm_map[name][tag] = cms
            for task, cm in cms.items():
                m = binary_metrics(cm)
                metrics_rows.append({'variant': name, 'tag': tag, 'recording_id': rec['recording_id'],
                                     'task_key': task, 'task': TASKS[task][0], 'unit': TASKS[task][1], **m})
            h, sp = binary_metrics(cms['hs']), binary_metrics(cms['pred_shift'])
            print(f'  {tag}: hs={h["n_class0"]}/{h["n_class1"]} frames, '
                  f'sp={sp["n_class0"]}/{sp["n_class1"]} frames', flush=True)
            del p
        for task in TASKS:
            cm = sum((cm_map[name][r['tag']][task] for r in tags), np.zeros((2, 2), dtype=np.int64))
            metrics_rows.append({'variant': name, 'tag': 'ALL_POOLED', 'recording_id': 'ALL_POOLED',
                                 'task_key': task, 'task': TASKS[task][0], 'unit': TASKS[task][1], **binary_metrics(cm)})
    # Constants share exactly the same selected target samples, not ideal 0.5/0.333 values.
    for task in TASKS:
        cm = sum((cm_map[reference['name']][r['tag']][task] for r in tags), np.zeros((2, 2), dtype=np.int64))
        constant = np.array([[cm[0].sum(), 0], [cm[1].sum(), 0]], dtype=np.int64)
        metrics_rows.append({'variant': 'constant_hold_no_shift', 'tag': 'ALL_POOLED', 'recording_id': 'ALL_POOLED',
                             'task_key': task, 'task': TASKS[task][0], 'unit': TASKS[task][1], **binary_metrics(constant)})
    count_rows = []
    for rec in tags + [{'tag': 'ALL_POOLED', 'recording_id': 'ALL_POOLED'}]:
        tag = rec['tag']
        er = event_rows if tag == 'ALL_POOLED' else [r for r in event_rows if r['tag'] == tag]
        rr = raw_rows if tag == 'ALL_POOLED' else [r for r in raw_rows if r['tag'] == tag]
        count_rows.append({'tag': tag, 'recording_id': rec['recording_id'], 'n_evaluated_windows': len(rr),
                           **{k: sum(r[k] for r in rr) for k in ('raw_shift', 'raw_hold', 'raw_pred_shift', 'raw_pred_hold')},
                           **counts_for(er)})
    # Across-model target counts must match because the plan and labels were frozen.
    for rec in tags:
        for task in TASKS:
            counts = [cm_map[v['name']][rec['tag']][task].sum(1).tolist() for v in variants]
            if any(c != counts[0] for c in counts):
                raise RuntimeError('Different event target counts across variants.')
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}'
    table = ['# Supplementary VAP-style event evaluation', '',
             'Frame-pooled across the included windows. Balanced accuracy and **macro-F1**.',
             'These are proxy events derived from Signing Activity, not independently annotated turn changes.', '',
             '| Variant | Projection NLL (all saved windows) ↓ | SHIFT/HOLD bal-acc ↑ | SHIFT/HOLD macro-F1 ↑ | SHIFT-pred bal-acc ↑ | SHIFT-pred macro-F1 ↑ |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in [v['name'] for v in variants] + ['constant_hold_no_shift']:
        hs = next(r for r in metrics_rows if r['variant'] == name and r['tag'] == 'ALL_POOLED' and r['task_key'] == 'hs')
        sp = next(r for r in metrics_rows if r['variant'] == name and r['tag'] == 'ALL_POOLED' and r['task_key'] == 'pred_shift')
        nll = (sum(float(r['projection_nll']) for r in base_metrics[name]) / len(tags)) if name in base_metrics else None
        table.append(f'| {name} | {fmt(nll)} | {fmt(hs["balanced_accuracy"])} | {fmt(hs["macro_f1"])} | '
                     f'{fmt(sp["balanced_accuracy"])} | {fmt(sp["macro_f1"])} |')
    table += ['', f'Event sampling seed: {args.sampling_seed}; event batch size: {args.batch_size}; '
                    f'drop_last: {not args.keep_last_batch}; dropped windows: {len(dropped)}.',
              'The NLL column is copied from the existing full saved evaluations; it is NOT recomputed on event-only samples.',
              'See event_counts.csv for candidate/selected intervals and scored frames; N/A denotes missing class support.',
              'metrics.csv also reports weighted-F1 for comparison with older test_f1_* logs, separately from macro-F1.',
              'hs2 / pred_shift2 are event-mean alternatives in metrics.csv, not the primary frame-pooled scores.', '']
    source_hashes = legacy.hashes
    serial_windows = [{k: v for k, v in w.items() if k != 'vad'} for w in windows]
    result = {
        'schema_version': 1, 'status': 'completed', 'evaluation_type': 'supplementary_VAP_style_proxy_events',
        'manifest': spec, 'checkpoints': metas,
        'original_source_sha256': read_json(PACKAGE_ROOT / 'model/SOURCE_MANIFEST.json')['original_source'],
        'packaged_source_sha256': source_hashes,
        'event_config': dataclasses.asdict(reference_event),
        'sampling': {'seed': args.sampling_seed, 'batch_size': args.batch_size, 'drop_last': not args.keep_last_batch,
                     'window_order': serial_windows, 'n_available_windows': len(windows), 'n_used_windows': len(selection),
                     'dropped_windows': [{k: v for k, v in w.items() if k != 'vad'} for w in dropped],
                     'final_sampling_carry': debt},
        'event_manifest_sha256': plan_hash,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                        'cpu_threads': args.threads, 'device': 'cpu'},
        'prediction_sources': predictions_meta,
        'recording_protocols': {tag: {k: v for k, v in data.items() if k not in ('vads', 'targets')}
                               for tag, data in cached.items()},
        'metrics': metrics_rows, 'event_counts': count_rows,
        'notes': [
            'TurnTakingEvents and ObjectiveVAP scoring preserve the supplied implementation; imports are namespaced.',
            'All four variants use exactly the same sampled events. No test-threshold optimization.',
            'Full mutual inactivity intervals are scored for hs. metric_time/pad only determine minimum silence in this source.',
            'pred_shift uses the 0.5 s interval before mutual inactivity and p_future from bins 2/3 (0.6-2.0 s).',
            'This source does NOT implement a general classifier for a shift within the next 600 ms.',
            'Round uses torch.round; exact 0.5 becomes 0. No second softmax is applied to saved probabilities.',
            'The original training test_f1_* fields use weighted-F1; macro-F1 is separately computed here.',
            'Frame samples are correlated, not independent events. Both frame and interval counts are reported.',
            'No-event/one-class primary scores are N/A, not zero. Raw zero-division compatibility fields are separate.',
            'Historical DGS data, RNG history, and paper scores are not reproduced by this JSL postprocessing.',
        ],
    }
    outdir.mkdir(parents=True, exist_ok=False)
    write_csv(outdir / 'metrics.csv', metrics_rows)
    write_csv(outdir / 'event_counts.csv', count_rows)
    write_csv(outdir / 'per_window_event_counts.csv', raw_rows)
    # Event list may be empty; write at least its fixed header in that case.
    if event_rows:
        write_csv(outdir / 'event_manifest.csv', event_rows)
    else:
        (outdir / 'event_manifest.csv').write_text('tag,recording_id,window_start_frame,event_type,target\n', encoding='utf-8')
    write_json(outdir / 'event_manifest.json', {'events': event_rows, 'sha256': plan_hash})
    (outdir / 'reference_table.md').write_text('\n'.join(table), encoding='utf-8')
    write_json(outdir / 'summary.json', result)  # completion marker written last
    print('\n' + '\n'.join(table), flush=True)
    print(f'Wrote results to {outdir}', flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description='Supplementary event metrics from saved example predictions. No inference.')
    p.add_argument('--manifest', type=Path, help='Optional explicit variant/recording manifest.')
    p.add_argument('--prediction-root', dest='root', type=Path, required=True,
                   help='Contains <variant>/<tag>/summary.json and predictions NPZ.')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--weights-dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sampling-seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--drop-last', action='store_true', help='Legacy alternative; reference result uses all saved windows.')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args(argv)
    args.keep_last_batch = not args.drop_last
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR [{type(exc).__name__}]: {exc}', file=sys.stderr)
        return 1
    return 0
