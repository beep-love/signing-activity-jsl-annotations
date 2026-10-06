from __future__ import annotations

import dataclasses
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import select_records, sha256, write_csv, write_json
from ..source import ModelSource


def plan_windows(ranges_ms: list[list[float]], total: int, hz: int, width: int, horizon: int) -> list[int]:
    if not ranges_ms:
        raise ValueError('Provide at least one common analysis range.')
    frames = []
    prev_end = -1
    for pair in sorted(ranges_ms):
        if len(pair) != 2:
            raise ValueError('Each analysis range must be [start_ms, end_ms].')
        start, end = map(float, pair)
        if not math.isfinite(start+end) or start < 0 or end <= start or start < prev_end:
            raise ValueError('Analysis ranges must be finite, nonnegative, disjoint intervals.')
        prev_end = end
        s = max(0, int(math.ceil(start / 1000 * hz - 1e-9)))
        e = min(total, int(math.floor(end / 1000 * hz + 1e-9)))
        frames.extend(range(s, e - width - horizon + 1, width))
    return frames


def binary_summary(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray) -> dict:
    result = {}
    for s, side in enumerate(('left', 'right')):
        t, f, n, z = [int(x[s]) for x in (tp,fp,fn,tn)]
        result[f'detection_{side}_f1'] = 2*t / (2*t+f+n) if 2*t+f+n else None
        result[f'detection_{side}_iou'] = t / (t+f+n) if t+f+n else None
        result[f'detection_{side}_accuracy'] = (t+z)/(t+f+n+z)
        result[f'active_fraction_{side}'] = (t+n)/(t+f+n+z)
    return result


def evaluate(cfg: dict, checkpoint: Path, output_dir: Path, ids: list[str] | None = None,
             trust: bool = False, model_config: dict | None = None, device: str | None = None,
             batch_size: int | None = None, max_windows: int | None = None,
             save_predictions: bool = False) -> dict:
    legacy = ModelSource()
    model, model_info = legacy.model_from_checkpoint(checkpoint, trust=trust, model_config=model_config)
    ec = cfg['evaluation']
    device_name = device or ec.get('device', 'cpu')
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu or a matching GPU PyTorch build.')
    dev = torch.device(device_name)
    model.to(dev).eval()
    hz = int(model.conf.frame_hz)
    if hz != int(cfg['frame_hz']):
        raise ValueError('Checkpoint and features have different frame rates.')
    width_float = float(ec.get('window_seconds', 20)) * hz
    width = round(width_float)
    warmup_float = float(ec.get('warmup_seconds', 0)) * hz
    warmup = round(warmup_float)
    if width <= 0 or abs(width-width_float) > 1e-6 or warmup < 0 or warmup >= width or abs(warmup-warmup_float)>1e-6:
        raise ValueError('Window/warmup seconds must represent valid whole frame counts.')
    batch_size = batch_size if batch_size is not None else int(ec.get('batch_size', 1))
    if batch_size < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError('batch_size and max_windows must be positive.')
    horizon = int(model.objective.horizon)
    confidence = float(ec.get('persistence_confidence', 0.9))
    if not (0.5 < confidence < 1):
        raise ValueError('persistence_confidence must lie strictly between 0.5 and 1.')
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f'Result directory already exists: {output_dir}; choose a fresh path.')
    output_dir.mkdir(parents=True)
    records = select_records(cfg, ids)
    codebook = model.objective.codebook.emb.weight.detach()
    rows, details, window_rows = [], [], []
    for rec in records:
        rid = rec['recording_id']
        cache = Path(cfg['work_dir']) / 'features' / f'{rid}.npz'
        sidecar = cache.with_suffix('.json')
        info = json.loads(sidecar.read_text(encoding='utf-8'))
        if info['feature_sha256'] != sha256(cache):
            raise ValueError(f'Modified/stale cache: {cache}')
        expected_fingerprint = info['fingerprint']
        if (expected_fingerprint['annotation_sha256'] != sha256(Path(rec['annotation_csv']))
                or expected_fingerprint['resampling'] != cfg['resampling']
                or expected_fingerprint['label_raster'] != cfg['label_raster']
                or expected_fingerprint['frame_hz'] != hz
                or not legacy.accepts_preparation(expected_fingerprint['preparation_sha256'])):
            raise ValueError(f'Feature config/source changed for {rid}; regenerate instead of scoring stale cache.')
        if info['source_video_path'] != rec['video_path']:
            raise ValueError('Video path changed since feature packing; verify the cache source explicitly.')
        if info['roi_xywh'] != rec['roi_xywh']:
            raise ValueError('ROI changed since feature packing.')
        with np.load(cache, allow_pickle=False) as z:
            xa, xb, ya, yb = [z[k] for k in ('X_a','X_b','y_a','y_b')]
            duration_ms = int(z['duration_ms'][0])
        if xa.shape != xb.shape or xa.shape != (len(ya),411) or len(yb) != len(ya):
            raise ValueError(f'Feature shape mismatch: {rid}')
        if not np.isfinite(xa).all() or not np.isfinite(xb).all():
            raise ValueError('Features contain non-finite values.')
        if not np.isin(ya, [0,1]).all() or not np.isin(yb, [0,1]).all():
            raise ValueError('Labels must be binary.')
        ranges = rec['analysis_ranges_ms']
        if not info['partial_extraction'] and max(e for s,e in ranges) > duration_ms + 2:
            raise ValueError('An analysis range extends beyond the actual recording.')
        starts = plan_windows(ranges, len(ya), hz, width, horizon)
        all_windows = len(starts)
        if max_windows is not None:
            starts = starts[:max_windows]
        if not starts:
            raise ValueError(f'{rid}: no full window + 2-second future labels available.')
        counts = {k: np.zeros(2,np.int64) for k in ('tp','fp','fn','tn')}
        n_frames = 0
        sums = dict(nll=0.0, detection_bce=0.0, oracle_persistence_nll=0.0, correct_state=0.0)
        brier = np.zeros((2,4),np.float64)
        pred = {'frame_index':[], 'projection_probabilities':[], 'target_state':[],
                'detection_probabilities':[], 'target_activity':[], 'window_start_frame':[]}
        roundtrip_changes = 0
        with torch.inference_mode():
            for b in range(0,len(starts),batch_size):
                ss = starts[b:b+batch_size]
                x = torch.from_numpy(np.stack([np.stack((xa[s:s+width],xb[s:s+width])) for s in ss])).to(dev)
                v_cpu = torch.stack([legacy.window_labels(ya,yb,s,width+horizon,hz,cfg['label_window']) for s in ss])
                v = v_cpu.to(dev)
                target = model.objective.get_labels(v)
                out = model(waveform=x)
                if target.shape != (len(ss),width) or out['logits'].shape != (len(ss),width,256):
                    raise ValueError('Original model/target shape mismatch; no output length is silently trimmed.')
                logits = out['logits'][:,warmup:]
                detection = out['vad'][:,warmup:]
                target = target[:,warmup:]
                current = v[:,warmup:width]
                if not torch.isfinite(logits).all() or not torch.isfinite(detection).all():
                    raise ValueError('Non-finite model output.')
                losses = model.objective.loss_vap(logits,target,reduction='none')
                va_losses = F.binary_cross_entropy_with_logits(detection,current,reduction='none').mean(-1)
                prob = logits.softmax(-1)
                dp = detection.sigmoid()
                truth_bits = model.objective.codebook.decode(target)
                marginal = (prob @ codebook).reshape(*target.shape,2,4)
                bit_error = (marginal-truth_bits).square()
                same = truth_bits == current[...,None]
                persist_losses = -(same*math.log(confidence)+(~same)*math.log(1-confidence)).sum((-2,-1))
                this_n = target.numel()
                n_frames += this_n
                sums['nll'] += float(losses.double().sum())
                sums['detection_bce'] += float(va_losses.double().sum())
                sums['oracle_persistence_nll'] += float(persist_losses.double().sum())
                sums['correct_state'] += int((logits.argmax(-1)==target).sum())
                brier += bit_error.double().sum((0,1)).cpu().numpy()
                positive, truth = dp >= 0.5, current.bool()
                for key, mask in [('tp',positive & truth),('fp',positive & ~truth),
                                  ('fn',~positive & truth),('tn',~positive & ~truth)]:
                    counts[key] += mask.sum((0,1)).cpu().numpy()
                for j,s in enumerate(ss):
                    raw = np.stack((ya[s:s+width+horizon],yb[s:s+width+horizon]),axis=-1)
                    roundtrip_changes += int(np.sum(v_cpu[j].numpy()!=raw))
                    window_rows.append({'recording_id':rid,'start_frame':s,'start_seconds':s/hz,
                                        'n_scored_frames':width-warmup,
                                        'projection_nll':float(losses[j].double().mean()),
                                        'detection_bce':float(va_losses[j].double().mean())})
                    if save_predictions:
                        nf = width-warmup
                        pred['frame_index'].append(np.arange(s+warmup,s+width,dtype=np.int64))
                        pred['window_start_frame'].append(np.full(nf,s,dtype=np.int64))
                        pred['projection_probabilities'].append(prob[j].cpu().numpy())
                        pred['target_state'].append(target[j].cpu().numpy())
                        pred['detection_probabilities'].append(dp[j].cpu().numpy())
                        pred['target_activity'].append(current[j].cpu().numpy().astype(np.uint8))
        row = {'recording_id':rid,'n_windows':len(starts),'n_scored_frames':n_frames,
               'projection_nll':sums['nll']/n_frames,
               'uniform_256_nll':math.log(256),
               'oracle_persistence_nll':sums['oracle_persistence_nll']/n_frames,
               'state_accuracy':sums['correct_state']/n_frames,
               'mean_bin_brier':float((brier/n_frames).mean()),
               'detection_bce':sums['detection_bce']/n_frames}
        row.update(binary_summary(**counts))
        for side_i,side in enumerate(('left','right')):
            for k in range(4):
                row[f'brier_{side}_bin{k+1}'] = float(brier[side_i,k]/n_frames)
        rows.append(row)
        details.append({'recording_id':rid,'features_sha256':info['feature_sha256'],
                        'analysis_ranges_ms':ranges,'analysis_ranges_reviewed':bool(rec.get('analysis_ranges_reviewed',False)),
                        'roi_reviewed':bool(rec.get('roi_reviewed',False)),
                        'partial_extraction':info['partial_extraction'], 'planned_windows':all_windows,
                        'evaluated_windows':len(starts),'warmup_frames_each_window':warmup,
                        'label_window_roundtrip_changed_bits':roundtrip_changes,
                        'feature_quality':info['quality'],
                        'max_positive_resampling_lookahead_ms':info['max_positive_resampling_lookahead_ms'],
                        'detection_confusion':{k:v.tolist() for k,v in counts.items()}})
        if save_predictions:
            with (output_dir/f'{rid}.predictions.npz').open('xb') as f:
                np.savez_compressed(f, **{k:np.concatenate(v,axis=0) for k,v in pred.items()})
        print(f'{rid}: NLL={row["projection_nll"]:.6f}, frames={n_frames}, windows={len(starts)}', flush=True)
    n_total = sum(r['n_scored_frames'] for r in rows)
    mean_keys = ['projection_nll','uniform_256_nll','oracle_persistence_nll','mean_bin_brier','detection_bce','state_accuracy']
    aggregate = {'n_recordings':len(rows),'n_scored_frames':n_total}
    for key in mean_keys:
        aggregate[key+'_micro'] = sum(r[key]*r['n_scored_frames'] for r in rows)/n_total
        aggregate[key+'_macro_recording'] = sum(r[key] for r in rows)/len(rows)
    result = {'schema_version':1,'evaluation_type':'frozen_DGS_checkpoint_on_JSL',
              'model':model_info,'configuration':{k:v for k,v in cfg.items() if not k.startswith('_')},
              'actual_evaluation_options':{'device':str(dev),'batch_size':batch_size,'window_frames':width,
                                          'horizon_frames':horizon,'warmup_frames':warmup,'max_windows':max_windows},
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                             'cuda_runtime':torch.version.cuda},
              'aggregate':aggregate,'recordings':rows,'diagnostics':details,
              'notes':[
                  'Projection NLL only; detection loss is separate. Event tasks are scored separately by scripts/evaluate_events.py.',
                  'Only full input+future-label windows; no future padding, no partial last-window padding.',
                  'Oracle persistence uses TRUE current labels and is an unequal-input diagnostic, not a visual competitor.',
                  'DGS vs JSL activity definitions, view geometry and pose extraction differ; this is exploratory external evaluation.',
                  'Nearest resampling may use a sub-frame future timestamp; see recorded lookahead. previous is the causal alternative.',
              ]}
    write_csv(output_dir/'per_recording.csv',rows)
    write_csv(output_dir/'per_window.csv',window_rows)
    write_json(output_dir/'summary.json',result)
    return result
