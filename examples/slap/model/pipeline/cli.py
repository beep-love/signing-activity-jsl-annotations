from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .common import load_config, select_records, write_json
from ..source import ModelSource, PACKAGE_ROOT, VARIANTS, descriptor_path


def config_args(parser):
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recording', action='append', help='Exact recording ID; may be repeated.')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='SLAP: local JSL video → pose → frozen activity projection.')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('init', help='Create a local config; review signer crops before full extraction.')
    q.add_argument('--annotations-root', type=Path, default=PACKAGE_ROOT.parents[1])
    q.add_argument('--video-dir', type=Path)
    q.add_argument('--work-dir', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--recording', action='append')
    q.add_argument('--openpose-binary', type=Path, required=True)
    q.add_argument('--model-folder', type=Path, required=True)
    q = sub.add_parser('inspect', help='Verify tensor-only weights and print the model configuration.')
    q.add_argument('--variant', choices=VARIANTS, default='eyes')
    q.add_argument('--weights-dir', type=Path)
    q = sub.add_parser('extract', help='Run local OpenPose on the two signer crops.')
    config_args(q)
    q.add_argument('--max-seconds', type=float)
    q.add_argument('--resume', action='store_true')
    q.add_argument('--remove-crops', action='store_true')
    q = sub.add_parser('pack', help='Build normalized 411-D features and 30-Hz activity labels.')
    config_args(q)
    q.add_argument('--resume', action='store_true')
    q = sub.add_parser('evaluate', help='Evaluate one frozen variant. Output directory must be new.')
    config_args(q)
    q.add_argument('--variant', choices=VARIANTS, default='eyes')
    q.add_argument('--weights-dir', type=Path)
    q.add_argument('--output-dir', type=Path, required=True)
    q.add_argument('--device', default=None)
    q.add_argument('--batch-size', type=int)
    q.add_argument('--max-windows', type=int)
    q.add_argument('--save-predictions', action='store_true')
    a = p.parse_args(argv)
    try:
        if a.command == 'init':
            from .media import init_config
            ar = a.annotations_root.expanduser().resolve()
            init_config(ar, a.video_dir or ar / 'movies', a.work_dir, a.output, a.recording)
            cfg = json.loads(a.output.read_text(encoding='utf-8'))
            cfg['openpose']['binary'] = str(a.openpose_binary.expanduser().resolve())
            cfg['openpose']['model_folder'] = str(a.model_folder.expanduser().resolve())
            write_json(a.output, cfg, overwrite=True)
            print(f'Created {a.output}; review roi_xywh and analysis_ranges_ms.')
            return 0
        if a.command == 'inspect':
            _, info = ModelSource().model_from_checkpoint(descriptor_path(a.variant, a.weights_dir))
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0
        cfg = load_config(a.config)
        recs = select_records(cfg, a.recording)
        if a.command == 'extract':
            from .media import extract_record
            for rec in recs:
                if not rec.get('roi_reviewed'):
                    print(f'WARNING: {rec["recording_id"]}: ROI not marked as reviewed.', file=sys.stderr)
                report = extract_record(cfg, rec, max_seconds=a.max_seconds, resume=a.resume, remove_crops=a.remove_crops)
                print(f'Extracted {rec["recording_id"]}: {report["num_frames"]} source frames per side.')
        elif a.command == 'pack':
            from .features import pack_record
            source = ModelSource()
            for rec in recs:
                print(json.dumps(pack_record(cfg, rec, source, resume=a.resume), ensure_ascii=False, indent=2))
        else:
            from .evaluate import evaluate
            result = evaluate(cfg, descriptor_path(a.variant, a.weights_dir), a.output_dir,
                              ids=a.recording, device=a.device, batch_size=a.batch_size,
                              max_windows=a.max_windows, save_predictions=a.save_predictions)
            print(json.dumps(result['aggregate'], indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR [{type(exc).__name__}]: {exc}', file=sys.stderr)
        return 1
