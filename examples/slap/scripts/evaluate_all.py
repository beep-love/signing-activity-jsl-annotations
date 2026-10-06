#!/usr/bin/env python3
"""Evaluate each configured recording separately, in the published order."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.source import PACKAGE_ROOT, VARIANTS, descriptor_path
from model.pipeline.common import load_config
from model.pipeline.evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--weights-dir', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    try:
        cfg = load_config(a.config)
        protocol = json.loads((PACKAGE_ROOT / 'results/protocol.json').read_text())
        # Frozen tags and order; no label-derived selection or chronological shuffle.
        order = protocol['sampling']['window_order']
        seen = set(); recordings = []
        for w in order:
            if w['tag'] not in seen:
                seen.add(w['tag']); recordings.append((w['tag'], w['recording_id']))
        mapped = {r['recording_id']: r for r in cfg['records']}
        available = [(t, rid) for t, rid in recordings if rid in mapped]
        if not available or len(available) != len(mapped):
            raise ValueError('Config contains IDs outside the six published recordings.')
        for v in a.variants:
            for tag, rid in available:
                if (a.output_root / v / tag).exists():
                    raise FileExistsError(f'Output exists: {a.output_root / v / tag}; use a fresh output-root.')
        for v in a.variants:
            for tag, rid in available:
                one = copy.deepcopy(cfg); one['records'] = [mapped[rid]]
                evaluate(one, descriptor_path(v, a.weights_dir), a.output_root / v / tag,
                         device=a.device, batch_size=1, save_predictions=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR [{type(exc).__name__}]: {exc}', file=sys.stderr); return 1
if __name__ == '__main__':
    raise SystemExit(main())
