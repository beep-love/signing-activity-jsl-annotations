from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

TIERS = ('LEFT_HUMAN_ANNOTATION', 'RIGHT_HUMAN_ANNOTATION')
SIDES = ('left', 'right')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Already exists: {path}; use a new output path.')
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    fd, tmp_name = tempfile.mkstemp(prefix='.jsl-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        if overwrite:
            os.replace(tmp_name, path)
        else:
            os.link(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_config(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('schema_version') != 1:
        raise ValueError('Unsupported config schema_version.')
    for key in ('annotations_root', 'work_dir'):
        if not cfg.get(key):
            raise ValueError(f'Missing config field: {key}')
        p = Path(cfg[key]).expanduser()
        if not p.is_absolute():
            p = path.parent / p
        cfg[key] = str(p.resolve())
    records = cfg.get('records', [])
    ids = [r['recording_id'] for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('records must have distinct recording_id values.')
    for rec in records:
        rid = rec['recording_id']
        if not rid or Path(rid).name != rid or rid in ('.', '..'):
            raise ValueError('recording_id must be a file-safe basename.')
        for key in ('video_path', 'annotation_csv'):
            p = Path(rec[key]).expanduser()
            if not p.is_absolute():
                p = path.parent / p
            rec[key] = str(p.resolve())
    cfg['_config_path'] = str(path)
    return cfg


def select_records(cfg: dict, ids: list[str] | None) -> list[dict]:
    if not ids:
        return cfg['records']
    wanted = set(ids)
    found = {r['recording_id'] for r in cfg['records']}
    if wanted - found:
        raise ValueError(f'Unknown recording IDs: {sorted(wanted - found)}')
    return [r for r in cfg['records'] if r['recording_id'] in wanted]


def csv_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot write an empty report.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
