"""Bundled source adapter for the unchanged local feature/evaluation pipeline."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import network, preprocessing, utils
from .pipeline.common import sha256

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('hands', 'eyes', 'mouth', 'hands_eyes_mouth')
MODALITIES = dict(zip(VARIANTS, ('hands', 'eyes', 'mouth', 'hands,eyes,mouth')))
ORIGINAL_PREPARATION_SHA256 = 'ecb8aa38642018cf79479e2c3ab39fa20c02fb549237db5f81d63336be6a52a3'


def descriptor_path(variant: str, weights_dir: Path | None = None) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    return Path(weights_dir or PACKAGE_ROOT / 'weights').expanduser().resolve() / f'{variant}.json'


def read_descriptor(path: Path, *, require_weights: bool = True) -> tuple[dict, Path]:
    path = Path(path).expanduser().resolve()
    if path.suffix == '.pt':
        path = path.with_suffix('.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or data.get('variant') not in VARIANTS:
        raise ValueError(f'Unsupported weight descriptor: {path}')
    fname = data.get('weights_file', '')
    if not fname or Path(fname).name != fname or not fname.endswith('.pt'):
        raise ValueError('weights_file must be a .pt basename beside its descriptor.')
    weights = path.parent / fname
    conf = data['model_config']
    fields = {f.name for f in dataclasses.fields(network.VapConfig)}
    if set(conf) != fields:
        raise ValueError(f'Incomplete/unknown model settings: missing={fields-set(conf)}, extra={set(conf)-fields}')
    if (conf['encoder_type'] != 'pose' or conf['frame_hz'] != 30 or conf['pose_dim_total'] != 411
            or conf['bin_times'] != [0.2, 0.4, 0.6, 0.8] or conf['pose_fusion'] != 'concat'
            or conf['pose_modalities'] != MODALITIES[data['variant']]
            or conf['lid_classify'] != 0 or conf['lang_cond'] != 0):
        raise ValueError('Descriptor is not one of the supplied 30-Hz / 411-D pose profiles.')
    if require_weights:
        if not weights.is_file() or not data.get('weights_sha256'):
            raise FileNotFoundError(f'Weights or their SHA256 are missing: {weights}. Check the distributed weights and corresponding JSON descriptor.')
        if sha256(weights) != data['weights_sha256']:
            raise ValueError(f'Weight SHA256 mismatch: {weights}')
    return data, weights


def check_codebook(model) -> None:
    expected = torch.tensor([[(i >> b) & 1 for b in range(8)] for i in range(256)], dtype=torch.float32)
    if not torch.equal(model.objective.codebook.emb.weight.detach().cpu(), expected):
        raise ValueError('Checkpoint codebook differs from the reference bit ordering.')


class ModelSource:
    """No external SLAP directory, training scripts or pickled configs are imported."""
    def __init__(self):
        self.root = PACKAGE_ROOT
        self.model_module = network
        self.prep = preprocessing
        self.utils = utils
        self.hashes = {str(p.relative_to(PACKAGE_ROOT)): sha256(p)
                       for p in sorted((PACKAGE_ROOT / 'model').rglob('*.py'))}
        self.preparation_sha256 = sha256(Path(preprocessing.__file__))

    def accepts_preparation(self, digest: str) -> bool:
        # The old full-file hash is accepted only because all three called
        # preparation functions are verbatim extracts and regression-tested.
        # New caches record the actual bundled preprocessing-file hash.
        return digest in (self.preparation_sha256, ORIGINAL_PREPARATION_SHA256)

    def feature_vector(self, person: dict | None, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if width <= 0 or height <= 0:
            raise ValueError('Image normalization requires positive width and height.')
        if person is None:
            return np.zeros(411, np.float32), np.zeros(137, np.uint8)
        blocks = [('pose_keypoints_2d', 25), ('hand_left_keypoints_2d', 21),
                  ('hand_right_keypoints_2d', 21), ('face_keypoints_2d', 70)]
        parts, masks = [], []
        for key, n in blocks:
            arr = person.get(key, [])
            if arr and len(arr) != n * 3:
                raise ValueError(f'{key}: expected {n * 3} values or an empty block, got {len(arr)}. '
                                 'Use BODY_25 + both hands + face70; do not substitute a different skeleton.')
            pts, mask = self.prep._extract_keypoints_2d(person, key, n_points=n)
            pts[:, 0] /= width
            pts[:, 1] /= height
            parts.append(pts.astype(np.float32).reshape(-1))
            masks.append(mask.astype(np.uint8))
        x = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32), np.concatenate(masks)

    def labels(self, segments_ms: list[tuple[int, int]], duration_ms: int, hz: int,
               mode: str) -> np.ndarray:
        if mode == 'legacy_floor_ceil':
            return self.prep._segments_to_frame_labels(segments_ms, duration_ms, hz)
        if mode == 'point_sample':
            t = np.arange(math.ceil(duration_ms / 1000 * hz), dtype=np.float64) / hz * 1000
            y = np.zeros(len(t), np.uint8)
            for s, e in segments_ms:
                y[(t >= s) & (t < e)] = 1
            return y
        raise ValueError(f'Unknown label_raster: {mode}')

    def window_labels(self, ya: np.ndarray, yb: np.ndarray, s: int, n: int,
                      hz: int, mode: str) -> torch.Tensor:
        if s < 0 or s + n > len(ya) or len(ya) != len(yb):
            raise ValueError('Insufficient true future labels. Padding is not permitted.')
        if mode == 'maai_roundtrip':
            segs = [self.prep._labels_to_segments_rel_sec(y, hz, s, s + n, prec=3)
                    for y in (ya, yb)]
            out = self.utils.vad_list_to_onehot(segs, duration=n / hz, frame_hz=hz)
            if out.shape != (n, 2):
                raise ValueError('Legacy seconds-to-frames yielded an unexpected length.')
            return out
        if mode == 'direct':
            return torch.from_numpy(np.stack((ya[s:s+n], yb[s:s+n]), axis=-1)).float()
        raise ValueError(f'Unknown label_window: {mode}')

    def model_from_checkpoint(self, path: Path, *, trust: bool = False,
                              model_config: dict | None = None):
        if trust:
            raise ValueError('Public inference does not accept trusted legacy pickles; use the distributed weights and corresponding JSON descriptor.')
        data, weights = read_descriptor(path)
        conf_data = data['model_config']
        if model_config is not None and model_config != conf_data:
            raise ValueError('A supplied model config differs from the weight descriptor.')
        blob = torch.load(weights, map_location='cpu', weights_only=True)
        if not isinstance(blob, dict) or not blob or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                          for k, v in blob.items()):
            raise TypeError('Public weights must be a nonempty tensor-only state_dict.')
        model = network.VapGPT(network.VapConfig(**conf_data))
        model.load_state_dict(blob, strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('Non-finite model state.')
        check_codebook(model)
        model.eval()
        info = {'checkpoint_sha256': sha256(weights), 'config_source': 'public JSON descriptor',
                'config': dataclasses.asdict(model.conf), 'strict_state_dict_load': True,
                'source_sha256': self.hashes, 'bin_frames': model.objective.bin_frames,
                'n_parameters': sum(p.numel() for p in model.parameters()),
                'weight_format': 'tensor_only_state_dict', 'variant': data['variant'],
                'original_checkpoint_sha256': data['source_checkpoint']['sha256']}
        return model, info
