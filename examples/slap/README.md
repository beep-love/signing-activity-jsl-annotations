# Signing Activity Projection example

[日本語](README_ja.md)

A pose-based Signing Activity Projection (SLAP) example.

[Reference results](results/README.md)

## Contents

```text
slap/
  README.md / README_ja.md
  model/       Pose model, preprocessing and evaluation
  weights/     Model weights, JSON configurations and checksums
  scripts/     Inference and evaluation
  results/     Fixed reference results, event list and evaluation settings
```

## Environment

Run all commands below, including installation, from `examples/slap/`.

Model loading and inference have been tested with Python 3.14 and PyTorch 2.10.

```bash
python -m pip install torch==2.10.0
python -m pip install -r requirements.txt
```

FFmpeg/ffprobe and OpenPose (BODY_25, hand and face models) are required separately.
OpenPose itself and its trained models are not included.

## Pretrained weights

`weights/` includes pretrained weights for four variants:
Hands, Eyes, Mouth, and Hands + Eyes + Mouth.
Select the model with `--variant`.

Check that the Eyes model loads:
```bash
python scripts/run.py inspect --variant eyes
```

## Prepare local data

Prepare the corpus videos and a directory for working caches, then specify the
obtained corpus videos and the released additional annotations.
Replace `/path/to/` below with your local paths.

```bash
python scripts/run.py init \
  --annotations-root ../.. \
  --video-dir ../../movies \
  --work-dir /path/to/local/slap_work \
  --output /path/to/local/jsl_config.json \
  --openpose-binary /path/to/openpose/build/examples/openpose/openpose.bin \
  --model-folder /path/to/openpose/models
```

Video names and the left/right signer mapping come from `metadata/recordings.csv`.
The initial ROIs split each video into left and right halves.

For one recording, add an option such as `--recording FO_09_10_AniN` to `init`,
`extract`, `pack` or `evaluate`. The ROIs and analysis ranges used for the
published results are recorded in [results/protocol.json](results/protocol.json).

```bash
python scripts/run.py extract --config /path/to/local/jsl_config.json
python scripts/run.py pack --config /path/to/local/jsl_config.json
```

`extract` crops the left and right ROIs from each original frame and runs OpenPose.
`pack` normalizes coordinates by the ROI width and height and resamples to 30 Hz.

## Projection evaluation

Example evaluation for one model:

```bash
python scripts/run.py evaluate \
  --config /path/to/local/jsl_config.json \
  --variant eyes --device cuda:0 --save-predictions \
  --output-dir /path/to/local/run_eyes
```

Example output for all four models:

```bash
python scripts/evaluate_all.py \
  --config /path/to/local/jsl_config.json \
  --variants hands eyes mouth hands_eyes_mouth \
  --device cuda:0 \
  --output-root /path/to/local/slap_predictions
```

Use `--device cpu` when not using a GPU.
For evaluation, set `--output-dir` to a directory that does not yet exist.

## Supplementary SHIFT/HOLD and SHIFT-pred evaluation

After inference for all six recordings with `evaluate_all.py` is complete, run:

```bash
python scripts/evaluate_events.py \
  --prediction-root /path/to/local/slap_predictions \
  --output-dir /path/to/local/slap_events
```

These are supplementary proxy tasks based on Voice Activity Projection (VAP).
See [results/README.md](results/README.md) for details.

## Terms and sources

See [LICENSE](LICENSE) and [the weight terms](weights/LICENSE_WEIGHTS.md).
