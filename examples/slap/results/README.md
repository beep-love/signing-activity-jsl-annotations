# SLAP example reference results

[日本語](README_ja.md)

Reference results for the [Lexical Signing Activity Projection models](../weights/README.md) trained on the [Public DGS Corpus](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start_en.html) and applied to the six JSL Signing Activity recordings. No additional training or threshold tuning was performed on JSL for any model.
The reported results were computed with Python 3.10, PyTorch 2.3.0, NumPy 1.23.5 and einops 0.7.0.

## Results

| Input / reference | Projection NLL ↓ | SHIFT/HOLD bal-acc ↑ | SHIFT/HOLD macro-F1 ↑ | SHIFT-pred bal-acc ↑ | SHIFT-pred macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hands | 5.4644 | 0.4353 | 0.4327 | 0.4603 | 0.4476 |
| Eyes | 4.1968 | 0.4527 | 0.4487 | 0.3135 | 0.3127 |
| Mouth | 5.6246 | 0.4691 | 0.4689 | 0.4040 | 0.3902 |
| Hands + Eyes + Mouth | 6.3997 | 0.4029 | 0.3896 | 0.3730 | 0.3592 |
| Uniform (256 states) | 5.5452 | — | — | — | — |
| Always HOLD / no-shift | — | 0.5000 | 0.3310 | 0.5000 | 0.3158 |

This example uses Projection NLL as its primary metric. It is the mean of the six per-recording mean NLL values.

SHIFT/HOLD and SHIFT-pred are supplementary proxy tasks based on Voice Activity Projection (VAP).
These event metrics are computed by summing the confusion matrices for frames within the selected intervals across all recordings.
See [protocol.json](protocol.json) for the extraction conditions and evaluation settings.

The constant event baseline shows the results of always predicting HOLD / no-shift.

## Supplementary event evaluation counts

| Recording | Saved windows | SHIFT / HOLD intervals | SHIFT/HOLD frames | SHIFT-pred positive / negative intervals | SHIFT-pred frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ani1 | 14 | 0 / 0 | 0 | 0 / 0 | 0 |
| Cur1 | 16 | 4 / 4 | 702 | 4 / 4 | 120 |
| Pro1 | 15 | 0 / 0 | 0 | 0 / 0 | 0 |
| Ani2 | 21 | 1 / 1 | 57 | 1 / 1 | 30 |
| Cur2 | 24 | 0 / 0 | 0 | 0 / 0 | 0 |
| Pro2 | 19 | 2 / 1 | 94 | 2 / 1 | 45 |
| Total | 109 | 7 / 6 | 853 | 7 / 6 | 195 |

Event metrics are supplementary reference values.
Ani1, Pro1 and Cur2 contain no eligible events under these extraction conditions, so their event metrics are N/A.
