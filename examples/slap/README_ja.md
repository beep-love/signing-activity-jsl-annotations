# Signing Activity Projection 実行例

[English](README.md)

姿勢特徴ベースのSigning Activity Projection（SLAP）実行例です。

[参照結果](results/README_ja.md)

## 構成

```text
slap/
  README.md / README_ja.md
  model/       姿勢モデル・前処理・評価処理
  weights/     モデル重み、JSON設定、チェックサム
  scripts/     実行・評価
  results/     固定した参照結果・イベント一覧・評価条件
```

## 実行環境

以下のコマンドは、インストール手順を含めて`examples/slap/`から実行します。

モデルの読み込みと推論はPython 3.14、PyTorch 2.10で動作確認しています。

```bash
python -m pip install torch==2.10.0
python -m pip install -r requirements.txt
```

FFmpeg／ffprobeとOpenPose（BODY_25・hand・faceモデル）が別途必要です。
OpenPose本体とその学習済みモデルは同梱していません。

## 学習済み重み

`weights/`には、Hands、Eyes、Mouth、Hands + Eyes + Mouthの4種類の学習済み重みが含まれます。
使用するモデルは`--variant`で指定します。

Eyesモデルの読み込み確認：
```bash
python scripts/run.py inspect --variant eyes
```

## ローカルデータの準備

コーパス動画と作業キャッシュ用のディレクトリを用意した上で、取得済みコーパス動画と、公開追加注釈を指定します。
以下の`/path/to/`を手元のパスに置き換えてください。

```bash
python scripts/run.py init \
  --annotations-root ../.. \
  --video-dir ../../movies \
  --work-dir /path/to/local/slap_work \
  --output /path/to/local/jsl_config.json \
  --openpose-binary /path/to/openpose/build/examples/openpose/openpose.bin \
  --model-folder /path/to/openpose/models
```

動画名と左右話者の対応は`metadata/recordings.csv`から読みます。初期ROIは動画の左右二分割です。

1対話だけ処理する場合は、`init`／`extract`／`pack`／`evaluate`に`--recording FO_09_10_AniN`のように指定できます。
掲載結果に用いたROIと分析範囲は[results/protocol.json](results/protocol.json)に記録しています。

```bash
python scripts/run.py extract --config /path/to/local/jsl_config.json
python scripts/run.py pack --config /path/to/local/jsl_config.json
```

`extract`は元の各フレームから左右のROIを切り出してOpenPoseを実行します。
`pack`はROIの幅・高さで座標を正規化し、30 Hzへ変換します。

## Projection評価

1モデル評価例：

```bash
python scripts/run.py evaluate \
  --config /path/to/local/jsl_config.json \
  --variant eyes --device cuda:0 --save-predictions \
  --output-dir /path/to/local/run_eyes
```

4モデル出力例：

```bash
python scripts/evaluate_all.py \
  --config /path/to/local/jsl_config.json \
  --variants hands eyes mouth hands_eyes_mouth \
  --device cuda:0 \
  --output-root /path/to/local/slap_predictions
```

GPUを使用しない場合は`--device cpu`を指定します。
評価の`--output-dir`には、未作成のディレクトリを指定してください。

## SHIFT/HOLD・SHIFT-predの補助評価

`evaluate_all.py`で6対話の推論が完了した後、次を実行します。

```bash
python scripts/evaluate_events.py \
  --prediction-root /path/to/local/slap_predictions \
  --output-dir /path/to/local/slap_events
```

これらはVoice Activity Projectionに基づくproxy課題です。
詳細は[results/README_ja.md](results/README_ja.md)を参照してください。

## 利用条件・出典

[LICENSE](LICENSE)と[重みの利用条件](weights/LICENSE_WEIGHTS.md)を参照してください。
