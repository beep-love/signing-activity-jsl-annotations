# SLAP実行例の参照結果

[English](README.md)

[Public DGS Corpus](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start_en.html)で学習した[Lexical Signing Activity Projectionモデル](../weights/README.md)を、JSL Signing Activity注釈の6対話に適用した参照結果です。各モデルはJSL上での追加学習や閾値調整は行っていません。
掲載結果はPython 3.10、PyTorch 2.3.0、NumPy 1.23.5、einops 0.7.0で算出しています。

## 結果

| 入力／参照 | Projection NLL ↓ | SHIFT/HOLD bal-acc ↑ | SHIFT/HOLD macro-F1 ↑ | SHIFT-pred bal-acc ↑ | SHIFT-pred macro-F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hands | 5.4644 | 0.4353 | 0.4327 | 0.4603 | 0.4476 |
| Eyes | 4.1968 | 0.4527 | 0.4487 | 0.3135 | 0.3127 |
| Mouth | 5.6246 | 0.4691 | 0.4689 | 0.4040 | 0.3902 |
| Hands + Eyes + Mouth | 6.3997 | 0.4029 | 0.3896 | 0.3730 | 0.3592 |
| 一様分布（256状態） | 5.5452 | — | — | — | — |
| 常にHOLD／no-shift | — | 0.5000 | 0.3310 | 0.5000 | 0.3158 |

Projection NLLを主指標としています。各対話の平均NLLを6対話で平均した値を使用しています。

SHIFT/HOLDとSHIFT-predは、Voice Activity Projection（VAP）に基づく補助的なproxy課題です。
これらのイベント指標は、採用区間内のフレームについて全対話の混同行列を合算して算出しています。
抽出条件と評価設定は[protocol.json](protocol.json)を参照してください。

定数イベントベースラインは常にHOLD／no-shiftを予測した場合の値です。

## 補助イベント評価の対象数

| 対話 | 保存窓数 | SHIFT / HOLD 区間数 | SHIFT/HOLD フレーム数 | SHIFT-pred 正例 / 負例区間数 | SHIFT-pred フレーム数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ani1 | 14 | 0 / 0 | 0 | 0 / 0 | 0 |
| Cur1 | 16 | 4 / 4 | 702 | 4 / 4 | 120 |
| Pro1 | 15 | 0 / 0 | 0 | 0 / 0 | 0 |
| Ani2 | 21 | 1 / 1 | 57 | 1 / 1 | 30 |
| Cur2 | 24 | 0 / 0 | 0 | 0 / 0 | 0 |
| Pro2 | 19 | 2 / 1 | 94 | 2 / 1 | 45 |
| 合計 | 109 | 7 / 6 | 853 | 7 / 6 | 195 |

イベント指標は補助的な参照値です。
Ani1・Pro1・Cur2では、この抽出条件に合うイベントが得られておらず、イベント指標はN/Aです。
