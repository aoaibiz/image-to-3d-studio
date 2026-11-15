# GLB quality inspection

## 新しい写真で汎用性を検品する

初見の被写体では、ジョブに送る写真を1枚に固定します。同じ被写体を別角度から撮った参考写真がある場合は別の場所に保存し、生成・着色・カメラ調整が終わるまで処理に渡さないでください。次の2点はメトロポリタン美術館が公開するパブリックドメインの作例です。

| 被写体 | 生成に使った写真 | 横・後ろで確認する箇所 |
| --- | --- | --- |
| 青いカバ像（17.9.1） | [入力写真](https://images.metmuseum.org/CRDImages/eg/original/DP248993.jpg) | 四本の脚の分離、顎下の空間、背の丸み、絵付けの連続性。[別角度の実物写真](https://www.metmuseum.org/art/collection/search/544227) |
| 木製の肘掛け椅子（1984.287） | [入力写真](https://images.metmuseum.org/CRDImages/ad/original/DP-26342-002.jpg) | 座面の奥行き、前後の脚、肘掛けの接続、背もたれの透かし、木部と布の塗り分け。[別角度の実物写真](https://www.metmuseum.org/art/collection/search/212) |

各写真を通常画面から1回だけ送るか、`node clients/cli.mjs generate <写真ファイル> <依頼を識別するキー>` を使用します。同じGLBを材質あり・無地で回し、検品画像のハッシュ、観測した方向、欠け・誤接続、見えない面の推定を記録します。TripoSG の場合は `generation.json` の元画像・距離場・形状生成の記録と、`appearance.json` の塗装元・推定カメラ・材質の記録を確認できます。元写真に写らない側の模様まで合っているかは、別角度の実物写真がある場合に比較できます。

`python quality/inspection.py model.glb --profile profile.json --validator path/to/gltf_validator`

通常の生成は画像1枚で進みます。特定部位の厚みを検査する場合、CLI または MCP の `inspect` に `measured-parts` を渡せます。モデルと元画像で部位を特定し、1〜6部位にそれぞれの範囲、異なる高さ3〜8点、最小厚み比を指定します。部位名は1〜48文字で、日本語も使えます。従来の `subject: cat` には引き続き `head` と `body` を指定します。プロファイルがないと部位断面は `review` です。

以下は幅1、厚さ0.02の薄い板を想定した例です。座標は glTF ノード変換後のワールド座標、軸は 0=X、1=Y、2=Z です。`up_axis` に垂直な面で切り、`depth_axis` 方向に測ります。実モデルの座標と用途に合わせて値を決めてください。

```json
{"subject":"measured-parts","up_axis":1,"depth_axis":2,"regions":{"薄板":{"bounds":[[-0.6,0.6],[0,1],[-0.1,0.1]],"levels":[0.2,0.5,0.8],"min_depth_ratio":0.015}}}
```

厚み比は断面内5か所の奥行きの中央値を同じ断面の幅で割った値です。各部位のしきい値を1断面でも下回ると `fail` です。開いた断面や判別しにくい重なりは `review` として表示します。`measured-parts` の上限は262,144面、候補三角形と断面の組み合わせは合計250,000組です。上限超過時も `review` になり、処理上の上限を造形不良として記録することはありません。外見と原画への一致は別途、同じ GLB の材質あり・無地の画像で検品します。スキン・アニメーション・モーフ、圧縮メッシュ、スパースアクセサー、外部バッファを含む場合は手動検品が必要です。

Format status is independent: provide the official [Khronos glTF Validator CLI](https://github.com/KhronosGroup/glTF-Validator#command-line-tool-usage) to run its JSON report; otherwise it remains unknown. Appearance always remains pending human review of matching-hash material and clay views. The report contains a SHA-256 of the inspected file; rerun after any changes. Validator executable identity and provenance are the caller's responsibility.

Run regression tests: `python -m unittest discover -s quality -p "test_*.py"`.
