# GLB quality inspection

## 新しい写真で汎用性を検品する

初見の被写体では、ジョブに送る写真を1枚に固定します。同じ被写体を別角度から撮った参考写真がある場合は別の場所に保存し、生成・着色・カメラ調整が終わるまで処理に渡さないでください。次の2点はメトロポリタン美術館が公開するパブリックドメインの作例です。

| 被写体 | 生成に使った写真 | 横・後ろで確認する箇所 |
| --- | --- | --- |
| 青いカバ像（17.9.1） | [入力写真](https://images.metmuseum.org/CRDImages/eg/original/DP248993.jpg) | 四本の脚の分離、顎下の空間、背の丸み、絵付けの連続性。[別角度の実物写真](https://www.metmuseum.org/art/collection/search/544227) |
| 木製の肘掛け椅子（1984.287） | [入力写真](https://images.metmuseum.org/CRDImages/ad/original/DP-26342-002.jpg) | 座面の奥行き、前後の脚、肘掛けの接続、背もたれの透かし、木部と布の塗り分け。[別角度の実物写真](https://www.metmuseum.org/art/collection/search/212) |

各写真を通常画面から1回だけ送るか、`node clients/cli.mjs generate <写真ファイル> <依頼を識別するキー>` を使用します。同じGLBを材質あり・無地で回し、検品画像のハッシュ、観測した方向、欠け・誤接続、見えない面の推定を記録します。TripoSG の場合は `generation.json` の元画像・距離場・形状生成の記録と、`appearance.json` の塗装元・推定カメラ・材質の記録を確認できます。元写真に写らない側の模様まで合っているかは、別角度の実物写真がある場合に比較できます。

`python quality/inspection.py model.glb --profile profile.json --validator path/to/gltf_validator`

The profile is optional. Without an explicitly selected cat, axes, and head/body regions, geometry is `review`. Coordinates and levels are in world space after glTF node transforms; axes are 0=X, 1=Y, 2=Z. The caller must derive segmentation and thresholds from the actual subject and reference. Example shape only (do not reuse these coordinates blindly):

```json
{"subject":"cat","up_axis":1,"depth_axis":2,"regions":{"head":{"bounds":[[-1,1],[1,2],[-0.6,0.6]],"levels":[1.2,1.5,1.8],"min_depth_ratio":0.2},"body":{"bounds":[[-1,1],[0,1],[-0.6,0.6]],"levels":[0.2,0.5,0.8],"min_depth_ratio":0.2}}}
```

Depth ratio is the median depth of five local scans through a horizontal cross-section divided by that same section's measured width. This avoids a snug depth bound concealing flatness and reduces the influence of an off-center tail. Multiple overlapping contours or an open section require review when they do not already fail the thickness test; disconnected front/back plates cannot pass. The measurement is a heuristic and depends on correct segmentation and orientation; no automatic measurement proves resemblance. A failed section blocks geometry; insufficient data requires review. Passing is not a claim of visual fidelity. Unknown subjects such as legitimately thin products are not subject to a cat thickness rule. Animated/skinned/morph, compressed, sparse and external-buffer assets require manual review. This is not a full glTF parser.

Format status is independent: provide the official [Khronos glTF Validator CLI](https://github.com/KhronosGroup/glTF-Validator#command-line-tool-usage) to run its JSON report; otherwise it remains unknown. Appearance always remains pending human review of matching-hash material and clay views. The report contains a SHA-256 of the inspected file; rerun after any changes. Validator executable identity and provenance are the caller's responsibility.

Run regression tests: `python -m unittest discover -s quality -p "test_*.py"`.
