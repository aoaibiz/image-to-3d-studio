# Pixal3D 公開写真デモ

CC0で公開済みの写真を、Hugging Face利用者 `victor` が運営するSpaceへ送ります。コミュニティ共有をオフにしても写真と出力はURLから取得でき、保存期間は未確認です。人物や私物の非公開写真には、Studioのローカル生成を使用してください。[送信先のSpace](https://huggingface.co/spaces/victor/pixal3d-studio)

通常のStudioセットアップ後、専用の保存先を指定します。

2026年11月16日の実機試験では、Web画面から新しい公開写真1枚を受け付け、提供元のイベントIDを取得しました。提供元は必要120秒・残り0秒と応答し、この試験に対するGLBの返却は0件でした。この経路の生成品質は未検証です。

```powershell
python -m studio.configure --jobs-dir .\jobs\public-cc0 --engine pixal3d-space --allow-public-photo-to-pixal-space
python -m studio.server --jobs-dir .\jobs\public-cc0
```

表示されたローカルURLで写真を1枚選び、「このCC0公開写真で3Dを生成」を押します。CLIでは `generate <写真> <リクエストキー> --public-cc0-to-pixal`、MCPでは `generate` の `publicCC0ToPixal: true` で、同じ写真への個別同意を渡します。通常の送信ではこのSpaceを利用できません。

端末内で既存の透過部分を保持し、通常の不透明な写真はセットアップ済みU²-Netで被写体を切り抜きます。Spaceへは黒背景の正方形に整えたPNGを送ります。元画像・メタデータ除去済み画像・切り抜き・送信画像のSHAを生成記録から追跡できます。

1ジョブの立体生成は匿名ZeroGPUの1回です。受付後は保存したイベントIDへ再接続します。受付結果が不明の場合は自動再送を止めます。保存されたテクスチャ付きGLBはSpaceが返したバイト列を保ち、埋込の基本色・金属度/粗さテクスチャを検査します。編集用glTF、生成記録、GLBと同じSHAの検品画像はStudioの成果物から開けます。撮影カメラの向きは未計測です。形状と見えない面の塗装は人による検品を要します。

この連携にモデルの重みは含まれません。上流のPixal3D実装はMITライセンスで、第三者依存・モデルの条件は上流文書を参照してください。この第三者Spaceの公開条件や稼働状況は提供者の管理下にあります。[開発元](https://github.com/TencentARC/Pixal3D)・[上流のライセンスと帰属](https://github.com/TencentARC/Pixal3D/blob/master/NOTICE)
