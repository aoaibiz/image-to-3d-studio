# image-to-3d-studio

画像を1枚貼り付けて「3D を生成」を押すと、Windows PC 内で立体を生成し、同じ画面で回して確認できます。通常操作でレシピやモデルを選ぶ必要はありません。

小物や製品など、初めての被写体にも同じ生成ボタンを使います。TripoSR が実際の画像から奥行きと見えない面を推定します。汎用の仕上げでは入力の輪郭を推論メッシュに照合し、輪郭・角・独立した部品を保護しながら表面と色を整えます。照合が難しい場合は推論版を残し、その理由をジョブに記録します。

写真から十分な特徴を測れた場合は、立ち姿の造形、玩具車の車体と車輪、またはグラスの層・クリーム・果実を組み立てる仕上げも自動選択します。パフェはグラス形状、配色、層の位置、果実を入力ごとに計測します。似たポーズが並ぶ画像では上段中央に近い1個を選び、使用した画像を表示します。

完成した画面の「入力画像と立体の形・色を見比べる」から、推論用に切り抜いた入力と、同じGLBの正面・左右を比較できます。切り抜きが飾りや細い部分を落としている場合は、元写真とも見比べてください。細い取っ手、透明な部分、隠れた形、細かい文字や模様には推定誤差があります。検品は正面・側面・背面・無地表示で行います。 [検証記録](VERIFICATION.md) ・ [ツールカタログ](https://mcp-tools-catalog.sparkling-poetry-53ca.workers.dev/tools/image-to-3d-studio)

## 生成済みのモデルを回す

未加工の製品写真を1枚ずつ渡した結果を、スマートフォンでも回せます。ティーポットは取っ手の穴や注ぎ口、カメラはレンズ周辺の塗装と側面の厚みを確認できます。ティーポットの口縁のうねり、カメラの背面や文字には再現できていない部分があります。

[ティーポットを回す](https://aoaibiz.github.io/image-to-3d-studio/objects/teapot/) ・ [カメラを回す](https://aoaibiz.github.io/image-to-3d-studio/objects/camera/) ・ [元写真と無地での比較・出典](demo/general/) ・ [モデルと編集用マスター](https://github.com/aoaibiz/image-to-3d-studio/releases/tag/v0.3.0-preview.1)

[自動生成した車を回す](https://aoaibiz.github.io/image-to-3d-studio/auto-car/) ・ [GLB・編集用マスター・オフライン版](https://github.com/aoaibiz/image-to-3d-studio/releases/tag/v0.2.0-preview.1) ・ [材質ありの10方向](demo/auto-result/sheet-material.jpg) ・ [無地の10方向](demo/auto-result/sheet-clay.jpg)

上の車は同梱の CC0 イラスト1枚を、新規インストールした環境の MCP に送って生成したものです。4輪、窓、車体、ライトを全方向から見られます。元絵のグリル・パネルの線は省略されています。1枚からの完成品品質には対象ごとの差があります。[実測結果](VERIFICATION.md)

前版の比較資料も残しています。[比較用の猫](https://aoaibiz.github.io/image-to-3d-studio/) ・ [従来の車](https://aoaibiz.github.io/image-to-3d-studio/car.html)

## 生成後の造形・塗装を調べる

自動生成後に手作業で接合部を調整したい場合は、別途、部位と寸法を指定したレシピから滑らかな面を出力できます。塗装の検査では、意図した色とテクスチャ上の領域を指定し、小さな異色のかたまりを探せます。どちらも任意のローカル CLI です。[造形レシピ](engine/CAST.md)・[塗装検査](quality/PAINT.md)

## 起動する

Windows 11 x64、Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、Node.js 22 と npm、WebGL を利用できる Chrome または Edge を用意してください。実測環境は Ryzen 7 255、RAM 29.8 GiB、Radeon 780M です。推論は CPU を使用します。初回に約1.68 GBの重みのほか、Python・依存パッケージ・検証ツールをダウンロードするため、数 GB の空き容量が必要です。

```powershell
git clone https://github.com/aoaibiz/image-to-3d-studio.git
cd image-to-3d-studio
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.runtime\venv\Scripts\python.exe -m studio.server
```

[ローカル画面](http://127.0.0.1:47631) を開きます。PNG・JPEG・WebPを貼り付けるか、ドロップまたはファイル選択し、「3D を生成」を押してください。GLBが開いたら、ドラッグで回転、ホイールで拡大します。背面・側面・上下面の視点、照明、白い無地、ワイヤーを切り替えられます。ダウンロードする GLB は表示中のものと同じです。停止はサーバーを起動したウィンドウで Ctrl+C。実行ポリシーの指定はこのセットアップ用プロセスにのみ適用されます。

## 画像と費用

標準の TripoSR 推論はこの PC 内で行い、画像のアップロードも外部生成 API への課金もありません。セットアップは GitHub、PyTorch、Hugging Face、Python と npm の配布元から依存物を取得します。セットアップで通常の写真から前景を切り抜く U2Net の重み（約176 MB）も取得・検証します。透過 PNG は既存のアルファを使用します。初期設定の前後で生成先や契約を切り替える必要はありません。

画像・設定・モデル・検品結果は clone 内の `jobs/studio/<ジョブID>/` に保存され、中止・失敗後も残ります。共有 PC では自分だけが読めるフォルダーを使ってください。削除するには、必要な成果物を別の場所へ保存し、サーバーを停止して対象ジョブのフォルダーをエクスプローラーで削除してください。SQLite にはジョブID・ハッシュ・履歴が残ります。全履歴を消す場合は停止中に専用の `jobs/studio` ディレクトリを削除します。モデル入りのオフライン HTML を渡すと、その中の GLB も渡ります。

## 検品する

- [車の元画像](demo/product-cutout.png)・[生成したGLB](demo/generated/model.glb)・[材質あり10視点](demo/generated/views/sheet-material.jpg)・[無地10視点](demo/generated/views/sheet-clay.jpg)・[目視検品の記録](demo/generated/review.json)
- [Khronos による形式検証](demo/generated/inspection.json)・[ブラウザーでの回転とダウンロード検証](demo/generated/views/manifest.json)・[検品画像と GLB の SHA-256](demo/generated/views/evidence.json)
- [編集可能なマスターとテクスチャ](demo/generated)。OBJ・MTL・material_0.png は一緒に保存してください。

検査状態は形式・部位の形状・外観に分かれます。生成直後の `needs_review` は人または画像を確認できる AI による検品待ちです。立ち姿として仕上げた対象では、画像から求めた頭・胴の位置で完成GLBの断面を検査します。範囲や断面が不明確な場合は要確認です。対象や範囲の手動指定もできます。尻尾を含む全体の奥行きだけでは合格にしません。薄い商品に猫の基準を一律適用しません。元絵に対する顔や模様の一致は、無地の側面で見る立体感とは別に確認します。[断面の指定と回帰テスト](quality/README.md)

## CLI・MCP・Codex

Web・CLI・MCP は同じローカルサービスとジョブを使用します。別の PowerShell で実行してください。

```powershell
node clients/cli.mjs diagnose
node clients/cli.mjs generate demo/product-cutout.png first-car
node clients/cli.mjs watch <ジョブID>
node clients/cli.mjs show <ジョブID>
node clients/cli.mjs export <ジョブID> model.glb my-car.glb
```

同一依頼の再送には同じキーを使います。ジョブの進捗は SSE で受け取り、再接続後は保存済みのジョブ ID に復帰します。MCP は stdio で `node <絶対パス>/clients/mcp.mjs` を起動し、保存した画像バイトを Base64 で渡します。画像を保存できないチャットでは、添付ファイルをPCに保存してから渡してください。

Codex のプラグイン・制作スキル、Claude Code・Cursor の設定例、手元の PC に直接接続できないクラウド AI の制約は [クライアント導入手順](clients/README.md) にまとめています。対応を実測した範囲は [検証結果](VERIFICATION.md) に記載しています。

## 再現とライセンス

```powershell
.\.runtime\venv\Scripts\python.exe tools/doctor.py --verify-weights
.\.runtime\venv\Scripts\python.exe -m unittest discover -s quality -p 'test_*.py'
node viewer/make-preview.cjs my-car.glb my-car-preview.html
```

[エンジンの版・設定](engine/README.md)、[ビューアーと検品画像の生成方法](viewer/README.md)、[検証結果と限界](VERIFICATION.md) を参照してください。プロジェクトのコードは [MIT](LICENSE)、車の原画は [CC0](demo/README.md) です。モデルと依存ライブラリの条件は [THIRD_PARTY.md](THIRD_PARTY.md) に分けて記録しています。
