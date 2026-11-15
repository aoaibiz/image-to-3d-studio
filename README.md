# image-to-3d-studio

画像を1枚貼り付けて「3D を生成」を押すと、Windows PC 内で立体を生成し、同じ画面で回して確認できます。通常操作でレシピやモデルを選ぶ必要はありません。

TripoSR が画像から基礎メッシュを生成します。正面向きで耳・折り曲げた腕・脚・横に出た尾を検出できたキャラクターは、輪郭と顔料を画像から測り、丸みのある頭・胴・手足と塗装を組み立てます。車輪と窓を検出できた横向きの玩具車では、車体・窓・車輪を立体の部品として配置します。対象別の仕上げは画像をもとに形状を組み直す処理です。推論メッシュはジョブに別途保存します。

特徴が揃わない場合は TripoSR の結果を開きます。いずれも背面・奥行きは推定で、生成後に材質ありと無地で外観を確認できます。似たポーズが規則的に並ぶシートでは、上段中央に近い1個を自動で選び、使ったポーズを画面に表示します。対象を指定したい場合はその部分を切り抜いて送ってください。

## 生成済みのモデルを回す

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

標準の TripoSR 推論はこの PC 内で行い、画像のアップロードも外部生成 API への課金もありません。セットアップは GitHub、PyTorch、Hugging Face、Python と npm の配布元から依存物を取得します。不透明画像を初めて処理する際は前景を分離するための重みも取得します。透過 PNG は既存のアルファを使用します。初期設定の前後で生成先や契約を切り替える必要はありません。

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
