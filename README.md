# image-to-3d-studio

画像を1枚貼り付けて「3D を生成」を押すと、手元のコンピューターで立体を生成し、同じ画面で回して確認できます。通常操作でレシピやモデルを選ぶ必要はありません。標準の CPU 生成には、Codex へのログインも、生成サービスの API キーも必要ありません。[各 AI から使う](docs/AGENT-CLIENTS.md)

小物や製品など、初めての被写体にも同じ生成ボタンを使います。TripoSR が実際の画像から奥行きと見えない面を推定します。汎用の仕上げでは入力の輪郭を推論メッシュに照合し、輪郭・角・独立した部品を保護しながら表面と色を整えます。照合が難しい場合は推論版を残し、その理由をジョブに記録します。

写真から十分な特徴を測れた場合は、立ち姿の造形、玩具車の車体と車輪、またはグラスの層・クリーム・果実を組み立てる仕上げも自動選択します。パフェはグラス形状、配色、層の位置、果実を入力ごとに計測します。似たポーズが並ぶ画像では上段中央に近い1個を選び、使用した画像を表示します。

完成した画面の「入力画像と立体の形・色を見比べる」から、推論用に切り抜いた入力と、同じGLBの正面・左右を比較できます。切り抜きが飾りや細い部分を落としている場合は、元写真とも見比べてください。細い取っ手、透明な部分、隠れた形、細かい文字や模様には推定誤差があります。検品は正面・側面・背面・無地表示で行います。 [検証記録](VERIFICATION.md) ・ [ツールカタログ](https://mcp-tools-catalog.sparkling-poetry-53ca.workers.dev/tools/image-to-3d-studio)

初めての被写体を種類別のレシピを使わずに生成する場合は、初回セットアップに `-TripoSG` を指定できます。その後は写真1枚と同じ生成ボタンで、形状推定、元写真からの着色、全周の検品画像作成まで進みます。実物の別角度の写真と比較した結果を公開しています。[初期設定と動作条件](engine/TRIPOSG.md) ・ [カバ像・椅子での検証](VERIFICATION.md)

## 生成済みのモデルを回す

未加工の製品写真を1枚ずつ渡した結果を、スマートフォンでも回せます。ティーポットは取っ手の穴や注ぎ口、カメラはレンズ周辺の塗装と側面の厚みを確認できます。ティーポットの口縁のうねり、カメラの背面や文字には再現できていない部分があります。

[ティーポットを回す](https://aoaibiz.github.io/image-to-3d-studio/objects/teapot/) ・ [カメラを回す](https://aoaibiz.github.io/image-to-3d-studio/objects/camera/) ・ [元写真と無地での比較・出典](demo/general/) ・ [モデルと編集用マスター](https://github.com/aoaibiz/image-to-3d-studio/releases/tag/v0.3.0-preview.1)

[自動生成した車を回す](https://aoaibiz.github.io/image-to-3d-studio/auto-car/) ・ [GLB・編集用マスター・オフライン版](https://github.com/aoaibiz/image-to-3d-studio/releases/tag/v0.2.0-preview.1) ・ [材質ありの10方向](demo/auto-result/sheet-material.jpg) ・ [無地の10方向](demo/auto-result/sheet-clay.jpg)

上の車は同梱の CC0 イラスト1枚を、新規インストールした環境の MCP に送って生成したものです。4輪、窓、車体、ライトを全方向から見られます。元絵のグリル・パネルの線は省略されています。1枚からの完成品品質には対象ごとの差があります。[実測結果](VERIFICATION.md)

前版の比較資料も残しています。[比較用の猫](https://aoaibiz.github.io/image-to-3d-studio/) ・ [従来の車](https://aoaibiz.github.io/image-to-3d-studio/car.html)

## 生成後の造形・塗装を調べる

自動生成後に手作業で接合部を調整したい場合は、別途、部位と寸法を指定したレシピから滑らかな面を出力できます。塗装の検査では、意図した色とテクスチャ上の領域を指定し、小さな異色のかたまりを探せます。どちらも任意のローカル CLI です。[造形レシピ](engine/CAST.md)・[塗装検査](quality/PAINT.md)

## 写真から背面の塗装を推定する（プレビュー）

この機能は[開発中のPR #2](https://github.com/aoaibiz/image-to-3d-studio/pull/2)にあります。試すときは専用フォルダーに `git clone --branch improve/unseen-paint https://github.com/aoaibiz/image-to-3d-studio.git` で取得してください。公開リリース `v0.4.0-preview.1` の塗装例と、開発版で新しく生成した結果は、[検証記録](VERIFICATION.md)で比較できます。

### 公式Stable Fast 3DとChatGPTを使う

ローカルGPUを使わずに試す場合の初回設定です。Codex CLIに本人のChatGPTアカウントでログインしてから実行します。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -AllowOfficialSpace -ChatGPTPhotoAnalysis -SendPhotoAndGuidesToOpenAI
.\.runtime\venv\Scripts\python.exe -m studio.server
```

写真を1枚貼って生成を押すと、切り抜いた対象をStability AIの公式Hugging Face Spaceに送り、テクスチャ付きの立体を1回生成します。OpenAIにはメタデータを除いた写真を送り、色・模様・部品を1回読み取ります。塗り足す場合だけ、写真と同じ立体から描いたガイドをOpenAIに送り、塗装案を最大1回作ります。元の材質を残す場合にも、撮影側との一致と反対側への模様の複製を検査します。初期診断では写真を送信しません。利用できる枠・待ち時間は各サービスに依存します。[送信内容とモデルの利用条件](engine/SF3D.md)

同じジョブに、入力、生成元のGLB、仕上げたGLB、編集用glTF、工程の記録を保存します。見た目はビューアーで確認できます。生成モデルが推定した見えない面と、入力写真で確認できる面は区別して評価してください。

写真に同じ模様が何度も写り、立体の輪郭に沿って測った間隔と写真の間隔が合う場合は、手元でその続きを描けます。この場合の外部塗装依頼は0回です。正面に一つだけある顔やロゴを繰り返す根拠には使いません。推定した画像を貼る場合は、滑らかにつながった面の上に、元の絵にも塗装案にもなかった線が現れていないかを複数の方向から調べます。切れ目が見つかると、その塗装の書き出しを保留します。検査を通った後も、元写真との一致と、横や後ろの自然さをビューアーで確認できます。

写真に張り出した口縁が見え、立体の上面が平らに塞がり、内壁を作る余地がある場合は、同じジョブに内側を推定した別案も保存できます。写真の解析・形状計測が条件を満たしたときだけ、結果画面に別案を回すリンクが現れます。元のモデルと別案はそれぞれGLBで保存できます。口径・深さは推定で、追加の外部生成は使いません。

### ローカルTripoSGとChatGPTを使う

横と背面を一続きの展開図で推定する開発版を使う場合の初回設定です。

```powershell
.\setup.ps1 -TripoSG -SourceSurfaceChart -ChatGPTPhotoAnalysis -SendPhotoAndGuidesToOpenAI
.\.runtime\venv\Scripts\python.exe -m studio.server
```

形状はPCで生成します。初回に指定した本人のChatGPT契約で、OpenAIに写真1枚を送り、配色と模様を読み取ります。追加の塗装が適する場合だけ、写真と、両端に実写部分を残した展開図を1回送ります。撮影側と推定側がつながるかを検査し、根拠が不足する場合は、その理由と検品用のモデルを表示します。背面に実際に何が描かれていたかは、別角度の実写資料で確認できます。

Windows で次の初期設定を1回行うと、写真を貼るだけで、Codex が色、繰り返し模様、正面にある部品と、反対面で役割が変わる可能性を読み取ります。本人の ChatGPT 利用枠で、写真の解析を1回、立体から作ったガイドを使った塗装の生成を1回依頼します。送信先は OpenAI、送信内容はメタデータを除いた元写真と立体の確認画像です。模様の指示は不要です。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -SendPhotoAndGuidesToOpenAI -ChatGPTPhotoAnalysis
.\.runtime\venv\Scripts\python.exe -m studio.server
```

Web 画面だけでこの構成を使うには、画像生成を利用できる Codex CLI に ChatGPT でログインしてください。写真の解析をローカルの Qwen で行う場合は `-ChatGPTPhotoAnalysis` を省きます。Codex 内蔵の画像生成を使う場合は、同フラグを `-UseCodexHost` に替えてください。短いインストール先（例 `C:\3d\studio`）を使うと、Windows のパス長制限による導入失敗を避けやすくなります。

生成した裏面に正面特有の部品が重なる、または推定画像の輪郭・穴が形状に合わない場合は投影を止めます。GLB の外観は正面・側面・背面と無地表示で検品してください。無料のローカル構成、外部送信の解除、モデルの利用条件は各手順で確認できます。[写真解析](engine/VISUAL_BRIEF.md) ・ [TripoSG](engine/TRIPOSG.md) ・ [クライアント設定](clients/README.md)

外部送信を解除するときは `.\.runtime\venv\Scripts\python.exe -m studio.configure --engine triposr` を実行し、Studio を再起動してください。

## 起動する

Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)、Node.js 22 と npm、WebGL 対応ブラウザーを用意します。初回に約 1.68 GB の重み、Python、依存ライブラリ、検品ツールをダウンロードします。重みと生成物を保存できる空き容量を確保してください。

| 生成を実行する端末 | 標準のローカル CPU 構成 |
| --- | --- |
| Windows 11 x64 | PowerShell を使用。従来版の実測環境は Ryzen 7 255、RAM 29.8 GiB、Radeon 780M。 |
| macOS 14 以降（Apple Silicon・Intel） | 下記のシェルを使用。推論エンジン単体の対象は macOS 13 以降。 |
| Linux x86_64 | 同じシェルを使用。検品用ブラウザーを含む構成の対象は Ubuntu 22.04/24.04、Debian 12/13。 |

macOS・Linux は依存パッケージと起動経路の静的レビューまで行いました。今回の変更でのインストール、生成、各 AI アプリへの接続は実機未検証です。上記は標準の TripoSR の対象です。追加エンジンは個別の導入条件に従ってください。[CPU エンジンの条件](engine/README.md) ・ [検品用ブラウザーの動作条件](https://github.com/microsoft/playwright/blob/v1.56.1/docs/src/intro-js.md#system-requirements)

**Windows**

```powershell
git clone https://github.com/aoaibiz/image-to-3d-studio.git
cd image-to-3d-studio
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.runtime\venv\Scripts\python.exe -m studio.server
```

**macOS・Linux**

```sh
git clone https://github.com/aoaibiz/image-to-3d-studio.git
cd image-to-3d-studio
sh setup.sh
.runtime/venv/bin/python -m studio.server
```

`curl` と `shasum` または `sha256sum` も使います。Linux で Chromium が共有ライブラリ不足を報告した場合は、対象ディストリビューションの管理者が `node viewer/node_modules/playwright/cli.js install-deps chromium` で OS 側の依存を用意した後、セットアップを再実行してください。

[ローカル画面](http://127.0.0.1:47631) を開きます。PNG・JPEG・WebPを貼り付けるか、ドロップまたはファイル選択し、「3D を生成」を押してください。GLBが開いたら、ドラッグで回転、ホイールで拡大します。背面・側面・上下面の視点、照明、白い無地、ワイヤーを切り替えられます。ダウンロードする GLB は表示中のものと同じです。停止はサーバーを起動したウィンドウで Ctrl+C。実行ポリシーの指定はこのセットアップ用プロセスにのみ適用されます。

## 任意のローカル形状エンジン

TripoSG を使う場合の初回セットアップは `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -TripoSG` です。セットアップ後は通常と同じコマンドで起動し、画像を1枚渡してください。

写真に写った小物や家具を、被写体名や専用レシピの指定なしで TripoSG に渡す実験用の経路があります。初回にジョブ保存先ごとに設定すると、Web・CLI・MCP のどこから写真を送っても同じ方式を使います。CPUでの生成には長時間かかります。写真に写らない形・材質・模様は推定です。[動作条件・利用条件・設定方法](engine/TRIPOSG.md)

## 画像と費用

標準の TripoSR 推論はこの PC 内で行い、画像のアップロードも外部生成 API への課金もありません。セットアップは GitHub、PyTorch、Hugging Face、Python と npm の配布元から依存物を取得します。セットアップで通常の写真から前景を切り抜く U2Net の重み（約176 MB）も取得・検証します。透過 PNG は既存のアルファを使用します。初期設定の前後で生成先や契約を切り替える必要はありません。

画像・設定・モデル・検品結果は clone 内の `jobs/studio/<ジョブID>/` に保存され、中止・失敗後も残ります。共有 PC では自分だけが読めるフォルダーを使ってください。削除するには、必要な成果物を別の場所へ保存し、サーバーを停止して対象ジョブのフォルダーをエクスプローラーで削除してください。SQLite にはジョブID・ハッシュ・履歴が残ります。全履歴を消す場合は停止中に専用の `jobs/studio` ディレクトリを削除します。モデル入りのオフライン HTML を渡すと、その中の GLB も渡ります。

## 検品する

- [車の元画像](demo/product-cutout.png)・[生成したGLB](demo/generated/model.glb)・[材質あり10視点](demo/generated/views/sheet-material.jpg)・[無地10視点](demo/generated/views/sheet-clay.jpg)・[目視検品の記録](demo/generated/review.json)
- [Khronos による形式検証](demo/generated/inspection.json)・[ブラウザーでの回転とダウンロード検証](demo/generated/views/manifest.json)・[検品画像と GLB の SHA-256](demo/generated/views/evidence.json)
- [編集可能なマスターとテクスチャ](demo/generated)。OBJ・MTL・material_0.png は一緒に保存してください。

検査状態は形式・部位の形状・外観に分かれます。生成直後の `needs_review` は人または画像を確認できる AI による検品待ちです。立ち姿として仕上げた対象では、画像から求めた頭・胴の位置で完成GLBの断面を検査します。範囲や断面が不明確な場合は要確認です。対象や範囲の手動指定もできます。尻尾を含む全体の奥行きだけでは合格にしません。薄い商品に猫の基準を一律適用しません。元絵に対する顔や模様の一致は、無地の側面で見る立体感とは別に確認します。[断面の指定と回帰テスト](quality/README.md)

## AI エージェントから使う

Web・CLI・標準 MCP は同じローカルサービスとジョブを使用します。Claude Code、Cursor、Gemini CLI、Copilot、Codex の設定は `node clients/agent-config.mjs` にホスト名を指定して表示できます。その他のエージェントは `stdio` の設定、または下記の CLI を使用してください。ローカルプロセスを起動できないクラウド AI には、その実行環境に Studio を置くか、利用者が管理する認証付きブリッジが必要です。[ホスト別の設定・画像の渡し方](docs/AGENT-CLIENTS.md)

Studio の起動後、別のターミナルから実行します。

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

公開版の更新時は[ツールカタログの掲載ページ](https://mcp-tools-catalog.sparkling-poetry-53ca.workers.dev/tools/image-to-3d-studio)にも、導入先・実際に配布したバージョン・利用できるMCPツールを反映してください。検証中の変更にはドラフトPRへのリンクを使います。
