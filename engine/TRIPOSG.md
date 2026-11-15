# TripoSG を Windows CPU で試す

この追加セットアップは、画像から形状を生成する TripoSG のコード、重み、専用 Python をローカルに準備します。通常のセットアップや既定の生成方式には影響しません。試験段階のため、生成時間と見た目は、対象ごとの実際の GLB で確認してください。セットアップの成功が示すのは、固定した資材の整合性、モデルクラスの読み込み、CPU の BF16 行列演算です。

## 動作条件

Windows x86-64、PowerShell、`uv`、モデルだけで約 7.95 GB のディスク容量が必要です。専用 Python 環境とダウンロードキャッシュの分も空けてください。CPU が BF16 をソフトウェアで処理する機種では、行列演算の検査に通っても推論に長時間かかる可能性があります。実行中の RAM の空きは生成側でも確認してください。上流の案内は CUDA GPU 向けで、ここでは本プロジェクトの CPU 実行経路を使います。[TripoSG モデルカード](https://huggingface.co/VAST-AI/TripoSG/blob/2c1c516d22d58db486a058d98d31bb6177344e06/README.md)

## 写真を1枚入れて使う

この方式を選ぶ場合は、リポジトリ直下で次の2行を実行します。最初のコマンドが前景分離・形状推定・着色・ビューアーを準備し、このインストールで使う方式を保存します。起動後は通常の画面に写真を貼り付けて生成してください。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -TripoSG
.\.runtime\venv\Scripts\python.exe -m studio.server
```

## ランタイムを個別に準備する

リポジトリ直下で実行します。

```powershell
.\engine\setup-triposg.ps1
```

コードは GitHub codeload、重みは Hugging Face の VAST-AI/TripoSG、Python パッケージは PyPI と PyTorch の公式配布先から取得します。Python 3.11.15 は通常セットアップの処理系があれば再利用し、なければ uv の管理する Python 配布元から専用ディレクトリに取得します。画像・認証情報・外部生成サービスは使用しません。uv の通常のダウンロードキャッシュを再利用します。別の場所に置く場合は、起動前に現在の PowerShell で `$env:UV_CACHE_DIR` を指定できます。

既に取得した資材を使う例です。モデルキャッシュには各 `.safetensors` と、同じリビジョンの設定 JSON、モデルカードを上流と同じ階層で置いてください。専用ランタイムにコピー後、各ファイルの SHA-256 を照合します。接続なしでの新規セットアップには、同じ 42 個の Windows wheel が uv キャッシュに入っている必要があります。

```powershell
.\engine\setup-triposg.ps1 -RuntimeDir .\.runtime\triposg -Python C:\path\to\python.exe -SourceArchive C:\cache\triposg-source.zip -WeightsCache C:\cache\triposg-model -Offline
```

ネットワーク中断時は同じコマンドを再実行してください。モデルの未完了ファイルはバイト単位で再開し、最終バイト数と SHA-256 が一致すると採用します。チェックサムの異なる既存ファイルを検出すると、該当パスを表示して停止します。そのファイルを別の場所へ移して再実行してください。自分のプロジェクトや既存の Python 環境を `-RuntimeDir` に指定せず、新規またはこのセットアップが作成した専用ディレクトリを使ってください。

通常セットアップ（リポジトリ直下の `setup.ps1`）には前景分離・着色・検品・ビューアーが含まれます。追加セットアップと合わせて完了したら、同じジョブ保存先に対して次の設定を1回だけ行い、Studioを起動し直します。

```powershell
.\.runtime\venv\Scripts\python.exe -m studio.configure --engine triposg
.\.runtime\venv\Scripts\python.exe -m studio.server
```

専用の保存先を使う場合は、両方のコマンドで同じ `--jobs-dir` を指定してください。専用ランタイムを別の場所に配置した場合は、起動する PowerShell の `STUDIO_TRIPOSG_RUNTIME` 環境変数にその絶対パスを指定します。設定後はWeb・CLI・MCPから通常どおり画像を1枚渡します。公開済みの標準方式に戻すには同じ保存先で `--engine triposr` を指定してStudioを起動し直してください。既に受付済みのジョブには受付時の方式を使います。

元の写真から被写体を切り抜き、TripoSGが推定した形状に、写っている範囲の色を投影します。見えない側の形と塗装は推定です。出力の `shape-master.glb` は着色前の形状、`model.glb` はビューアーに表示するテクスチャ付きモデルです。`generation.json` と `appearance.json` には生成条件・ハッシュ・投影の検査記録を保存します。写真と形状を重ねられない場合は着色工程で停止し、ローカルの途中状態を残します。

## 固定版と記録

| 対象 | 検証値または保存先 |
| --- | --- |
| VAST-AI-Research/TripoSG | GitHub commit `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` |
| ソース ZIP | SHA-256 `cbb21f7dbb9f6759b84f00b5bcdce5c75a5130ae3a63d012646cd274fd945062` |
| VAST-AI/TripoSG | Hugging Face revision `2c1c516d22d58db486a058d98d31bb6177344e06` |
| Windows CPython 3.11.15 の wheel | [triposg-runtime.lock](triposg-runtime.lock) のバージョンと SHA-256 |
| 実行用 Python、ソース、重み | `.runtime/triposg/venv/Scripts/python.exe`、`source/`、`weights/` |
| 検証記録 | `.runtime/triposg/manifest.json` |

記録の `status=runtime-verified` はインストール検査の結果です。`source.files` と `weights.files` に個別のハッシュ、`diagnostics` にライブラリの版と演算結果を保存します。インストーラー単独では画像の生成・品質検品を行いません。利用を終了するときは TripoSG のジョブを終了して、このセットアップ専用のランタイムディレクトリを取り除けます。

実行中のジョブを再開する場合は、受付時のインストールを保持してください。形状の途中状態は `identity.json` に記録したエンジンのコード・重み・実行条件のハッシュに結び付いています。エンジンを更新する場合は、先に現在のジョブを終了するか、元のインストールとジョブを保管して新しい場所でセットアップします。

空きメモリが不足したときは、現在の工程と必要な容量を表示します。最大10分待ち、空きが戻ったら同じ工程を1回だけ再開します。途中で中止した場合も保存済みのステップを利用できます。形状生成後は元の写真と輪郭を照合します。正投影の候補で重なりが足りない場合は、遠近による見かけの大きさの違いも調べます。採用したカメラと同じ投影で色を配置し、最終モデルの見た目は引き続き全方向の確認対象です。

## 第三者の利用条件

本プロジェクトで検証した固定版は、DiT実装に同梱の規約でEU域外を利用対象とし、第5条(c)で生成結果の利用・配布・表示にも地域を定めています。TripoSGによる検品モデルは各利用者が対象地域のローカル環境で生成してください。GitHubの公開デモには従来のTripoSRで作成した配布可能なモデルを掲載しています。

上流の重みのモデルカードとコードのトップレベルには MIT が明記されています。CPU 経路でも使う DiT 実装には Tencent Hunyuan Community License の全文が含まれ、EU を除く地域を対象としています。上流の NOTICE には Diffusers、RMBG、FlashVDM の出典・利用条件も記載されています。重みと第三者コードはこのリポジトリに同梱せず、専用ディレクトリに上流の NOTICE、LICENSE とソース内ライセンスを保存します。各配布元の条件・地域を確認してください。[上流 LICENSE](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/LICENSE) · [NOTICE](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/NOTICE) · [DiT 実装内のライセンス](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py)
