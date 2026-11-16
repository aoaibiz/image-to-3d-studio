# Windows・macOS・Linux で AI から画像を渡す

ローカルの MCP（stdio）を起動できる AI には共通のサーバーを、コマンドを実行できる AI には CLI を使います。標準のローカル CPU 生成（TripoSR）に Codex や OpenAI アカウントは必要ありません。エージェント、Node.js、Studio は同じ実行環境に置いてください。以下はリポジトリのルートから実行します。

## 初回設定

1. [README の導入条件](../README.md)に従って、Windows では `setup.ps1`、macOS・Linux では `bash setup.sh` でローカル生成をセットアップします。
2. Windows は `.\.runtime\venv\Scripts\python.exe -m studio.server`、macOS・Linux は `.runtime/venv/bin/python -m studio.server` で Studio を起動し、このプロセスを開いたままにします。既定は `127.0.0.1:47631` です。
3. Node.js 22 を使い、クライアントの依存関係がまだ入っていなければ `npm ci --prefix clients` を実行します。エージェントを使う端末で下表の形式を選びます。自分で組み込む場合は `stdio` が返す実行ファイルを、返された引数配列で直接起動してください。シェルや固定の作業ディレクトリは不要です。

例：`node clients/agent-config.mjs cursor`。**コマンドは設定の断片を表示するだけ**です。出力のサーバー定義を既存設定に追加し、他のエントリは残してください。出力には端末固有の絶対パスが入るのでコミットせず、端末やインストール先を変えたら再発行してください。

| エージェント | 引数 | 追加先 |
| --- | --- | --- |
| [Claude Code](https://code.claude.com/docs/en/mcp) | `claude` | 対象プロジェクトの `.mcp.json` |
| [Cursor](https://cursor.com/docs/mcp) | `cursor` | 対象プロジェクトの `.cursor/mcp.json` |
| [Gemini CLI](https://geminicli.com/docs/tools/mcp-server/) | `gemini` | `.gemini/settings.json` の `mcpServers` |
| [VS Code の Copilot / Copilot CLI](https://code.visualstudio.com/docs/agents/reference/mcp-configuration) | `vscode` / `copilot` | Copilot の `~/.copilot/mcp-config.json`（`COPILOT_HOME` を変更した場合はその配下）。従来の VS Code Local 用 `servers` 形式とは別です。 |
| [Codex](https://developers.openai.com/codex/mcp/) | `codex` | `~/.codex/config.toml`。同じサーバーのプラグインを既に入れた場合は二重登録不要です。 |
| その他の [MCP stdio](https://modelcontextprotocol.io/docs/learn/architecture) 対応ホスト | `stdio` | ホストの設定画面でサーバー名を `image-to-3d-studio` とし、出力の `command` と `args` を設定 |

プロジェクトを開き直し、ホストが求める場合はサーバーの起動を許可します。CLI のみ使う場合、上表の登録は不要です。

## 写真を渡して結果を取り出す

1. MCP の `engine` と `diagnose` で生成先・準備状態を確認します。標準ローカル構成では `local: true` です。外部送信を含む構成なら表示された送信先に対する利用者の選択に従います。
2. `generate` に `imageBase64`（PNG/JPEG/WebP の生バイトの Base64、データ URL は不可）、`mime`、同じ依頼で再利用する `requestKey`、`engine` が返した `pipeline` の値を `expectedPipeline` として渡します。サービス側の方式が変わった場合は送信が拒否されます。ホストが添付をファイルとして提供する場合は、次の CLI が直接ファイルを読み込むので Base64 を会話に貼る必要はありません。画像の制限は 20 MiB、一部ホストのツール引数の上限はさらに小さくなります。
3. 返った `job.id` を保持し、`wait` で進捗を待ちます。切断や待機タイムアウトの後は同じ ID に `status` / `wait` で再接続します。異なるキーで生成し直さないでください。中止には `cancel` を使います。
4. 出力のあるジョブでは `artifact_chunk` で `model.glb` を取得します。`offset=0` から始め、返された `nextOffset` が null になるまで各チャンクを Base64 デコードして順番に連結します。MCP の読み出し開始位置の上限は 64 MiB です。64 MiB を超えるモデルには、ファイル全体をストリームで保存する CLI の `export` を使ってください。処理の完了と外観の合格は別です。材質あり・無地で各方向を確認し、判断できない場合は「要確認」と伝えます。

MCP がないエージェントでも、同じ端末でシェルを実行できればこの手順を使えます。以下の `PIPELINE_ID` は `engine` の戻り値、`JOB_ID` は `generate` の戻り値に置き換えてください。

```text
node clients/cli.mjs engine
node clients/cli.mjs diagnose
node clients/cli.mjs generate "path/to/photo.jpg" photo-001 --pipeline PIPELINE_ID
node clients/cli.mjs watch JOB_ID
node clients/cli.mjs show JOB_ID
node clients/cli.mjs export JOB_ID model.glb output.glb
```

**クラウドからの利用:** コンテナやリモート開発機の `127.0.0.1` は手元の PC ではありません。その環境内に Studio とクライアントを導入するか、別途認証・暗号化・ファイル受け渡しに対応するブリッジが必要です。ブリッジや外部公開された生成 API は同梱していません。MCP・ローカルコマンド・独自の接続手段のいずれもないチャットだけの AI には直接接続できません。`show` の URL も同じ端末向けです。

**確認範囲:** 従来版では Windows の Codex と公式 Node MCP クライアントを実際に接続しています（[当時の記録](../VERIFICATION.md)）。今回追加した設定出力・各エージェントの接続・macOS/Linux での生成について実機テストは行っていません。各アプリ共通なのは接続方式とジョブ API です。写真1枚での全周品質は保証していません。
