# Local clients

Requires Node.js 22 and the local Studio service listening at `127.0.0.1:47631`. The clients do not start the service. Install locally with `npm ci` in this directory.

CLI:

```
node cli.mjs diagnose
node cli.mjs generate path/to/image.png your-stable-request-key
node cli.mjs status JOB_ID
node cli.mjs watch JOB_ID
node cli.mjs cancel JOB_ID
node cli.mjs resume JOB_ID
node cli.mjs show JOB_ID
node cli.mjs inspect JOB_ID
node cli.mjs export JOB_ID model.glb path/to/output.glb
```

`generate` requires a stable request key. Reuse it when retrying the same image with the same pipeline. Use a new key when intentionally generating the same image with a different pipeline. `export` requires `--overwrite` to replace an existing file. `show` prints the local preview URL.

MCP uses stdio: run `node clients/mcp.mjs` with the absolute path appropriate to the installation. Tools accept image bytes encoded as base64, never local paths. `artifact_chunk` returns up to 256 KiB per call and a `nextOffset`; its maximum offset is 64 MiB. `wait` uses the local event stream. Cancelling the MCP call stops waiting; use the `cancel` tool to cancel the job itself.

Run `npm test`; the tests reserve isolated ephemeral loopback ports. This uses the official SDK client and a local test service, and checks the actual saved PNG bytes. It does not run inference or prove the real service works.

## Codex local plugin

The plugin is local, not a cloud connector. Node 22, `npm ci` in `clients`, and the running Studio service are prerequisites. This example uses an isolated Codex home; it writes the marketplace and plugin installation there only. From PowerShell in the checkout:

```powershell
$studioRoot = (Get-Location).Path
$testCodexHome = Join-Path $studioRoot 'jobs/codex-plugin-test'
New-Item -ItemType Directory -Force $testCodexHome | Out-Null
$env:CODEX_HOME = $testCodexHome
npm ci --prefix clients
node plugins/image-to-3d-studio/bootstrap.mjs $testCodexHome $studioRoot
codex plugin marketplace add $studioRoot
codex plugin add image-to-3d-studio@image-to-3d-local
codex plugin list
```

For a normal Codex home, pass that home's explicit absolute path to the bootstrap and use the corresponding Codex instance to register/install. Bootstrap writes exactly `plugins/image-to-3d-studio-root.txt` inside the chosen Codex home; the file contains the checkout's absolute path. This is needed because installed plugins are cached elsewhere and child processes do not necessarily inherit custom environment variables. Repeat bootstrap after moving the checkout. Restart Codex after installation. Remove the plugin through `codex plugin remove image-to-3d-studio@image-to-3d-local`; remove the pointer file separately if no longer needed. No global npm installation is used.

Tested with Windows Codex CLI 0.169.0-alpha.11.1 in a separate Codex home: marketplace add, plugin add/list, app-server discovery of all nine MCP tools, and an app-server `diagnose` call against the real local service. The official MCP SDK stdio client also submitted a real saved PNG and retrieved a GLB in chunks; this does not establish that a chat attachment is automatically available to the plugin.

## Other hosts and remote use

Claude Code and Cursor can run a local stdio MCP server, but neither host has been tested with this project. Following their current documentation, point their local MCP registration at `node` with an absolute argument `<checkout>/clients/mcp.mjs`; install dependencies and start Studio first. Claude Code's `claude mcp add --transport stdio image-to-3d-studio -- node <absolute-checkout>/clients/mcp.mjs` writes host configuration, so run it only in the intended scope. Cursor's project `.cursor/mcp.json` may contain `{"mcpServers":{"image-to-3d-studio":{"type":"stdio","command":"node","args":["<absolute-checkout>/clients/mcp.mjs"]}}}`; replace the placeholder with a properly JSON-escaped path. Verify each host's permissions and tool discovery. References: https://code.claude.com/docs/en/mcp and https://cursor.com/docs/mcp.

Cloud agents cannot reach the user's `127.0.0.1` through this local setup. Remote access would require a separately designed and authorized authenticated bridge and explicit exposure decisions; no tunnel or bridge is provided by default.

For optional cross-section analysis of 1–6 named parts or the cat head and body, run `node cli.mjs inspect JOB_ID profile.json`; without a profile it reads the existing `inspection.json`. `node cli.mjs review JOB_ID review.json` records a manual evaluation after examining both material and clay views. The review JSON requires `sha256`, `decision` (`pass`, `fail`, or `review`), `reference_fidelity`, `major_part_volume`, `attachment_material`, and `inspected_views` exactly `["material","clay"]`. Each text field is required and limited to 4,000 characters. Job `status` describes processing; `review` records the independent manual decision. A new geometric inspection invalidates an earlier manual review. The MCP `inspect` and `review` tools use the same fields and validation.

## One-picture workflow and review

Use one PNG, JPEG, or WebP (up to 20 MiB) as the reference. Start the local service first; run `diagnose`, then submit the selected image bytes once with a stable request key. Record the returned job ID and use `wait` or `watch` for the event stream. `status` reports the processing state and the separate `review` field records an inspection decision. `needs_review` means the assets are ready for inspection, not that the shape or appearance has passed.

Inspect the original reference, model, and hash-matched material and clay views together, including the back, underside, joints, and parts visible in the input. Use `inspection.json` for format and measured-geometry results and `settings.json` for input/output hashes, processing and any applied or skipped steps. An image cannot establish hidden surfaces, physical scale, or manufacturing suitability. The manual cat profile supported by the `inspect` tool is explicitly specified by the operator; it is not a general-purpose animal, vehicle or product classifier. The optional geometric casting and atlas color tools have separately documented narrow inputs and cannot certify a model: see `engine/CAST.md` and `quality/PAINT.md`. Preserve input and output hashes and report uncertain or skipped analysis as such.

The service chooses the pipeline at startup. Use `node cli.mjs diagnose` to check the service environment. Read `GET /api/engine` on that same service to identify the exact pipeline configured for new jobs. The web UI shows the selection before enabling generation. Both supported pipelines process images locally.

- `triposr-auto-finish-v2` uses TripoSR for the base shape and applies category-specific finishing when the image supports it. See each job’s `settings.json` for the steps actually applied.
- `triposg-local-v1` uses TripoSG for shape inference and colors the result locally. CPU generation can take hours. The model estimates the object's orientation, so rotate it to find the side shown in the photo.

An HTTP client can read `pipeline` and `local` from `GET /api/engine`, then include that exact pipeline in the `X-Studio-Pipeline` header on `POST /api/jobs`, along with its stable `Idempotency-Key`. If a configuration change causes a conflict, check the active pipeline before submitting again. To recover an existing job, use its job ID. In the viewer, the input/model comparison link appears when the job includes a comparison sheet matching the displayed GLB hash.

For an alternate local service port, start the service with its --port option and set STUDIO_PORT to a decimal integer from 1 to 65535 in the CLI or standalone MCP process environment. The host remains fixed at 127.0.0.1. The default is 47631. The packaged Codex plugin may not inherit custom environment variables; use its default port unless the host is explicitly configured to pass the variable.
