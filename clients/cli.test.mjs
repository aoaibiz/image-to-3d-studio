import { spawn } from "node:child_process";
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
const server = http.createServer((req, res) => {
  assert.equal(req.headers["x-studio-client"], "1");
  res.end(JSON.stringify({ ok: true }));
});
await new Promise((resolve, reject) =>
  server.listen(47631, "127.0.0.1", resolve).on("error", reject),
);
try {
  await test("CLI diagnose calls local studio", async () => {
    const child = spawn(process.execPath, ["cli.mjs", "diagnose"], {
      cwd: import.meta.dirname,
    });
    let stdout = "",
      stderr = "";
    child.stdout.on("data", (x) => (stdout += x));
    child.stderr.on("data", (x) => (stderr += x));
    const code = await new Promise((r) => child.on("close", r));
    assert.equal(code, 0, stderr);
    assert.equal(JSON.parse(stdout).ok, true);
  });
} finally {
  await new Promise((r) => server.close(r));
}
