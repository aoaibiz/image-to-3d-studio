import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
const id = "12345678-1234-4123-8123-123456789abc";
const image = await readFile(
  new URL("../demo/product-cutout.png", import.meta.url),
);
let received;
const service = http.createServer(async (req, res) => {
  assert.equal(req.headers["x-studio-client"], "1");
  if (req.method === "POST" && req.url === "/api/jobs") {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    received = Buffer.concat(chunks);
    assert.equal(req.headers["idempotency-key"], "stable-test-key");
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify({ job: { id, status: "queued" } }));
  }
  if (req.url === `/api/jobs/${id}/events`) {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    return res.end(
      `id: 1\ndata: ${JSON.stringify({ id: 1, job_id: id, payload: { status: "needs_review" } })}\n\n`,
    );
  }
  if (req.url === `/api/jobs/${id}`)
    return res.end(JSON.stringify({ job: { id, status: "needs_review" } }));
  if (req.url === `/api/jobs/${id}/artifacts/model.glb`)
    return res.end(Buffer.from("test-artifact"));
  if (req.url === "/api/doctor") return res.end(JSON.stringify({ ok: true }));
  res.writeHead(404).end();
});
await new Promise((resolve, reject) =>
  service.listen(0, "127.0.0.1", resolve).on("error", reject),
);
const transport = new StdioClientTransport({
  command: process.execPath,
  env: { ...process.env, STUDIO_PORT: String(service.address().port) },
  args: [
    new URL("./mcp.mjs", import.meta.url).pathname.replace(
      /^\/(?=[A-Za-z]:)/,
      "",
    ),
  ],
});
const client = new Client({ name: "integration-test", version: "1.0.0" });
try {
  await client.connect(transport);
  await test("official MCP SDK stdio submits exact saved image, waits on SSE, and reads chunks", async () => {
    const tools = await client.listTools();
    assert(tools.tools.some((x) => x.name === "generate"));
    const submit = await client.callTool({
      name: "generate",
      arguments: {
        imageBase64: image.toString("base64"),
        mime: "image/png",
        requestKey: "stable-test-key",
      },
    });
    assert.equal(submit.isError, undefined);
    assert.deepEqual(received, image);
    const wait = await client.callTool({
      name: "wait",
      arguments: { jobId: id },
    });
    assert.equal(JSON.parse(wait.content[0].text).job.status, "needs_review");
    const part = await client.callTool({
      name: "artifact_chunk",
      arguments: { jobId: id, filename: "model.glb", offset: 0, limit: 4 },
    });
    assert.equal(
      Buffer.from(JSON.parse(part.content[0].text).data, "base64").toString(),
      "test",
    );
  });
} finally {
  await client.close();
  await new Promise((r) => service.close(r));
}
