import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { readFile, mkdtemp, writeFile, unlink, rmdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
const id = "12345678-1234-4123-8123-123456789abc";
const image = await readFile(
  new URL("../demo/product-cutout.png", import.meta.url),
);
const hostDir = await mkdtemp(join(tmpdir(), "studio-test-"));
const tokenFile = join(hostDir, "host-access.token");
await writeFile(tokenFile, "t".repeat(43), {mode:0o600});
let received, inspected, returnedPaint;
const service = http.createServer(async (req, res) => {
  assert.equal(req.headers["x-studio-client"], "1");
  if (req.method === "POST" && req.url === `/api/jobs/${id}/host/claim`) {
    assert.equal(req.headers["x-studio-host-token"], "t".repeat(43));
    return res.end(JSON.stringify({ job_id:id, generate_once:true, claim_id:"c".repeat(43), request_sha256:"a".repeat(64) }));
  }
  if (req.method === "POST" && req.url === `/api/jobs/${id}/host/submit`) {
    assert.equal(req.headers["x-studio-host-token"], "t".repeat(43));
    assert.equal(req.headers["x-studio-tool-call"], "fixture-call-id");
    assert.equal(req.headers["x-studio-claim"], "c".repeat(43));
    assert.equal(req.headers["x-studio-request-sha256"], "a".repeat(64));
    const parts = [];
    for await (const chunk of req) parts.push(chunk);
    returnedPaint = Buffer.concat(parts);
    assert.equal(req.headers["x-studio-paint-sha256"], createHash("sha256").update(returnedPaint).digest("hex"));
    return res.end(JSON.stringify({job:{id,status:"queued"}}));
  }
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
  if (req.method === "POST" && req.url === `/api/jobs/${id}/inspect`) {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    inspected = JSON.parse(Buffer.concat(chunks).toString());
    return res.end(JSON.stringify({ job: { id, status: "needs_review" } }));
  }
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
  env: { ...process.env, STUDIO_PORT: String(service.address().port), STUDIO_HOST_TOKEN_FILE:tokenFile },
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
  await test("host stdio adapter preserves claim, binding and exact returned PNG", async () => {
    const got = await client.callTool({name:"claim_paint",arguments:{jobId:id}});
    const claim = JSON.parse(got.content[0].text);
    assert.equal(claim.generate_once, true);
    const args = {jobId:id,claimId:claim.claim_id,requestSha256:claim.request_sha256,toolCallId:"fixture-call-id",pngBase64:image.toString("base64")};
    const put = await client.callTool({name:"submit_paint",arguments:args});
    assert.equal(JSON.parse(put.content[0].text).job.id, id);
    assert.deepEqual(returnedPaint,image);
    assert.equal((await client.callTool({name:"submit_paint",arguments:{...args,pngBase64:"???"}})).isError,true);
  });
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
  await test("MCP inspect accepts named sections and rejects unbounded profiles", async () => {
    const panel = {
      bounds: [[-0.6, 0.6], [0, 1], [-0.1, 0.1]],
      levels: [0.2, 0.5, 0.8],
      min_depth_ratio: 0.015,
    };
    const profile = {
      subject: "measured-parts", up_axis: 1, depth_axis: 2, regions: { 薄板: panel },
    };
    const measure = (profile) => client.callTool({
      name: "inspect", arguments: { jobId: id, profile },
    });
    assert.equal((await measure(profile)).isError, undefined);
    assert.deepEqual(inspected, { profile });
    const tooMany = Object.fromEntries(Array.from({ length: 7 }, (_, i) => [`part${i}`, panel]));
    for (const regions of [
      {}, tooMany, { panel: { ...panel, levels: [0.2, 0.2, 0.2] } }, { panel, ["__proto__"]: panel },
    ]) {
      assert.equal((await measure({ ...profile, regions })).isError, true);
    }
  });
} finally {
  await client.close();
  await new Promise((r) => service.close(r));
  await unlink(tokenFile);
  await rmdir(hostDir);
}
