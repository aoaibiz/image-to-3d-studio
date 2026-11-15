import { createWriteStream } from "node:fs";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const port = process.env.STUDIO_PORT ?? "47631";
if (!/^[0-9]{1,5}$/.test(port) || Number(port) < 1 || Number(port) > 65535)
  throw Error("STUDIO_PORT must be an integer from 1 to 65535");
export const BASE = `http://127.0.0.1:${Number(port)}`;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export function jobPath(id) {
  if (!uuid.test(id)) throw Error("Invalid job ID");
  return `/api/jobs/${id}`;
}
export function artifactPath(id, name) {
  if (!/^[a-zA-Z0-9_.-]+$/.test(name) || name === "." || name === "..")
    throw Error("Invalid artifact filename");
  return `${jobPath(id)}/artifacts/${encodeURIComponent(name)}`;
}
export async function request(path, options = {}) {
  const response = await fetch(BASE + path, {
    ...options,
    redirect: "error",
    headers: { "X-Studio-Client": "1", ...options.headers },
  });
  if (!response.ok)
    throw Error(
      `Studio ${response.status}: ${(await response.text()).slice(0, 1000)}`,
    );
  return response;
}
export async function json(path, options) {
  return (await request(path, options)).json();
}
async function hostHeaders() {
  const file = process.env.STUDIO_HOST_TOKEN_FILE || new URL("../jobs/studio/host-access.token", import.meta.url);
  let token;
  try { token = (await readFile(file, "utf8")).trim(); } catch {
    throw Error("ホスト生成の初回設定と STUDIO_HOST_TOKEN_FILE を確認してください。");
  }
  if (!/^[A-Za-z0-9_-]{43}$/.test(token)) throw Error("ホストの設定が無効です。");
  return { "X-Studio-Host-Token": token };
}
export async function claimPaint(id) {
  return json(jobPath(id) + "/host/claim", { method: "POST", headers: await hostHeaders() });
}
export async function submitPaint(id, claim, requestSha, toolCall, png) {
  if (!/^[A-Za-z0-9_-]{43}$/.test(claim) || !/^[0-9a-f]{64}$/.test(requestSha)
      || !/^[A-Za-z0-9_-]{1,160}$/.test(toolCall) || !png.length || png.length > 20971520)
    throw Error("受付番号、要求ハッシュ、画像生成ID、PNGサイズを確認してください。");
  return json(jobPath(id) + "/host/submit", { method: "POST", headers: {
    ...await hostHeaders(), "Content-Type": "image/png", "X-Studio-Claim": claim,
    "X-Studio-Request-Sha256": requestSha, "X-Studio-Tool-Call": toolCall,
    "X-Studio-Paint-Sha256": createHash("sha256").update(png).digest("hex"),
  }, body: png });
}
export function submit(bytes, mime, key, publicCC0ToPixal = false) {
  if (!["image/png", "image/jpeg", "image/webp"].includes(mime))
    throw Error("Image must be PNG, JPEG, or WebP");
  if (!/^[A-Za-z0-9._:-]{1,128}$/.test(key || ""))
    throw Error(
      "A stable request key of 1–128 letters, digits, dot, underscore, colon or hyphen is required",
    );
  return json("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": mime, "Idempotency-Key": key,
      ...(publicCC0ToPixal === true ? {
        "X-Studio-Pipeline": "pixal3d-space-v1",
        "X-Studio-CC0-Source": createHash("sha256").update(bytes).digest("hex"),
      } : {}),
    },
    body: bytes,
  });
}
export async function watch(id, onEvent = () => {}, signal) {
  const res = await request(jobPath(id) + "/events", {
    headers: { Accept: "text/event-stream", "X-Studio-Yield-Host": "1" },
    signal,
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let match;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const data = frame
          .split(/\r?\n/)
          .filter((x) => x.startsWith("data:"))
          .map((x) => x.slice(5).trimStart())
          .join("\n");
        if (!data) continue;
        const event = JSON.parse(data);
        await onEvent(event);
      }
      if (done) {
        const current = await json(jobPath(id));
        if (
          ["needs_review", "review_hold", "failed", "cancelled", "interrupted", "awaiting_host_image"].includes(
            current.job?.status,
          )
        )
          return current;
        throw Error("Event stream ended before a terminal state");
      }
      if (buffer.length > 1024 * 1024) throw Error("SSE frame too large");
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}
export async function exportFile(id, name, destination, overwrite = false) {
  const res = await request(artifactPath(id, name));
  await pipeline(
    Readable.fromWeb(res.body),
    createWriteStream(destination, { flags: overwrite ? "w" : "wx" }),
  );
}
export async function artifactChunk(id, name, offset, limit) {
  if (
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    offset > 64 * 1024 * 1024 ||
    !Number.isSafeInteger(limit) ||
    limit < 1 ||
    limit > 256 * 1024
  )
    throw Error("Invalid chunk bounds");
  const res = await request(artifactPath(id, name));
  const reader = res.body.getReader();
  let position = 0,
    chunks = [],
    total = 0,
    eof = false;
  try {
    while (total <= limit) {
      const { value, done } = await reader.read();
      if (done) {
        eof = true;
        break;
      }
      const end = position + value.length;
      if (end > offset) {
        const part = value.subarray(
          Math.max(0, offset - position),
          Math.min(value.length, offset + limit + 1 - position),
        );
        chunks.push(part);
        total += part.length;
      }
      position = end;
      if (position > offset + limit) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  const data = Buffer.concat(chunks);
  return {
    offset,
    data: data.subarray(0, limit).toString("base64"),
    nextOffset:
      data.length > limit || !eof
        ? offset + Math.min(data.length, limit)
        : null,
    encoding: "base64",
  };
}

export async function sendJson(path, value) {
  return json(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}
