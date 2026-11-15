import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { json, submit, watch, jobPath, artifactChunk, claimPaint, submitPaint } from "./service.mjs";
const server = new McpServer({ name: "image-to-3d-studio", version: "1.0.0" });
const id = z.string().uuid();
const result = (value) => ({
  content: [{ type: "text", text: JSON.stringify(value) }],
});
const register = (name, description, inputSchema, fn) =>
  server.registerTool(
    name,
    { description, inputSchema },
    async (args, extra) => {
      try {
        return result(await fn(args, extra));
      } catch (e) {
        return { isError: true, content: [{ type: "text", text: e.message }] };
      }
    },
  );
register("diagnose", "Check the local studio service.", {}, () =>
  json("/api/doctor"),
);
register(
  "generate",
  "Submit base64 image bytes with a stable request key. Reusing the key is idempotent.",
  {
    publicCC0ToPixal: z.literal(true).optional().describe("Attest this particular image is public CC0 and authorize sending it to the third-party Hugging Face Space victor/pixal3d-studio. Requires the Pixal demo configuration."),
    imageBase64: z.string(),
    mime: z.enum(["image/png", "image/jpeg", "image/webp"]),
    requestKey: z.string().regex(/^[A-Za-z0-9._:-]{1,128}$/),
  },
  ({ imageBase64, mime, requestKey, publicCC0ToPixal }) => {
    if (imageBase64.length > Math.ceil((20 * 1024 * 1024) / 3) * 4)
      throw Error("画像は20MB以下で選び直してください。");
    const bytes = Buffer.from(imageBase64, "base64");
    if (
      !bytes.length ||
      bytes.length > 20 * 1024 * 1024 ||
      bytes.toString("base64") !== imageBase64
    )
      throw Error("画像のBase64形式を確認してください。");
    return submit(bytes, mime, requestKey, publicCC0ToPixal);
  },
);
register("status", "Read job status.", { jobId: id }, ({ jobId }) =>
  json(jobPath(jobId)),
);
register("claim_paint", "Claim the fixed image instruction and two local paths for an explicitly configured host job. Only generate when generate_once is true.", { jobId: id }, ({ jobId }) => claimPaint(jobId));
register("submit_paint", "Return the original PNG from the authorized native ImageGen tool. Repeating the same result is idempotent.", {
  jobId: id, claimId: z.string().regex(/^[A-Za-z0-9_-]{43}$/),
  requestSha256: z.string().regex(/^[0-9a-f]{64}$/), toolCallId: z.string().regex(/^[A-Za-z0-9_-]{1,160}$/),
  pngBase64: z.string().min(4).max(Math.ceil(20971520 / 3) * 4),
}, ({jobId, claimId, requestSha256, toolCallId, pngBase64}) => {
  const bytes = Buffer.from(pngBase64, "base64");
  if (bytes.toString("base64") !== pngBase64) throw Error("PNGのBase64形式を確認してください。");
  return submitPaint(jobId, claimId, requestSha256, toolCallId, bytes);
});
register(
  "wait",
  "Wait for job completion through the event stream.",
  { jobId: id },
  ({ jobId }, extra) =>
    watch(
      jobId,
      async (event) => {
        const token = extra._meta?.progressToken;
        if (token !== undefined)
          await extra.sendNotification({
            method: "notifications/progress",
            params: {
              progressToken: token,
              progress: Number(event.id) || 0,
              message: JSON.stringify(event.payload),
            },
          });
      },
      extra.signal,
    ),
);
for (const action of ["cancel", "resume"])
  register(action, `${action} a job.`, { jobId: id }, ({ jobId }) =>
    json(jobPath(jobId) + "/" + action, { method: "POST" }),
  );
register(
  "artifact_chunk",
  "Retrieve an artifact as base64 chunks of at most 256 KiB; nextOffset is null at end.",
  {
    jobId: id,
    filename: z.string().min(1).max(255),
    offset: z.number().int().min(0).max(67108864),
    limit: z.number().int().min(1).max(262144),
  },
  ({ jobId, filename, offset, limit }) =>
    artifactChunk(jobId, filename, offset, limit),
);

import { profileSchema, reviewSchema } from "./schemas.mjs";
import { sendJson } from "./service.mjs";
register(
  "inspect",
  "Measure cross-sections in 1–6 named regions, or use the cat head/body profile. A new inspection invalidates the prior manual review.",
  { jobId: id, profile: profileSchema },
  ({ jobId, profile }) => sendJson(jobPath(jobId) + "/inspect", { profile }),
);
register(
  "review",
  "Record a manual review of the current GLB after viewing both material and clay. Job state remains separate from manual review.",
  { jobId: id, review: reviewSchema },
  ({ jobId, review }) => sendJson(jobPath(jobId) + "/review", review),
);

await server.connect(
  new StdioServerTransport(process.stdin, process.stdout, {
    maxBufferSize: 30 * 1024 * 1024,
  }),
);
