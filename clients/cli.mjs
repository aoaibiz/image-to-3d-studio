#!/usr/bin/env node
import { readFile, stat } from "node:fs/promises";
import { extname } from "node:path";
import {
  BASE,
  claimPaint, submitPaint,
  json,
  submit,
  watch,
  jobPath,
  artifactPath,
  exportFile,
  sendJson,
} from "./service.mjs";
import { profileSchema, reviewSchema } from "./schemas.mjs";
const [command, ...args] = process.argv.slice(2);
const output = (x) => console.log(JSON.stringify(x, null, 2));
try {
  switch (command) {
    case "diagnose":
      output(await json("/api/doctor"));
      break;
    case "generate": {
      const [file, key] = args;
      if (!file || !key) throw Error("generate <image> <stable-request-key>");
      const mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
      }[extname(file).toLowerCase()];
      output(await submit(await readFile(file), mime, key));
      break;
    }
    case "status":
      output(await json(jobPath(args[0])));
      break;
    case "claim-paint":
      output(await claimPaint(args[0]));
      break;
    case "submit-paint": {
      const [id, claim, sha, call, file] = args;
      if (!file || (await stat(file)).size > 20971520) throw Error("20MB以下の画像生成PNGを指定してください。");
      output(await submitPaint(id, claim, sha, call, await readFile(file)));
      break;
    }
    case "watch":
      output(await watch(args[0], (event) => output(event)));
      break;
    case "cancel":
    case "resume":
      output(await json(jobPath(args[0]) + "/" + command, { method: "POST" }));
      break;
    case "show": {
      const { job } = await json(jobPath(args[0]));
      console.log(job.status === "review_hold" ? BASE + "/?job=" + job.id : BASE + artifactPath(args[0], "offline-preview.html"));
      break;
    }
    case "inspect": {
      if (args[1]) {
        const profile = profileSchema.parse(JSON.parse(await readFile(args[1], "utf8")));
        output(await sendJson(jobPath(args[0]) + "/inspect", { profile }));
      } else {
        const { job } = await json(jobPath(args[0]));
        output(await json(artifactPath(args[0], job.status === "review_hold" ? "review-hold.json" : "inspection.json")));
      }
      break;
    }
    case "review":
      if (!args[1]) throw Error("review <job-id> <review-json>");
      output(
        await sendJson(
          jobPath(args[0]) + "/review",
          reviewSchema.parse(JSON.parse(await readFile(args[1], "utf8"))),
        ),
      );
      break;
    case "export": {
      const [id, name, destination, option] = args;
      if (!destination || (option && option !== "--overwrite"))
        throw Error(
          "export <job-id> <artifact-filename> <destination> [--overwrite]",
        );
      await exportFile(id, name, destination, option === "--overwrite");
      console.log(destination);
      break;
    }
    default:
      throw Error(
        "Commands: diagnose, generate <image> <stable-request-key>, status, watch, cancel, resume, show, export, inspect, claim-paint <id>, submit-paint <id> <claim> <request-sha> <tool-call-id> <png>",
      );
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
