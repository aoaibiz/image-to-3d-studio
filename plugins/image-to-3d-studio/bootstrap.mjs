import { resolve, join } from "node:path";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
const [homeArg, rootArg] = process.argv.slice(2);
if (!homeArg || !rootArg)
  throw Error(
    "Usage: node bootstrap.mjs <absolute-codex-home> <absolute-studio-checkout>",
  );
const home = resolve(homeArg),
  root = resolve(rootArg);
if (!existsSync(join(root, "clients", "mcp.mjs")))
  throw Error("Checkout does not contain clients/mcp.mjs");
if (!existsSync(home)) throw Error("Codex home does not exist");
const dir = join(home, "plugins");
mkdirSync(dir, { recursive: true });
writeFileSync(join(dir, "image-to-3d-studio-root.txt"), root + "\n", {
  encoding: "utf8",
});
console.log(
  "Configured checkout in " + join(dir, "image-to-3d-studio-root.txt"),
);
