import { dirname, join, resolve } from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
const pluginRoot = dirname(fileURLToPath(import.meta.url));
const directRoot = resolve(pluginRoot, "../..");
const codexHome = resolve(pluginRoot, "../../../../..");
const isCache =
  dirname(dirname(dirname(dirname(pluginRoot)))) === join(codexHome, "plugins");
const pointer = join(codexHome, "plugins", "image-to-3d-studio-root.txt");
let root = existsSync(join(directRoot, "clients", "mcp.mjs"))
  ? directRoot
  : null;
if (!root && isCache && existsSync(pointer))
  root = readFileSync(pointer, "utf8").trim();
if (!root || !existsSync(join(resolve(root), "clients", "mcp.mjs"))) {
  console.error(
    "Studio checkout not configured. Run the documented bootstrap with an explicit Codex home and checkout, then npm ci in clients.",
  );
  process.exit(1);
}
const child = spawn(
  process.execPath,
  [join(resolve(root), "clients", "mcp.mjs")],
  { stdio: "inherit", windowsHide: true },
);
child.on("error", (e) => {
  console.error(e.message);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});
