#!/usr/bin/env node
// Print locally resolved paths only. Never edit host settings or read credentials.
import { fileURLToPath } from "node:url";

const [host, ...rest] = process.argv.slice(2);
const name = "image-to-3d-studio";
const launch = { command: process.execPath, args: [fileURLToPath(new URL("./mcp.mjs", import.meta.url))] };
if (rest.length || !["stdio", "claude", "cursor", "vscode", "copilot", "gemini", "codex"].includes(host)) {
  console.error("Usage: node clients/agent-config.mjs stdio|claude|cursor|vscode|copilot|gemini|codex");
  process.exitCode = 2;
} else if (host === "codex") {
  // Basic TOML strings accept JSON's path, quote and control escapes.
  const quote = s => JSON.stringify(s).replaceAll("\x7f", "\\u007f");
  console.log(`[mcp_servers.${name}]\ncommand = ${quote(launch.command)}\nargs = [${quote(launch.args[0])}]`);
} else {
  const server = host === "gemini" ? launch : { type: "stdio", ...launch };
  if (host === "copilot" || host === "vscode") server.tools = ["*"];
  const config = host === "stdio" ? launch : { mcpServers: { [name]: server } };
  console.log(JSON.stringify(config, null, 2));
}
