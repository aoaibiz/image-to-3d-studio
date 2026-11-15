import { spawnSync } from "node:child_process";
import { test } from "node:test";
import assert from "node:assert/strict";
for (const port of ["0", "65536", "127.0.0.1:80", "1/../../", "-1"]) {
  test(`invalid STUDIO_PORT ${port} is rejected`, () => {
    const r = spawnSync(process.execPath, ["cli.mjs", "diagnose"], {
      cwd: import.meta.dirname,
      env: { ...process.env, STUDIO_PORT: port },
      encoding: "utf8",
    });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /STUDIO_PORT/);
  });
}
