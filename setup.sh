#!/bin/sh
set -eu

cd "$(dirname "$0")"
root=$(pwd -P)
for tool in node npm; do
    command -v "$tool" >/dev/null || { echo "$tool is required (install Node.js 22 with npm)." >&2; exit 1; }
done
node -e 'if (Number(process.versions.node.split(".")[0]) < 22) { console.error("Node.js 22 or newer is required."); process.exit(1); }'
runtime="$root/.runtime"
for dir in "$runtime" "$runtime/validator" "$runtime/npm-cache" "$runtime/browsers"; do
    [ ! -L "$dir" ] || { echo "Use a dedicated checkout ($dir is a symlink)." >&2; exit 1; }
done
mkdir -p "$runtime"
lock="$runtime/.studio-install.lock"
mkdir "$lock" 2>/dev/null || { echo "Setup is active. After an interrupted setup, remove the empty $lock once no installer is running." >&2; exit 1; }
trap 'rmdir "$lock"' EXIT
trap 'exit 1' HUP INT TERM
marker="$runtime/.studio-posix-v1"
[ ! -L "$marker" ] || { echo "Unexpected installer marker. Use a fresh checkout." >&2; exit 1; }
if [ ! -f "$marker" ]; then
    for dir in "$runtime/validator" "$runtime/npm-cache" "$runtime/browsers"; do
        [ ! -e "$dir" ] || { echo "Found unmanaged $dir. Choose a fresh checkout." >&2; exit 1; }
    done
    : > "$marker"
fi
sh "$root/engine/setup-posix.sh"
mkdir -p "$runtime/npm-cache" "$runtime/validator"
export npm_config_cache="$runtime/npm-cache"
npm ci --prefix "$root/clients"
npm ci --include=dev --prefix "$root/viewer"
cp "$root/tools/validator/package.json" "$root/tools/validator/package-lock.json" "$runtime/validator/"
npm ci --ignore-scripts --prefix "$runtime/validator"
install -m 755 "$root/tools/validator/gltf_validator" "$runtime/validator/gltf_validator"

# Each checkout uses the Chromium build paired with its locked Playwright version.
PLAYWRIGHT_BROWSERS_PATH="$runtime/browsers" node "$root/viewer/node_modules/playwright/cli.js" install chromium
PLAYWRIGHT_BROWSERS_PATH="$runtime/browsers" node -e '
    const { chromium } = require("./viewer/node_modules/playwright");
    chromium.launch({headless: true}).then(browser => browser.close()).catch(error => {
        console.error(error.message);
        console.error("Chromium could not start. Check the operating system requirements in README.md, then rerun setup.sh.");
        process.exitCode = 1;
    });
'
echo 'Studio is ready. From this directory, run .runtime/venv/bin/python -m studio.server'
echo 'Open http://127.0.0.1:47631'
