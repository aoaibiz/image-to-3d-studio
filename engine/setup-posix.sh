#!/bin/sh
# Install the local TripoSR CPU engine. Use a dedicated checkout; no sudo.
set -eu
die() { printf '%s\n' "$*" >&2; exit 1; }
root=$(CDPATH='' cd -P "$(dirname "$0")/.." && pwd)
rt=$root/.runtime
src=$rt/TripoSR
upstream=https://github.com/VAST-AI-Research/TripoSR.git
commit=107cefdc244c39106fa830359024f6a2f1c78871
python_version=3.11.15
[ "$#" -eq 0 ] || die 'Usage: sh engine/setup-posix.sh'
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64|Darwin/x86_64)
    major=$(sw_vers -productVersion | cut -d. -f1)
    [ "$major" -ge 13 ] || die 'The pinned native wheels require macOS 13 or later.'
    ;;
  Linux/x86_64)
    libc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || die 'Linux requires glibc, not musl.'
    printf '%s\n' "$libc" | awk -F '[ .]' '$1=="glibc" && ($2>2 || ($2==2 && $3>=28)) {ok=1} END {exit !ok}' ||
      die 'This CPU environment targets Linux x86_64 with glibc 2.28 or later.'
    ;;
  *) die 'No native engine wheel set for this OS/CPU. Use a supported Linux x86_64 or macOS host.' ;;
esac
for tool in git uv curl; do
  command -v "$tool" >/dev/null 2>&1 || die "Install $tool first (uv: https://docs.astral.sh/uv/getting-started/installation/)."
done
if command -v sha256sum >/dev/null 2>&1; then
  digest() { sha256sum < "$1" | awk '{print $1}'; }
else
  command -v shasum >/dev/null 2>&1 || die 'Install shasum or sha256sum.'
  digest() { shasum -a 256 < "$1" | awk '{print $1}'; }
fi
umask 077
for dir in "$rt" "$src" "$rt/venv" "$rt/python" "$rt/uv-cache" "$rt/model" "$rt/u2net"; do
  [ ! -L "$dir" ] || die "Refusing symlink: $dir. Use a dedicated checkout."
done
mkdir -p "$rt"
lock=$rt/.setup-posix.lock
mkdir "$lock" 2>/dev/null || die "Installer active or interrupted. If no installer is running, remove the empty $lock and retry."
part=''
cleanup() {
  [ -z "$part" ] || rm -f "$part"
  rmdir "$lock"
}
trap cleanup 0
trap 'exit 1' HUP INT TERM
marker=$rt/.engine-posix-v1
[ ! -L "$marker" ] || die "Refusing symlink: $marker"
if [ ! -f "$marker" ]; then
  for dir in "$rt/venv" "$src" "$rt/model" "$rt/u2net" "$rt/python" "$rt/uv-cache"; do
    [ ! -e "$dir" ] || die "Unmanaged $dir already exists; use a fresh checkout. No existing environment was changed."
  done
  : > "$marker"
fi
mkdir -p "$rt/model" "$rt/u2net" "$rt/python" "$rt/uv-cache"
export UV_PYTHON_INSTALL_DIR="$rt/python" UV_CACHE_DIR="$rt/uv-cache" UV_NO_CONFIG=1
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
if [ ! -e "$src" ]; then
  git clone --no-checkout "$upstream" "$src"
  git -C "$src" -c advice.detachedHead=false checkout --detach "$commit"
fi
[ "$(git -C "$src" rev-parse HEAD)" = "$commit" ] &&
  [ "$(git -C "$src" remote get-url origin)" = "$upstream" ] &&
  [ -z "$(git -C "$src" status --porcelain)" ] ||
  die "Unexpected or modified TripoSR checkout in $src. Use a fresh checkout."
uv python install --no-bin "$python_version"
py=$rt/venv/bin/python
if [ ! -x "$py" ]; then
  [ ! -e "$rt/venv" ] || die "Incomplete venv: $rt/venv. Use a fresh checkout."
  uv venv --managed-python --python "$python_version" "$rt/venv"
fi
"$py" -I -c 'import platform, sys; sys.exit(0 if platform.python_version()==sys.argv[1] and sys.prefix!=sys.base_prefix else "Unexpected interpreter in the dedicated venv")' "$python_version"
uv pip install --python "$py" -r "$root/engine/requirements-posix.txt"

# Never install interrupted downloads or bytes that differ from the pinned SHA-256.
download() {
  dest=$1
  expected=$2
  shift 2
  if [ -f "$dest" ] && [ "$(digest "$dest")" = "$expected" ]; then return 0; fi
  [ ! -d "$dest" ] || die "Expected a file: $dest"
  part=$(mktemp "$dest.XXXXXXXX")
  for url do
    if curl -q --proto '=https' --proto-redir '=https' --fail --location --retry 2 --connect-timeout 20 \
        --speed-limit 1024 --speed-time 60 --silent --show-error -H 'Accept: application/octet-stream' --output "$part" "$url" &&
        [ "$(digest "$part")" = "$expected" ]; then
      mv -f "$part" "$dest"
      part=''
      return 0
    fi
  done
  die "Failed to obtain verified $(basename "$dest"); check the network and rerun setup."
}
model=https://huggingface.co/stabilityai/TripoSR/resolve/5b521936b01fbe1890f6f9baed0254ab6351c04a
download "$rt/model/config.yaml" 74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62 "$model/config.yaml"
download "$rt/model/dino-config.json" b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c \
  https://huggingface.co/facebook/dino-vitb16/resolve/f205d5d8e640a89a2b8ef0369670dfc37cc07fc2/config.json
download "$rt/model/model.ckpt" 429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee "$model/model.ckpt"
download "$rt/u2net/u2net.onnx" 8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491 \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
  "https://api.github.com/repos/danielgatis/rembg/releases/assets/85857932?request=$(date +%s)"
printf '%s\n' 'Local TripoSR dependencies and weights installed. Input images stay on this machine.'
