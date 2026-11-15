param([string]$CachedWeights = '')
$savedEnv = @{}
foreach ($name in @('UV_PYTHON_INSTALL_DIR', 'UV_CACHE_DIR', 'UV_NO_CONFIG', 'GIT_CONFIG_NOSYSTEM', 'GIT_CONFIG_GLOBAL')) { $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (!(Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' }
New-Item -ItemType Directory -Force .runtime, .runtime\model, .runtime\python, .runtime\uv-cache | Out-Null
$env:UV_PYTHON_INSTALL_DIR = Join-Path $root '.runtime\python'
$env:UV_CACHE_DIR = Join-Path $root '.runtime\uv-cache'
$env:UV_NO_CONFIG = '1'
$env:GIT_CONFIG_NOSYSTEM = '1'
$env:GIT_CONFIG_GLOBAL = Join-Path $root '.runtime\gitconfig'
if (!(Test-Path $env:GIT_CONFIG_GLOBAL)) { Set-Content -LiteralPath $env:GIT_CONFIG_GLOBAL -Value '' }
function Get-VerifiedFile([string]$Url, [string]$Destination, [string]$Hash, [string[]]$Extra = @()) {
  if ((Test-Path $Destination) -and (Get-FileHash $Destination -Algorithm SHA256).Hash.ToLower() -eq $Hash) { return }
  $tmp = "$Destination.download"
  try {
    & curl.exe -L --fail --retry 3 --retry-all-errors -sS @Extra -o $tmp $Url
    if ($LASTEXITCODE) { throw "Download failed: $Url" }
    if ((Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower() -ne $Hash) { throw "Checksum mismatch: $Url" }
    Move-Item -Force -LiteralPath $tmp -Destination $Destination
  } finally { if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Force } }
}
if (!(Test-Path .runtime\TripoSR\.git)) { git clone https://github.com/VAST-AI-Research/TripoSR.git .runtime\TripoSR; if ($LASTEXITCODE) { throw 'clone failed' } }
git -C .runtime\TripoSR checkout 107cefdc244c39106fa830359024f6a2f1c78871
if ($LASTEXITCODE) { throw 'checkout failed' }
uv python install --no-bin --no-registry 3.11.15
if ($LASTEXITCODE) { throw 'python install failed' }
if (!(Test-Path .runtime\venv\Scripts\python.exe)) { uv venv --python 3.11.15 .runtime\venv; if ($LASTEXITCODE) { throw 'venv failed' } }
uv pip install --python .runtime\venv\Scripts\python.exe torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE) { throw 'torch install failed' }
uv pip install --python .runtime\venv\Scripts\python.exe -r engine\requirements.txt
if ($LASTEXITCODE) { throw 'dependency install failed' }
$rev = '5b521936b01fbe1890f6f9baed0254ab6351c04a'
Get-VerifiedFile "https://huggingface.co/stabilityai/TripoSR/resolve/$rev/config.yaml" '.runtime\model\config.yaml' '74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62'
Get-VerifiedFile 'https://huggingface.co/facebook/dino-vitb16/resolve/f205d5d8e640a89a2b8ef0369670dfc37cc07fc2/config.json' '.runtime\model\dino-config.json' 'b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c'
$expected = '429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee'
if ($CachedWeights -and !(Test-Path '.runtime\model\model.ckpt')) {
  if (!(Test-Path -LiteralPath $CachedWeights) -or (Get-FileHash -LiteralPath $CachedWeights -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Cached model does not match official SHA-256' }
  Copy-Item -LiteralPath $CachedWeights -Destination '.runtime\model\model.ckpt.download'
  if ((Get-FileHash '.runtime\model\model.ckpt.download' -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Copied model checksum mismatch' }
  Move-Item -Force '.runtime\model\model.ckpt.download' '.runtime\model\model.ckpt'
}
Get-VerifiedFile "https://huggingface.co/stabilityai/TripoSR/resolve/$rev/model.ckpt" '.runtime\model\model.ckpt' $expected
Write-Output 'Python setup complete. Input images remain local.'

} finally {
  foreach ($name in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process') }
}
