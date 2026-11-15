param([string]$CachedWeights = '', [switch]$TripoSG, [switch]$SendPhotoAndGuidesToOpenAI, [switch]$UseCodexHost, [switch]$ChatGPTPhotoAnalysis)
if ($ChatGPTPhotoAnalysis -and (!$SendPhotoAndGuidesToOpenAI -or $UseCodexHost)) {
  throw 'Use -ChatGPTPhotoAnalysis with -SendPhotoAndGuidesToOpenAI for the ChatGPT CLI observation and paint pipeline. -UseCodexHost selects a separate pipeline.'
}
if ($UseCodexHost -and !$SendPhotoAndGuidesToOpenAI) {
  throw 'Use -UseCodexHost with -SendPhotoAndGuidesToOpenAI to enable photo transfer for the native host.'
}
if ($SendPhotoAndGuidesToOpenAI) {
  $TripoSG = $true
  Write-Host 'Guided paint sends a photo and two rendered views to OpenAI using your ChatGPT plan. Check your remaining allowance and credit settings before enabling it. This preview cannot measure account charges.'
  if ($ChatGPTPhotoAnalysis) { Write-Host 'Photo analysis also sends one metadata-stripped photo to OpenAI for a text observation. Each job permits one observation and one ImageGen request.' }
}
$savedEnv = @{}
foreach ($name in @('npm_config_cache', 'npm_config_userconfig', 'npm_config_globalconfig')) { $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force .runtime, .runtime\validator, .runtime\npm-cache | Out-Null
& (Join-Path $PSScriptRoot 'engine\setup.ps1') -CachedWeights $CachedWeights
if (!(Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Node.js/npm required: https://nodejs.org/' }
$env:npm_config_cache = Join-Path $PSScriptRoot '.runtime\npm-cache'
$env:npm_config_userconfig = Join-Path $PSScriptRoot '.runtime\npm-user.npmrc'
$env:npm_config_globalconfig = Join-Path $PSScriptRoot '.runtime\npm-global.npmrc'
if (!(Test-Path $env:npm_config_userconfig)) { Set-Content -LiteralPath $env:npm_config_userconfig -Value '' }
if (!(Test-Path $env:npm_config_globalconfig)) { Set-Content -LiteralPath $env:npm_config_globalconfig -Value '' }
npm.cmd ci --prefix viewer
if ($LASTEXITCODE) { throw 'viewer npm ci failed' }
npm.cmd ci --prefix clients
if ($LASTEXITCODE) { throw 'clients npm ci failed' }
$zip = Join-Path $PSScriptRoot '.runtime\validator\gltf-validator-win64.zip'
$exe = Join-Path $PSScriptRoot '.runtime\validator\gltf_validator.exe'
$zipHash = 'c5068f51205deedc28acc3529ee7e11ee60e853454f673093398eba80142202c'
$exeHash = '4388a152ff90b68c6430ae03862e05e257a9d50a500ed7d0eb1cd420dc75ff96'
if (!(Test-Path $zip) -or (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $zipHash) {
  $tmp = "$zip.download"
  try {
    curl.exe -L --fail --retry 3 --retry-all-errors -sS -H 'Accept: application/octet-stream' -o $tmp 'https://api.github.com/repos/KhronosGroup/glTF-Validator/releases/assets/200903724'
    if ($LASTEXITCODE -or (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower() -ne $zipHash) { throw 'Khronos validator download or hash check failed' }
    Move-Item -Force $tmp $zip
  } finally { if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Force } }
}
Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $PSScriptRoot '.runtime\validator') -Force
if ((Get-FileHash $exe -Algorithm SHA256).Hash.ToLower() -ne $exeHash) { throw 'Khronos validator executable hash mismatch' }
if ($TripoSG) {
  $shapeOptions = @{}
  if ($env:STUDIO_TRIPOSG_RUNTIME) {
    $shapeOptions.RuntimeDir = & (Join-Path $PSScriptRoot '.runtime\venv\Scripts\python.exe') -c 'from engine.local_shape_job import runtime_path; print(runtime_path())'
    if ($LASTEXITCODE) { throw 'Use an absolute STUDIO_TRIPOSG_RUNTIME path.' }
  }
  & (Join-Path $PSScriptRoot 'engine\setup-triposg.ps1') @shapeOptions
  $engineOptions = @('--engine', 'triposg')
  if ($ChatGPTPhotoAnalysis) {
    $engineOptions = @('--engine', 'triposg-observed', '--chatgpt-photo-analysis', '--allow-send-to-openai')
  } elseif ($SendPhotoAndGuidesToOpenAI) {
    $visionRoot = & (Join-Path $PSScriptRoot '.runtime\venv\Scripts\python.exe') -c 'from engine.guided_shape_job import vision_runtime_path; print(vision_runtime_path())'
    if ($LASTEXITCODE) { throw 'Use an absolute STUDIO_VISUAL_BRIEF_RUNTIME path.' }
    & (Join-Path $PSScriptRoot '.runtime\venv\Scripts\python.exe') -m engine.visual_brief_setup $visionRoot
    if ($LASTEXITCODE) { throw 'Local visual analysis setup failed. Run the same setup again to resume.' }
    $guidedEngine = if ($UseCodexHost) { 'triposg-guided-host' } else { 'triposg-guided' }
    $engineOptions = @('--engine', $guidedEngine, '--allow-send-to-openai')
  }
  & (Join-Path $PSScriptRoot '.runtime\venv\Scripts\python.exe') -m studio.configure @engineOptions
  if ($LASTEXITCODE) { throw 'Could not select the TripoSG engine for this Studio checkout.' }
}
Write-Host 'Setup complete. Start: .\.runtime\venv\Scripts\python.exe -m studio.server'
Write-Host 'Open http://127.0.0.1:47631'

} finally {
  foreach ($name in $savedEnv.Keys) {
    if ($null -eq $savedEnv[$name]) {
      Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    } else {
      [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process')
    }
  }
}
