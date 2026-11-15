param([string]$CachedWeights = '')
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
