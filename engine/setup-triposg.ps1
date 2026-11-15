[CmdletBinding()]
param(
    [string]$RuntimeDir = (Join-Path (Split-Path -Parent $PSScriptRoot) '.runtime\triposg'),
    [string]$WeightsCache,
    [string]$SourceArchive,
    [string]$Python,
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'This experimental TripoSG installer targets 64-bit Windows on x86-64.'
}
$uv = (Get-Command uv -CommandType Application -ErrorAction Stop).Source
$runtime = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RuntimeDir)
$marker = Join-Path $runtime '.image-to-3d-triposg-v1'
$owner = 'image-to-3d-studio TripoSG v1'
if (Test-Path -LiteralPath $runtime) {
    if ((Get-Item -LiteralPath $runtime).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Choose a dedicated runtime directory without a junction or symlink.'
    }
    $owned = (Test-Path -LiteralPath $marker -PathType Leaf) -and ([IO.File]::ReadAllText($marker) -eq $owner)
    if (!$owned -and (Get-ChildItem -LiteralPath $runtime -Force | Select-Object -First 1)) {
        throw "Choose an empty directory for TripoSG. Existing files in $runtime belong to another runtime."
    }
} else {
    New-Item -ItemType Directory -Path $runtime | Out-Null
}
$lease = [IO.File]::Open((Join-Path $runtime '.install.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$savedPythonDir = $env:UV_PYTHON_INSTALL_DIR
try {
    if (!(Test-Path -LiteralPath $marker -PathType Leaf)) { [IO.File]::WriteAllText($marker, $owner) }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
    if (!$Python) {
        $installed = Join-Path (Split-Path -Parent $PSScriptRoot) '.runtime\venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $installed -PathType Leaf) {
            $Python = $installed
        } else {
            $opts = @('--no-config')
            if ($Offline) { $opts += '--offline' }
            & $uv @opts python install --no-bin --no-registry 3.11.15
            if ($LASTEXITCODE) { throw 'Python 3.11.15 unavailable. Use -Python with a local 64-bit Python 3.11.15 interpreter.' }
            $Python = & $uv @opts python find --managed-python 3.11.15
            if ($LASTEXITCODE) { throw 'Installed interpreter could not be located.' }
        }
    }
    $argsForInstaller = @('--runtime', $runtime, '--uv', $uv)
    if ($WeightsCache) { $argsForInstaller += @('--weights-cache', (Resolve-Path -LiteralPath $WeightsCache).Path) }
    if ($SourceArchive) { $argsForInstaller += @('--source-archive', (Resolve-Path -LiteralPath $SourceArchive).Path) }
    if ($Offline) { $argsForInstaller += '--offline' }
    & $Python -I -B -u (Join-Path $PSScriptRoot 'triposg_setup.py') @argsForInstaller
    if ($LASTEXITCODE) { throw 'TripoSG setup incomplete. Correct the reported cause and rerun the same command.' }
} finally {
    if ($null -eq $savedPythonDir) {
        Remove-Item -LiteralPath Env:UV_PYTHON_INSTALL_DIR -ErrorAction SilentlyContinue
    } else {
        $env:UV_PYTHON_INSTALL_DIR = $savedPythonDir
    }
    $lease.Dispose()
}
