# QuixiGate Windows installer (https://github.com/QuixiAI/QuixiGate)
# Usage (PowerShell): irm https://raw.githubusercontent.com/QuixiAI/QuixiGate/main/install.ps1 | iex

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.ServicePointManager]::SecurityProtocol

$Repo = if ($env:QUIXIGATE_REPO) { $env:QUIXIGATE_REPO } else { "QuixiAI/QuixiGate" }
$Version = if ($env:QUIXIGATE_VERSION) { $env:QUIXIGATE_VERSION } else { "latest" }

$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

# Resolve native Windows install directory:
# 1. Explicit QUIXIGATE_INSTALL_DIR override if set
# 2. C:\Program Files\QuixiGate\bin when running in an elevated Administrator shell
# 3. %LOCALAPPDATA%\Programs\QuixiGate\bin for standard user shells (Windows standard per-user Programs location)
if ($env:QUIXIGATE_INSTALL_DIR) {
    $InstallDir = $env:QUIXIGATE_INSTALL_DIR
    $PathTarget = "User"
} elseif ($IsAdmin -and $env:ProgramFiles) {
    $InstallDir = Join-Path $env:ProgramFiles "QuixiGate\bin"
    $PathTarget = "Machine"
} else {
    $LocalPrograms = if ($env:LOCALAPPDATA) {
        Join-Path $env:LOCALAPPDATA "Programs"
    } else {
        Join-Path $HOME "AppData\Local\Programs"
    }
    $InstallDir = Join-Path $LocalPrograms "QuixiGate\bin"
    $PathTarget = "User"
}

function Write-Step([string]$Message) {
    Write-Host "==> $Message"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$ArchRaw = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
$CandidateTargets = switch -Regex ($ArchRaw) {
    "ARM64" { @("aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc") }
    "AMD64|x86_64" { @("x86_64-pc-windows-msvc") }
    default { @() }
}

$BaseUrl = if ($Version -eq "latest") {
    "https://github.com/$Repo/releases/latest/download"
} else {
    "https://github.com/$Repo/releases/download/$Version"
}

$InstalledPrebuilt = $false
$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("quixigate-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

try {
    foreach ($Target in $CandidateTargets) {
        $Archive = "quixigate-$Target.zip"
        $ArchivePath = Join-Path $TmpDir $Archive
        $ShaPath = "$ArchivePath.sha256"

        try {
            Invoke-WebRequest -Uri "$BaseUrl/$Archive" -OutFile $ArchivePath -UseBasicParsing
        } catch {
            continue
        }

        try {
            Invoke-WebRequest -Uri "$BaseUrl/$Archive.sha256" -OutFile $ShaPath -UseBasicParsing
            $Expected = ((Get-Content -Raw $ShaPath).Trim() -split '\s+')[0].ToLowerInvariant()
            $Actual = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($Expected -and ($Expected -ne $Actual)) {
                throw "SHA-256 checksum mismatch for $Archive (expected $Expected, got $Actual)"
            }
        } catch {
            if ($_.Exception.Message -like "SHA-256 checksum mismatch*") {
                throw
            }
        }

        $ExtractDir = Join-Path $TmpDir "extracted"
        if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
        Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force

        $ExePath = Join-Path $ExtractDir "quixigate.exe"
        if (-not (Test-Path $ExePath)) {
            throw "Archive $Archive did not contain quixigate.exe"
        }

        $DestExe = Join-Path $InstallDir "quixigate.exe"
        Copy-Item -Path $ExePath -Destination $DestExe -Force
        $InstalledPrebuilt = $true
        Write-Step "Installed prebuilt quixigate ($Target) to $DestExe"
        break
    }

    if (-not $InstalledPrebuilt) {
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            throw "No prebuilt release found for Windows/$ArchRaw and 'cargo' is not installed. Install Rust via https://rustup.rs first."
        }
        Write-Step "Building quixigate from source via cargo (https://github.com/$Repo)..."
        $CargoStage = if ($env:TMP_CARGO_ROOT) { $env:TMP_CARGO_ROOT } else { Join-Path $TmpDir "cargo-root" }
        & cargo install --git "https://github.com/$Repo.git" --locked --root $CargoStage quixigate
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        $BuiltExe = Join-Path $CargoStage "bin\quixigate.exe"
        if (-not (Test-Path $BuiltExe)) {
            throw "cargo install did not produce quixigate.exe"
        }
        $DestExe = Join-Path $InstallDir "quixigate.exe"
        Copy-Item -Path $BuiltExe -Destination $DestExe -Force
        Write-Step "Installed quixigate to $DestExe"
    }
} finally {
    if (Test-Path $TmpDir) {
        Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
    }
}

$NormInstall = $InstallDir.TrimEnd('\')
$SessionEntries = ($env:PATH -split ';') | ForEach-Object { $_.TrimEnd('\') }
if ($SessionEntries -notcontains $NormInstall) {
    $env:PATH = "$InstallDir;$env:PATH"
}

$PersistedPath = [Environment]::GetEnvironmentVariable("Path", $PathTarget)
$PersistedEntries = if ($PersistedPath) { ($PersistedPath -split ';') | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') } } else { @() }
if ($PersistedEntries -notcontains $NormInstall) {
    $UpdatedPath = if ($PersistedPath) { "$InstallDir;$PersistedPath" } else { $InstallDir }
    [Environment]::GetEnvironmentVariable("Path", $PathTarget) | Out-Null
    [Environment]::SetEnvironmentVariable("Path", $UpdatedPath, $PathTarget)
    Write-Host "`nAdded $InstallDir to your $PathTarget PATH."
}

Write-Host "`nQuickstart:`n  quixigate serve`n  quixigate login openrouter"
