# hf2browser installer for Windows — picks the binary, checks it, puts it on your PATH.
#
#   irm https://raw.githubusercontent.com/muthuishere/hf2browser/main/install.ps1 | iex
#
# Environment:
#   HF2BROWSER_VERSION   tag to install (default: the latest release)
#   HF2BROWSER_BIN_DIR   where to install (default: %LOCALAPPDATA%\hf2browser)

$ErrorActionPreference = 'Stop'

$repo    = 'muthuishere/hf2browser'
$version = if ($env:HF2BROWSER_VERSION) { $env:HF2BROWSER_VERSION } else { 'latest' }
$binDir  = if ($env:HF2BROWSER_BIN_DIR) { $env:HF2BROWSER_BIN_DIR } else { "$env:LOCALAPPDATA\hf2browser" }

$asset = 'hf2browser-windows-amd64.exe'
$base  = if ($version -eq 'latest') {
  "https://github.com/$repo/releases/latest/download"
} else {
  "https://github.com/$repo/releases/download/$version"
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  Write-Host "downloading $asset ($version)..."
  Invoke-WebRequest -Uri "$base/$asset" -OutFile "$tmp\$asset"

  # A binary you pipe from the internet deserves at least a checksum check.
  try {
    Invoke-WebRequest -Uri "$base/SHA256SUMS" -OutFile "$tmp\SHA256SUMS"
    $want = (Select-String -Path "$tmp\SHA256SUMS" -Pattern " $([regex]::Escape($asset))$").Line -split '\s+' | Select-Object -First 1
    $got  = (Get-FileHash -Algorithm SHA256 "$tmp\$asset").Hash.ToLower()
    if ($want -and $got -ne $want.ToLower()) {
      throw "checksum mismatch for $asset - refusing to install"
    }
    if ($want) { Write-Host 'checksum ok' }
  } catch [System.Net.WebException] {
    Write-Host 'checksums unavailable, skipping verification'
  }

  New-Item -ItemType Directory -Force -Path $binDir | Out-Null
  Move-Item -Force "$tmp\$asset" "$binDir\hf2browser.exe"
  Write-Host "installed $binDir\hf2browser.exe"

  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if ($userPath -notlike "*$binDir*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$binDir", 'User')
    Write-Host "added $binDir to your PATH - open a new terminal to pick it up"
  }

  Write-Host ''
  Write-Host 'next:  hf2browser serve      # search -> convert -> chat, all local'
  Write-Host '       hf2browser init       # write an editable hf2browser.json'
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
