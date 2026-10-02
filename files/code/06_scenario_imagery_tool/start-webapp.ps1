# Start the massing tool. Double-click start-webapp.cmd, or run this file.
#
# The key survives reboots: IMAGE_API_KEY is a persistent user environment variable, so it is in
# the registry for this Windows account. It is read from the registry rather than this shell's
# environment, so a key set after this shell was opened still counts. server.py reads
# OPENAI_API_KEY, which is what the OpenAI SDK convention calls it, and the relay speaks the same
# protocol under a different address -- this script only renames it.
#
# .relay-key is the fallback for a machine where that variable was never set: save-key.ps1
# writes the key there encrypted with Windows DPAPI, readable only by this account on this
# machine. NEWAPI_API_KEY, the name the first relay's key had, is the last resort. None of these
# paths ever puts the key on screen or in a log.
#
# The address is the one thing that cannot be guessed. Set it once and it stays:
#   setx OPENAI_BASE_URL "https://<relay-host>/v1"
# Without it the code falls back to api.openai.com, where a relay key is rejected -- the server
# still starts and the page still loads, and only generation fails, which is a confusing way to
# find out.

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = 'D:\APP\Anaconda3\envs\scats\python.exe'

# --- key ---------------------------------------------------------------------------------
$key = [Environment]::GetEnvironmentVariable('IMAGE_API_KEY', 'User')
if (-not $key) { $key = $env:IMAGE_API_KEY }
$source = 'IMAGE_API_KEY'
if (-not $key) {
    $store = Join-Path $here '.relay-key'
    if (Test-Path $store) {
        $secure = Get-Content $store | ConvertTo-SecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        $key = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        $source = '.relay-key'
    }
}
if (-not $key) {
    $key = [Environment]::GetEnvironmentVariable('NEWAPI_API_KEY', 'User')
    $source = 'NEWAPI_API_KEY (the earlier relay)'
}
if (-not $key) {
    Write-Host ''
    Write-Host 'No relay key found.' -ForegroundColor Red
    Write-Host '  either set it once:   setx IMAGE_API_KEY "<key>"'
    Write-Host '  or store it encrypted: powershell -ExecutionPolicy Bypass -File save-key.ps1'
    Write-Host ''
    exit 1
}
$env:OPENAI_API_KEY = $key
Write-Host "key loaded from $source" -ForegroundColor Green

# --- address -----------------------------------------------------------------------------
# The registry first: setx writes there but only reaches processes started afterwards, so a
# shell opened before it was set still carries the old address, and a server launched from it
# would talk to the old relay. Then this shell's environment, then the built-in default.
$url = [Environment]::GetEnvironmentVariable('OPENAI_BASE_URL', 'User')
if ($url) { $env:OPENAI_BASE_URL = $url }
if (-not $env:OPENAI_BASE_URL) {
    $env:OPENAI_BASE_URL = 'https://cpapro.jsuer.com/v1'
    Write-Host 'relay address not in the environment, using the built-in default' -ForegroundColor Yellow
}
Write-Host "relay: $env:OPENAI_BASE_URL" -ForegroundColor Green

# The URL is read once at import time in massing_generation_v4, so it has to be right in the
# environment this process starts with -- setting it afterwards changes nothing. A relay key is
# refused at the official endpoint with a 401 that reads like a bad key, not a wrong address.
if ($env:OPENAI_BASE_URL -like '*api.openai.com*') {
    Write-Host 'that is the official endpoint, and a relay key will be refused there' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $python)) {
    Write-Host "python not found at $python" -ForegroundColor Red
    exit 1
}

# Windows lets a second server bind a port the first one already holds, and then routes
# requests to whichever it feels like. A stale instance started with the old address will
# answer some of them with the old relay, a failure that looks like a bad key rather than a
# stale process. So clear the port first.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*server.py*' } |
    ForEach-Object {
        Write-Host "  stopping earlier instance, PID $($_.ProcessId)" -ForegroundColor DarkGray
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host 'starting on http://127.0.0.1:8000 -- press Ctrl+C to stop' -ForegroundColor Cyan
Start-Process 'http://127.0.0.1:8000'
& $python (Join-Path $here 'server.py')
