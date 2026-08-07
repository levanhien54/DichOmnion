param(
    [string]$RuntimeRoot = (Join-Path $env:TEMP "dichomnion-runpod-runtime"),
    [string]$GatewayUrl = "http://127.0.0.1:8787",
    [switch]$AllowDegradedAvailability,
    [ValidateRange(1, 16)]
    [int]$MaxCandidates = 3
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$apiKeyFile = Join-Path $repoRoot "hungkingface.txt"
$gatewayVarsFile = Join-Path $repoRoot "apps\gateway\.dev.vars"
$specPath = Join-Path $RuntimeRoot "pod.v2.production.json"
$placementPath = Join-Path $RuntimeRoot "placement.v1.production.json"
$statePath = Join-Path $RuntimeRoot "provision.intent.json"

if (-not (Test-Path -LiteralPath $apiKeyFile) -or -not (Test-Path -LiteralPath $gatewayVarsFile)) {
    throw "server_credentials_missing"
}
foreach ($path in @($specPath, $placementPath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "provision_artifact_missing" }
}

$runpodLine = Get-Content -LiteralPath $apiKeyFile |
    Where-Object { $_ -match '^\s*api\s+runpod\s*:\s*(rpa_[A-Za-z0-9]{32,256})\s*$' } |
    Select-Object -First 1
if (-not $runpodLine -or $runpodLine -notmatch '^\s*api\s+runpod\s*:\s*(rpa_[A-Za-z0-9]{32,256})\s*$') {
    throw "runpod_api_key_invalid"
}
$runpodApiKey = $Matches[1]

$adminLine = Get-Content -LiteralPath $gatewayVarsFile |
    Where-Object { $_ -match '^\s*WORKER_TARGET_ADMIN_TOKEN\s*=\s*"([^"]+)"\s*$' } |
    Select-Object -First 1
if (-not $adminLine -or $adminLine -notmatch '^\s*WORKER_TARGET_ADMIN_TOKEN\s*=\s*"([^"]+)"\s*$') {
    throw "gateway_admin_token_missing"
}
$gatewayAdminToken = $Matches[1]
if ([string]::IsNullOrWhiteSpace($gatewayAdminToken)) { throw "gateway_admin_token_missing" }

$existing = @(Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'sdk_controller\.py.*auto-provision' } |
    Select-Object -ExpandProperty ProcessId)
if ($existing.Count -gt 0) { throw "controller_already_running" }

$env:RUNPOD_API_KEY = $runpodApiKey
$env:GATEWAY_ADMIN_URL = $GatewayUrl
$env:GATEWAY_ADMIN_TOKEN = $gatewayAdminToken
$env:RUNPOD_PROVISION_SPEC_PATH = $specPath
$env:RUNPOD_PROVISION_PLACEMENT_PATH = $placementPath
$env:RUNPOD_PROVISION_STATE_PATH = $statePath
$env:RUNPOD_PROVISION_MAX_HOURLY_COST_USD = "0.50"
$env:RUNPOD_PROVISION_MAX_RUNTIME_HOURS = "1.50"
$env:RUNPOD_PROVISION_MAX_COMPUTE_COST_USD = "0.75"
$env:RUNPOD_PROVISION_MAX_CANDIDATES = [string]$MaxCandidates
$env:RUNPOD_PROVISION_MIN_MEMORY_GB = "48"
$env:RUNPOD_PROVISION_ALLOW_DEGRADED_AVAILABILITY = $(if ($AllowDegradedAvailability) { "1" } else { "0" })
$env:RUNPOD_PROVISION_RECONCILE_TIMEOUT_SECONDS = "120"
$env:RUNPOD_PROVISION_CLEANUP_TIMEOUT_SECONDS = "180"
$env:RUNPOD_PROVISION_POLL_SECONDS = "3"
$env:RUNPOD_TARGET_TTL_SECONDS = "90"
$env:RUNPOD_HEARTBEAT_SECONDS = "20"
$env:RUNPOD_PROBE_INTERVAL_SECONDS = "5"
$env:RUNPOD_READY_TIMEOUT_SECONDS = "1800"
$env:RUNPOD_UNHEALTHY_GRACE_SECONDS = "120"
$env:RUNPOD_WORKER_TRANSPORT = "runpod_proxy"
$env:RUNPOD_WORKER_MAX_REQUEST_MS = "90000"
foreach ($proxyName in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR")) {
    Remove-Item "Env:$proxyName" -ErrorAction SilentlyContinue
}

$logRoot = Join-Path $env:TEMP "dichomnion-runpod-controller"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stdout = Join-Path $logRoot "stdout.log"
$stderr = Join-Path $logRoot "stderr.log"
$python = Join-Path $repoRoot "apps\gpu-worker\.runpod-controller-venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "controller_venv_missing" }
$process = Start-Process -FilePath $python `
    -ArgumentList @("-u", "runpod\sdk_controller.py", "--auto-provision", "--allow-resume", "--max-hourly-cost-usd", "0.50") `
    -WorkingDirectory (Join-Path $repoRoot "apps\gpu-worker") `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

[pscustomobject]@{
    controllerPid = $process.Id
    stdout = $stdout
    stderr = $stderr
    maxHourlyUsd = "0.50"
    maxCandidates = $MaxCandidates
    degradedAvailabilityExplicit = [bool]$AllowDegradedAvailability
} | ConvertTo-Json -Depth 3
