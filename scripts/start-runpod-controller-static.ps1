param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9]{8,32}$')]
    [string]$PodId,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^https://[a-z0-9-]+-8000\.proxy\.runpod\.net$')]
    [string]$WorkerUrl,
    [string]$GatewayUrl = 'http://127.0.0.1:8787',
    [decimal]$MaxHourlyCostUsd = 0.50,
    [string]$LogRoot = (Join-Path $env:TEMP 'dichomnion-runpod-controller-static')
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$apiKeyFile = Join-Path $repoRoot 'hungkingface.txt'
$gatewayVarsFile = Join-Path $repoRoot 'apps\gateway\.dev.vars'

if (-not (Test-Path -LiteralPath $apiKeyFile) -or -not (Test-Path -LiteralPath $gatewayVarsFile)) {
    throw 'server_credentials_missing'
}

$apiLine = Get-Content -LiteralPath $apiKeyFile |
    Where-Object { $_ -match '^\s*api\s+runpod\s*:\s*rpa_[A-Za-z0-9]{32,256}\s*$' } |
    Select-Object -First 1
$runpodApiKey = [regex]::Match([string]$apiLine, 'rpa_[A-Za-z0-9]{32,256}').Value
if ([string]::IsNullOrWhiteSpace($runpodApiKey)) {
    throw 'runpod_api_key_invalid'
}

$adminLine = Get-Content -LiteralPath $gatewayVarsFile |
    Where-Object { $_ -match '^\s*WORKER_TARGET_ADMIN_TOKEN\s*=\s*"[^"]+"\s*$' } |
    Select-Object -First 1
$gatewayAdminToken = [regex]::Match(
    [string]$adminLine,
    '^\s*WORKER_TARGET_ADMIN_TOKEN\s*=\s*"([^"]+)"\s*$'
).Groups[1].Value
if ([string]::IsNullOrWhiteSpace($gatewayAdminToken)) {
    throw 'gateway_admin_token_missing'
}

$existing = @(Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'sdk_controller\.py.*--allow-resume' } |
    Select-Object -ExpandProperty ProcessId)
if ($existing.Count -gt 0) {
    throw 'controller_already_running'
}

$env:RUNPOD_API_KEY = $runpodApiKey
$env:GATEWAY_ADMIN_URL = $GatewayUrl
$env:GATEWAY_ADMIN_TOKEN = $gatewayAdminToken
$env:RUNPOD_POD_ID = $PodId
$env:RUNPOD_WORKER_URL = $WorkerUrl
$env:RUNPOD_WORKER_TRANSPORT = 'runpod_proxy'
$env:RUNPOD_WORKER_MAX_REQUEST_MS = '90000'
$env:RUNPOD_TARGET_TTL_SECONDS = '90'
$env:RUNPOD_HEARTBEAT_SECONDS = '20'
$env:RUNPOD_PROBE_INTERVAL_SECONDS = '5'
$env:RUNPOD_READY_TIMEOUT_SECONDS = '1800'
$env:RUNPOD_UNHEALTHY_GRACE_SECONDS = '120'
$env:RUNPOD_WORKER_PORT = '8000'
$env:RUNPOD_REPLACE_STOPPED_ON_CAPACITY = '0'

foreach ($proxyName in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE', 'SSL_CERT_DIR')) {
    Remove-Item "Env:$proxyName" -ErrorAction SilentlyContinue
}

$repoPython = Join-Path $repoRoot 'apps\gpu-worker\.runpod-controller-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $repoPython)) {
    throw 'controller_venv_missing'
}

New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$stdout = Join-Path $LogRoot 'stdout.log'
$stderr = Join-Path $LogRoot 'stderr.log'
$process = Start-Process -FilePath $repoPython `
    -ArgumentList @('-u', 'runpod\sdk_controller.py', '--allow-resume', '--max-hourly-cost-usd', ([string]$MaxHourlyCostUsd)) `
    -WorkingDirectory (Join-Path $repoRoot 'apps\gpu-worker') `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

[pscustomobject]@{
    controllerPid = $process.Id
    podId = $PodId
    gatewayUrl = $GatewayUrl
    stdout = $stdout
    stderr = $stderr
    maxHourlyUsd = ([string]$MaxHourlyCostUsd)
} | ConvertTo-Json -Compress
