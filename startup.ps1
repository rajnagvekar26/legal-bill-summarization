<#
.SYNOPSIS
    Automated startup script for the Cross-Architecture Legal Bill Summarization platform & Knowledge Graph.

.DESCRIPTION
    Initializes the virtual environment, verifies dependencies, inspects the Graphify knowledge graph,
    and launches the FastAPI backend (port 8000), Streamlit dashboard (port 8501), and optional
    interactive Knowledge Graph visualizer (port 8088) with health probes, environment alignment, and clean shutdown.

.PARAMETER OnlyAPI
    Launch only the FastAPI backend service on port 8000.

.PARAMETER OnlyDashboard
    Launch only the Streamlit dashboard on port 8501.

.PARAMETER OnlyGraph
    Launch only the Graphify Knowledge Graph visualizer on port 8088.

.PARAMETER WithGraph
    Launch the interactive Knowledge Graph visualizer alongside the API and Dashboard. (Alias: -Graph)

.PARAMETER UpdateGraph
    Run an incremental AST update on the Graphify knowledge graph prior to service launch.

.PARAMETER RunTests
    Execute the pytest test suite before starting platform services.

.PARAMETER ApiPort
    Port number for the FastAPI backend (default: 8000).

.PARAMETER DashboardPort
    Port number for the Streamlit dashboard (default: 8501).

.PARAMETER RemoteApiUrl
    Base URL for a remote inference API, such as the current Colab LocalTunnel URL.
    When provided, the local FastAPI process is not started.

.PARAMETER GraphPort
    Port number for the Knowledge Graph HTTP server (default: 8088).

.PARAMETER NoBrowser
    Do not automatically open web browser windows.
#>

[CmdletBinding()]
param(
    [switch]$OnlyAPI,
    [switch]$OnlyDashboard,
    [switch]$OnlyGraph,
    [Alias("Graph")]
    [switch]$WithGraph,
    [switch]$UpdateGraph,
    [switch]$RunTests,
    [int]$ApiPort = 8000,
    [int]$DashboardPort = 8501,
    [string]$RemoteApiUrl = "",
    [int]$GraphPort = 8088,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
Set-Location $ScriptDir

# Enforce project root on PYTHONPATH and enable unbuffered console output
$env:PYTHONPATH = $ScriptDir
$env:PYTHONUNBUFFERED = "1"

if (-not [string]::IsNullOrWhiteSpace($RemoteApiUrl)) {
    if ($OnlyAPI) {
        throw "-OnlyAPI cannot be used with -RemoteApiUrl because the remote API is not hosted locally."
    }
    if ($RemoteApiUrl -notmatch '^https?://') {
        throw "-RemoteApiUrl must be an absolute http:// or https:// base URL."
    }

    $RemoteApiUrl = $RemoteApiUrl.TrimEnd("/")
    $env:SUMMARIZATION_API_URL = $RemoteApiUrl
}

# Styling helper functions
function Write-Header {
    param([string]$Text)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor White
    Write-Host "========================================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host "[OK] $Text" -ForegroundColor Green
}

function Write-Info {
    param([string]$Text)
    Write-Host "[INFO] $Text" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Text)
    Write-Host "[ERROR] $Text" -ForegroundColor Red
}

Write-Header "Cross-Architecture Legal Bill Summarization Platform"

# Knowledge Graph context inspection from graphify-out
$GraphDir = Join-Path $ScriptDir "graphify-out"
$GraphReportPath = Join-Path $GraphDir "GRAPH_REPORT.md"
$GraphHtmlPath = Join-Path $GraphDir "graph.html"
$GraphStats = $null

if (Test-Path $GraphReportPath) {
    try {
        $reportContent = [System.IO.File]::ReadAllText($GraphReportPath, [System.Text.Encoding]::UTF8)
        if ($reportContent -match "(\d+)\s+nodes.*?(\d+)\s+edges.*?(\d+)\s+communities") {
            $GraphStats = "$($Matches[1]) nodes | $($Matches[2]) edges | $($Matches[3]) communities"
        }
    } catch {}
}

if ($GraphStats) {
    Write-Host "  Architecture Graph: $GraphStats (graphify-out/)" -ForegroundColor DarkGray
    Write-Host "  Supported Models:   BART Base Dense | LED Sparse 4K | Qwen2.5-1.5B Causal" -ForegroundColor DarkGray
    Write-Host ""
}

# 1. Locate Virtual Environment
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$VenvUvicorn = Join-Path $ScriptDir ".venv\Scripts\uvicorn.exe"
$VenvStreamlit = Join-Path $ScriptDir ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Info "Virtual environment not detected. Creating .venv..."
    python -m venv .venv
    Write-Success "Created virtual environment at .venv"
    
    Write-Info "Upgrading pip and installing requirements..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    Write-Success "Installed dependencies from requirements.txt"
} else {
    Write-Success "Found existing virtual environment at .venv"
}

# Verify critical executables
if (-not (Test-Path $VenvUvicorn)) {
    $VenvUvicorn = "$VenvPython -m uvicorn"
}
if (-not (Test-Path $VenvStreamlit)) {
    $VenvStreamlit = "$VenvPython -m streamlit"
}

# Graphify Updater helper
function Invoke-GraphifyUpdate {
    Write-Info "Running incremental AST update on knowledge graph..."
    $graphifyPyPath = Join-Path $ScriptDir "graphify-out\.graphify_python"
    $graphifyExe = Get-Command graphify -ErrorAction SilentlyContinue
    
    if ($graphifyExe) {
        & graphify update .
    } elseif (Test-Path $graphifyPyPath) {
        $graphifyPy = (Get-Content $graphifyPyPath -Raw).Trim()
        if (Test-Path $graphifyPy) {
            $graphifyDir = Split-Path -Parent $graphifyPy
            $graphifyBin = Join-Path $graphifyDir "graphify.exe"
            if (Test-Path $graphifyBin) {
                & $graphifyBin update .
            } else {
                & $graphifyPy -m graphify update .
            }
        }
    } else {
        Write-Err "Graphify CLI not found. Skipping graph update."
    }
}

# Run Graphify update if requested
if ($UpdateGraph) {
    Invoke-GraphifyUpdate
}

# Run Verification Tests if requested
if ($RunTests) {
    Write-Info "Running pytest test suite across models, guardrails, and API..."
    & $VenvPython -m pytest -v tests/
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Test suite failed with exit code $LASTEXITCODE. Aborting startup."
        exit $LASTEXITCODE
    }
    Write-Success "All tests passed successfully!"
}

$ProcessesToClean = @()

# Cleanup on exit
$cleanupBlock = {
    Write-Host "`nShutting down legal summarizer services..." -ForegroundColor Yellow
    foreach ($proc in $ProcessesToClean) {
        if ($proc -and -not $proc.HasExited) {
            try {
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                Write-Host "Stopped process ID $($proc.Id)" -ForegroundColor DarkGray
            } catch {}
        }
    }
    Write-Host "Shutdown complete." -ForegroundColor Green
}

try {
    # 2. Launch Knowledge Graph Viewer (if requested)
    $graphProcess = $null
    if ($OnlyGraph -or $WithGraph) {
        if (Test-Path $GraphHtmlPath) {
            Write-Info "Starting Knowledge Graph viewer on http://localhost:$GraphPort..."
            $graphArgs = "-m http.server $GraphPort --directory graphify-out"
            $graphProcess = Start-Process -FilePath $VenvPython -ArgumentList $graphArgs -PassThru -NoNewWindow
            $ProcessesToClean += $graphProcess
            Start-Sleep -Milliseconds 600
            Write-Success "Knowledge Graph is LIVE at http://localhost:$GraphPort/graph.html"
            
            if (-not $NoBrowser) {
                Start-Process "http://localhost:$GraphPort/graph.html"
            }
        } else {
            Write-Err "Knowledge graph not found at $GraphHtmlPath. Run 'graphify update .' or /graphify first."
        }
    }

    if ($OnlyGraph) {
        Write-Header "Knowledge Graph Running in Standalone Mode"
        Write-Host "  * Interactive Graph: http://localhost:$GraphPort/graph.html" -ForegroundColor Cyan
        Write-Host "  * Architecture Audit: graphify-out\GRAPH_REPORT.md" -ForegroundColor Cyan
        Write-Host "`nPress Ctrl+C to terminate.`n" -ForegroundColor White
        
        while ($graphProcess -and -not $graphProcess.HasExited) {
            Start-Sleep -Seconds 1
        }
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($RemoteApiUrl)) {
        Write-Info "Checking remote inference API at $RemoteApiUrl..."
        try {
            $remoteHealth = Invoke-RestMethod `
                -Uri "$RemoteApiUrl/api/health" `
                -Method Get `
                -Headers @{"bypass-tunnel-reminder" = "true"} `
                -TimeoutSec 20
            if (-not $remoteHealth -or $remoteHealth.status -ne "healthy") {
                throw "Remote health endpoint did not report healthy status."
            }
            $remoteDevice = if ($remoteHealth.device) { $remoteHealth.device } else { "unknown device" }
            Write-Success "Remote inference API is healthy (Device: $remoteDevice)"
        } catch {
            throw "Could not verify the remote inference API at $RemoteApiUrl. Confirm the Colab server and tunnel are running. $($_.Exception.Message)"
        }
    }

    # 3. Launch FastAPI Backend
    $apiProcess = $null
    if (-not $OnlyDashboard -and [string]::IsNullOrWhiteSpace($RemoteApiUrl)) {
        Write-Info "Starting FastAPI backend on http://localhost:$ApiPort..."
        
        $apiArgs = "src.api.app:app --host 127.0.0.1 --port $ApiPort --reload"
        $apiProcess = Start-Process -FilePath $VenvPython -ArgumentList "-m uvicorn $apiArgs" -PassThru -NoNewWindow
        $ProcessesToClean += $apiProcess

        # Health probe loop
        $healthy = $false
        $attempts = 0
        $healthInfo = $null
        Write-Info "Waiting for FastAPI health endpoint to respond..."
        while (-not $healthy -and $attempts -lt 15) {
            Start-Sleep -Seconds 1
            $attempts++
            try {
                $response = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
                if ($response -and $response.status -eq "healthy") {
                    $healthy = $true
                    $healthInfo = $response
                }
            } catch {}
        }

        if ($healthy) {
            $deviceInfo = if ($healthInfo.device) { $healthInfo.device } else { "CPU" }
            $modelsCount = if ($healthInfo.supported_models) { ($healthInfo.supported_models).Count } else { 3 }
            Write-Success "FastAPI backend is LIVE at http://localhost:$ApiPort (Device: $deviceInfo, Models: $modelsCount)"
        } else {
            Write-Err "FastAPI health check timed out. Continuing startup..."
        }
    }

    # 4. Launch Streamlit Dashboard
    $dashProcess = $null
    if (-not $OnlyAPI) {
        Write-Info "Starting Streamlit dashboard on http://localhost:$DashboardPort..."
        
        $browserFlag = if ($NoBrowser) { "--server.headless true" } else { "--server.headless false" }
        $dashArgs = "run src/dashboard/app.py --server.port $DashboardPort $browserFlag"
        
        if ($OnlyDashboard) {
            # Run in foreground if only dashboard requested
            & $VenvPython -m streamlit run src/dashboard/app.py --server.port $DashboardPort
        } else {
            # Run in background and open browser
            $dashProcess = Start-Process -FilePath $VenvPython -ArgumentList "-m streamlit $dashArgs" -PassThru -NoNewWindow
            $ProcessesToClean += $dashProcess
            
            Write-Success "Streamlit dashboard is LIVE at http://localhost:$DashboardPort"
            
            if (-not $NoBrowser) {
                Start-Sleep -Seconds 2
                Start-Process "http://localhost:$DashboardPort"
            }
            
            Write-Header "Services Running Successfully"
            if ([string]::IsNullOrWhiteSpace($RemoteApiUrl)) {
                Write-Host "  * FastAPI API Docs:    http://localhost:$ApiPort/docs" -ForegroundColor Cyan
                Write-Host "  * FastAPI Health:      http://localhost:$ApiPort/api/health" -ForegroundColor Cyan
            } else {
                Write-Host "  * Remote API Docs:     $RemoteApiUrl/docs" -ForegroundColor Cyan
                Write-Host "  * Remote API Health:   $RemoteApiUrl/api/health" -ForegroundColor Cyan
            }
            Write-Host "  * Streamlit Dashboard: http://localhost:$DashboardPort" -ForegroundColor Cyan
            if ($WithGraph) {
                Write-Host "  * Knowledge Graph:     http://localhost:$GraphPort/graph.html" -ForegroundColor Cyan
            } elseif (Test-Path $GraphHtmlPath) {
                Write-Host "  * Knowledge Graph:     file://$($GraphHtmlPath.Replace('\', '/')) (Tip: Use -WithGraph for HTTP server)" -ForegroundColor DarkCyan
            }
            if (Test-Path $GraphReportPath) {
                Write-Host "  * Graph Audit Report:  graphify-out\GRAPH_REPORT.md" -ForegroundColor DarkGray
            }
            Write-Host "`n[Tips] Query Graph: graphify query `"<question>`" | Update Graph: .\startup.ps1 -UpdateGraph" -ForegroundColor DarkGray
            Write-Host "`nPress Ctrl+C to terminate all services.`n" -ForegroundColor White
            
            # Keep script alive to handle Ctrl+C and crash detection
            while ($true) {
                if ($apiProcess -and $apiProcess.HasExited) {
                    Write-Err "FastAPI backend exited unexpectedly (Exit code: $($apiProcess.ExitCode))."
                    break
                }
                if ($dashProcess -and $dashProcess.HasExited) {
                    Write-Err "Streamlit dashboard exited unexpectedly (Exit code: $($dashProcess.ExitCode))."
                    break
                }
                if ($graphProcess -and $graphProcess.HasExited) {
                    Write-Err "Knowledge Graph server exited unexpectedly (Exit code: $($graphProcess.ExitCode))."
                    break
                }
                Start-Sleep -Seconds 1
            }
        }
    } elseif ($OnlyAPI) {
        Write-Header "FastAPI Running in Standalone Mode"
        Write-Host "  * API Docs:   http://localhost:$ApiPort/docs" -ForegroundColor Cyan
        Write-Host "  * API Health: http://localhost:$ApiPort/api/health" -ForegroundColor Cyan
        Write-Host "`nPress Ctrl+C to terminate.`n" -ForegroundColor White
        
        while ($apiProcess -and -not $apiProcess.HasExited) {
            Start-Sleep -Seconds 1
        }
    }
}
finally {
    & $cleanupBlock
}
