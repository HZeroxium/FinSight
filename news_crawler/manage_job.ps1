# News Crawler Service Management Script
# PowerShell version for cross-platform support

param(
  [Parameter(Mandatory = $false)]
  [ValidateSet("start", "stop", "status", "run", "config", "help")]
  [string]$Command = "help"
)

# Set up paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = $ScriptDir
Set-Location $ProjectDir

# Colors for output
$Colors = @{
  Green  = "Green"
  Red    = "Red"
  Yellow = "Yellow"
  Blue   = "Blue"
  Cyan   = "Cyan"
}

function Write-ColoredOutput {
  param(
    [string]$Message,
    [string]$Color = "White"
  )
  Write-Host $Message -ForegroundColor $Color
}

function Show-Help {
  Write-ColoredOutput "News Crawler Job Management Script" $Colors.Cyan
  Write-ColoredOutput "====================================" $Colors.Cyan
  Write-Host ""
  Write-ColoredOutput "Usage: .\manage_job.ps1 [command]" $Colors.Blue
  Write-Host ""
  Write-ColoredOutput "Commands:" $Colors.Yellow
  Write-Host "  start   - Start the news crawler cron job service"
  Write-Host "  stop    - Stop the news crawler cron job service"
  Write-Host "  status  - Check the status of the job service"
  Write-Host "  run     - Run a manual news crawl job"
  Write-Host "  config  - Display current configuration"
  Write-Host "  help    - Show this help message"
  Write-Host ""
  Write-ColoredOutput "Examples:" $Colors.Yellow
  Write-Host "  .\manage_job.ps1 start"
  Write-Host "  .\manage_job.ps1 status"
  Write-Host "  .\manage_job.ps1 run"
  Write-Host "  .\manage_job.ps1 stop"
}

function Initialize-Environment {
  # Check if virtual environment exists
  if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-ColoredOutput "Activating virtual environment..." $Colors.Blue
    & ".venv\Scripts\Activate.ps1"
  }
  elseif (Test-Path ".venv\bin\activate") {
    Write-ColoredOutput "Activating virtual environment..." $Colors.Blue
    & ".venv\bin\activate"
  }
  else {
    Write-ColoredOutput "No virtual environment found, using global Python..." $Colors.Yellow
  }

  # Set environment variables
  $env:PYTHONPATH = $ProjectDir
    
  # Check if .env file exists and suggest creating it
  if (-not (Test-Path ".env")) {
    Write-ColoredOutput "Warning: .env file not found. Copy .env.example to .env and configure it." $Colors.Yellow
  }
}

function Start-Service {
  Write-ColoredOutput "Starting News Crawler Job Service..." $Colors.Green
  python -m src.news_crawler_job start
}

function Stop-Service {
  Write-ColoredOutput "Stopping News Crawler Job Service..." $Colors.Red
  python -m src.news_crawler_job stop
}

function Get-ServiceStatus {
  Write-ColoredOutput "Checking News Crawler Job Service status..." $Colors.Blue
  python -m src.news_crawler_job status
}

function Start-ManualRun {
  Write-ColoredOutput "Running manual news crawl..." $Colors.Green
  python -m src.news_crawler_job run
}

function Show-Configuration {
  Write-ColoredOutput "Displaying current configuration..." $Colors.Blue
  python -m src.news_crawler_job config
}

# Main execution
try {
  Initialize-Environment
    
  switch ($Command) {
    "start" { Start-Service }
    "stop" { Stop-Service }
    "status" { Get-ServiceStatus }
    "run" { Start-ManualRun }
    "config" { Show-Configuration }
    "help" { Show-Help }
    default { Show-Help }
  }
}
catch {
  Write-ColoredOutput "Error: $_" $Colors.Red
  exit 1
}

Write-Host ""
Write-ColoredOutput "Operation completed." $Colors.Green
