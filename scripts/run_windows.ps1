param(
    [ValidateSet("smoke", "inspect", "extract", "research", "gan", "adjusted")]
    [string]$Stage = "smoke",
    [string]$DataRoot = "D:\EEG_Dataset\HBN_100Hz",
    [string]$Manifest = "local\prepared\manifest.csv",
    [string]$Regions = "local\regions.json",
    [string]$Features = "local\features.csv"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$PythonExe = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { throw "Create .venv and install the project first; see README.md." }
function Invoke-Project {
    & $PythonExe -m hbn_r3.cli @args
    if ($LASTEXITCODE -ne 0) { throw "Project command failed (exit $LASTEXITCODE)." }
}
$RunId = Get-Date -Format "yyyyMMdd_HHmmss_fff"
switch ($Stage) {
    "smoke" {
        Invoke-Project demo --output "demo_data\features.csv" --n-per-release 32
        Invoke-Project run --features "demo_data\features.csv" --config "configs\smoke.json" --output "outputs\smoke_$RunId"
        Invoke-Project plot --results "outputs\smoke_$RunId"
    }
    "inspect" { Invoke-Project inspect-data --root $DataRoot --output "local\inventory.csv" }
    "extract" { Invoke-Project extract --manifest $Manifest --regions $Regions --output $Features }
    "research" { Invoke-Project run --features $Features --config "configs\research.json" --output "outputs\research_$RunId" }
    "gan" { Invoke-Project run --features $Features --config "configs\gan_research.json" --output "outputs\gan_$RunId" }
    "adjusted" { Invoke-Project run --features $Features --config "configs\research_adjusted.json" --output "outputs\adjusted_$RunId" }
}
