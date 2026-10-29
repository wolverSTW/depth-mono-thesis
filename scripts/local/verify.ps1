param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ResearchArgs)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location -LiteralPath $projectDir
try {
    & python scripts/research.py verify @ResearchArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
