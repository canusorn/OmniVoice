# OmniVoice Docker entrypoint for Windows (PowerShell)
# Creates required directories before starting Docker

$BasePath = Split-Path -Parent $MyInvocation.MyCommand.Path

$Dirs = @(
    "$BasePath\cache\huggingface",
    "$BasePath\cache\torch",
    "$BasePath\output",
    "$BasePath\ref_audio"
)

foreach ($Dir in $Dirs) {
    if (-not (Test-Path -LiteralPath $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
        Write-Host "Created: $Dir"
    }
}

Write-Host "Starting OmniVoice container..."
docker compose -f "$BasePath\docker-compose.yml" up --build -d

if ($?) {
    Write-Host "OmniVoice is running at http://localhost:8001"
    Write-Host ""
    Write-Host "Output files:  $BasePath\output"
    Write-Host "Ref audio:     $BasePath\ref_audio"
    Write-Host ""
    Write-Host "View logs: docker compose logs -f"
    Write-Host "Stop:      docker compose down"
}
