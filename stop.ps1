# Stop OmniVoice container
docker compose -f "$PSScriptRoot\docker-compose.yml" down
Write-Host "Container stopped."
