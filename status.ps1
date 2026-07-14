# Check OmniVoice download and server status
$container = docker ps --filter "name=omnivoice" --format "{{.Status}}" 2>$null
if (-not $container) {
    Write-Host "Container is not running." -ForegroundColor Red
    Write-Host "Start with: docker compose up -d"
    exit 1
}

Write-Host "Container: $container" -ForegroundColor Green

$lines = docker logs omnivoice --tail 5 2>&1
$modelLine = $lines | Where-Object { $_ -match "Model loaded|Running|Error|Traceback|Loading" }
if ($modelLine) {
    Write-Host "Status: $modelLine" -ForegroundColor Yellow
}

$total = docker exec omnivoice python3 -c "
import os
d='/root/.cache/huggingface/hub/models--k2-fsa--OmniVoice/blobs'
total=0
for f in os.listdir(d):
    fp=os.path.join(d,f)
    if os.path.isfile(fp):
        total+=os.path.getsize(fp)
print(f'{total/1024/1024:.1f}')
" 2>$null

if ($total) {
    Write-Host "Model downloaded: $total MB" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Open browser: http://localhost:8001"
Write-Host "Check cache:  .\cache\huggingface\hub\models--k2-fsa--OmniVoice"
Write-Host "Output:       .\output"
Write-Host "Ref audio:    .\ref_audio"
Write-Host ""
Write-Host "View logs:    docker logs -f omnivoice"
Write-Host "Stop:         docker compose down"
