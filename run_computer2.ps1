# 컴퓨터 2 실행 (약간 지연 후 실행)
Write-Host "=========================================="
Write-Host "컴퓨터 2 실행"
Write-Host "=========================================="
Write-Host "현재 시간: $(Get-Date)"
Write-Host "컴퓨터 1이 먼저 실행되기를 기다리는 중..."
Write-Host ""

# 컴퓨터 1이 먼저 실행되도록 2초 대기
Start-Sleep -Seconds 2

Write-Host "컴퓨터 2 실행 시작: $(Get-Date)"
Write-Host ""

py -3.11 -m accelerate.commands.launch 02_finetune_local.py
