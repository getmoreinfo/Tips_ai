# 컴퓨터 1 실행 (메인 노드 - 먼저 실행)
Write-Host "=========================================="
Write-Host "컴퓨터 1 (메인 노드) 실행"
Write-Host "=========================================="
Write-Host "현재 시간: $(Get-Date)"
Write-Host ""

py -3.11 -m accelerate.commands.launch 02_finetune_local.py
