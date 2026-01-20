# ========================================
# 실행 시간 설정 (여기만 수정하세요!)
# ========================================
$targetHour = 14   # 시간 (24시간 형식) - 오후 2시
$targetMinute = 0  # 분
$targetSecond = 0  # 초 (기본값 0)

# ========================================
# 실행 시간 계산 (오늘 날짜에 설정된 시간)
# ========================================
$now = Get-Date
$targetTime = Get-Date -Year $now.Year -Month $now.Month -Day $now.Day -Hour $targetHour -Minute $targetMinute -Second $targetSecond

# 만약 설정한 시간이 이미 지났다면 내일로 설정
if ($targetTime -lt $now) {
    $targetTime = $targetTime.AddDays(1)
}

$delay = ($targetTime - $now).TotalSeconds

if ($delay -gt 0) {
    Write-Host "=========================================="
    Write-Host "컴퓨터 3 - 대기 중: $([math]::Round($delay, 1))초 후 실행"
    Write-Host "목표 시간: $targetTime"
    Write-Host "현재 시간: $now"
    Write-Host "=========================================="
    Write-Host ""
    
    # 매 초 카운트다운
    while ($delay -gt 1) {
        $delay = ($targetTime - (Get-Date)).TotalSeconds
        if ($delay -gt 1) {
            Write-Host "남은 시간: $([math]::Round($delay, 1))초     " -NoNewline
            Write-Host "`r" -NoNewline
            Start-Sleep -Seconds 1
        }
    }
    
    # 마지막 1초 대기
    Start-Sleep -Seconds $delay
}

Write-Host ""
Write-Host "=========================================="
Write-Host "컴퓨터 3 실행 시작: $(Get-Date)"
Write-Host "=========================================="
Write-Host ""

# 실행 명령어
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
