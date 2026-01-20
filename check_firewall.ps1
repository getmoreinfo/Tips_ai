# 방화벽 설정 확인 스크립트
# 사용법: PowerShell에서 .\check_firewall.ps1 실행

Write-Host "=========================================="
Write-Host "방화벽 설정 확인"
Write-Host "=========================================="
Write-Host ""

$port = "29500"

Write-Host "포트 29500 (PyTorch Distributed Training) 방화벽 규칙 확인:"
Write-Host ""

# 포트 29500 관련 규칙 확인
$rules = Get-NetFirewallPortFilter | Where-Object {$_.LocalPort -eq $port} | Get-NetFirewallRule

if ($rules.Count -eq 0) {
    Write-Host "[WARNING] 포트 $port 에 대한 방화벽 규칙이 없습니다!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "방화벽 규칙을 생성해야 합니다." -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "방화벽 규칙 생성 방법:"
    Write-Host "1. Windows 보안 > 방화벽 및 네트워크 보호"
    Write-Host "2. 고급 설정"
    Write-Host "3. 인바운드 규칙 > 새 규칙"
    Write-Host "4. 포트 선택 > TCP > 특정 로컬 포트: $port"
    Write-Host "5. 연결 허용 > 모든 프로필"
    Write-Host "6. 이름: 'PyTorch Distributed Training'"
    Write-Host ""
    
    Write-Host "또는 PowerShell에서 자동 생성:"
    Write-Host ""
    Write-Host "New-NetFirewallRule -DisplayName 'PyTorch Distributed Training' -Direction Inbound -LocalPort $port -Protocol TCP -Action Allow" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "[OK] 포트 $port 에 대한 방화벽 규칙이 있습니다:" -ForegroundColor Green
    Write-Host ""
    
    foreach ($rule in $rules) {
        Write-Host "규칙 이름: $($rule.DisplayName)"
        Write-Host "  활성화: $($rule.Enabled)"
        Write-Host "  방향: $($rule.Direction)"
        Write-Host "  동작: $($rule.Action)"
        Write-Host ""
        
        if ($rule.Enabled -eq $false) {
            Write-Host "[WARNING] 규칙이 비활성화되어 있습니다!" -ForegroundColor Yellow
            Write-Host "활성화하려면 다음 명령 실행:" -ForegroundColor Yellow
            Write-Host "Enable-NetFirewallRule -Name '$($rule.Name)'" -ForegroundColor Cyan
            Write-Host ""
        }
        
        if ($rule.Action -eq "Block") {
            Write-Host "[ERROR] 규칙이 차단(Block)으로 설정되어 있습니다!" -ForegroundColor Red
            Write-Host "허용(Allow)으로 변경해야 합니다!" -ForegroundColor Red
            Write-Host ""
        }
    }
}

Write-Host "=========================================="
Write-Host "전체 방화벽 상태 확인:"
Write-Host "=========================================="
Write-Host ""

$profileStatus = Get-NetFirewallProfile | Select-Object Name, Enabled

foreach ($profile in $profileStatus) {
    $status = if ($profile.Enabled) { "활성화됨" } else { "비활성화됨" }
    Write-Host "$($profile.Name) 프로필: $status"
}

Write-Host ""
