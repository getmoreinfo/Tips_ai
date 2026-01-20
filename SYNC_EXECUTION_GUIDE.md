# 3대 컴퓨터 동시 실행 가이드

## ⚠️ 중요: Cursor AI는 자동 실행 불가능

Cursor AI는 **정확한 시간에 자동으로 명령을 실행하는 기능이 없습니다.**
사용자가 직접 실행해야 합니다.

---

## 방법 1: 수동 동시 실행 (추천) ⭐

### 준비 단계:

#### 각 컴퓨터에서 미리 명령어 준비:

**컴퓨터 1 (메인):**
```powershell
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

**컴퓨터 2:**
```powershell
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

**컴퓨터 3:**
```powershell
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

### 실행 방법:

1. **1시47분 정각 전에 각 컴퓨터의 PowerShell을 준비**
2. **명령어를 입력하되 Enter는 누르지 않음**
3. **1시47분 정각에 Enter 키를 동시에 누름**

---

## 방법 2: 지연 실행 스크립트 (더 정확)

### 각 컴퓨터에서 실행할 스크립트:

**예: 1시47분에 실행하려면**

```powershell
# 1시47분까지 대기한 후 실행
$targetTime = Get-Date "2026-01-20 13:47:00"
$now = Get-Date
$delay = ($targetTime - $now).TotalSeconds

if ($delay -gt 0) {
    Write-Host "대기 중: $([math]::Round($delay, 1))초 후 실행"
    Start-Sleep -Seconds $delay
}

Write-Host "실행 시작!"
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

### 더 간단한 버전:

```powershell
# 현재 시간 기준으로 10초 후 실행 예시
Start-Sleep -Seconds 10
Write-Host "실행 시작!"
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

---

## 방법 3: 시간 계산 스크립트

### 각 컴퓨터에서 실행:

```powershell
# 실행 파일 생성
$script = @"
`$targetTime = Get-Date "2026-01-20 13:47:00"
`$now = Get-Date
`$delay = (`$targetTime - `$now).TotalSeconds

if (`$delay -gt 0) {
    Write-Host "대기 중: `$([math]::Round(`$delay, 1))초 후 실행"
    Start-Sleep -Seconds `$delay
}

Write-Host "실행 시작: `$(Get-Date)"
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
"@

$script | Out-File -FilePath "run_at_1347.ps1" -Encoding UTF8

# 실행
.\run_at_1347.ps1
```

---

## 방법 4: 가장 간단한 방법

### 각 컴퓨터에서:

1. **PowerShell에서 명령어 입력**
2. **Enter는 누르지 않음**
3. **다른 컴퓨터와 시간 확인**
4. **"1, 2, 3, 시작!" 신호로 동시에 Enter**

---

## ⚠️ 주의사항

### 1. 시간 동기화 확인

모든 컴퓨터의 시계가 정확한지 확인:

```powershell
Get-Date
```

시간이 다르면 동기화:

```powershell
# 시간 동기화 (관리자 권한 필요)
w32tm /resync
```

### 2. 실행 순서

**컴퓨터 1 (메인)이 먼저 실행되어야 합니다!**

순서:
1. **컴퓨터 1** 먼저 실행
2. **컴퓨터 2, 3** 동시에 실행 (1초 이내)

### 3. 실행 타이밍

- 컴퓨터 1: 1시47분 00초
- 컴퓨터 2, 3: 1시47분 01초 (또는 컴퓨터 1 실행 직후)

---

## 추천 방법

### 가장 확실한 방법:

1. **컴퓨터 1에서 먼저 실행:**
   ```powershell
   py -3.11 -m accelerate.commands.launch 02_finetune_local.py
   ```

2. **컴퓨터 1이 "학습 시작" 메시지를 보이면, 컴퓨터 2, 3에서 동시에 실행**

3. **또는 1-2초 간격으로 실행:**
   - 컴퓨터 1: 1시47분 00초
   - 컴퓨터 2: 1시47분 01초
   - 컴퓨터 3: 1시47분 01초

---

## 자동 실행 스크립트 생성

### 각 컴퓨터에 저장할 스크립트:

**run_at_1347.ps1:**

```powershell
# 1시47분 00초에 실행
$targetTime = Get-Date "2026-01-20 13:47:00"
$now = Get-Date
$delay = ($targetTime - $now).TotalSeconds

if ($delay -gt 0) {
    Write-Host "=========================================="
    Write-Host "대기 중: $([math]::Round($delay, 1))초 후 실행"
    Write-Host "목표 시간: $targetTime"
    Write-Host "현재 시간: $now"
    Write-Host "=========================================="
    
    # 매 초 카운트다운
    while ($delay -gt 1) {
        $delay = ($targetTime - (Get-Date)).TotalSeconds
        if ($delay -gt 1) {
            Write-Host "남은 시간: $([math]::Round($delay, 1))초" -NoNewline
            Write-Host "`r" -NoNewline
            Start-Sleep -Seconds 1
        }
    }
    
    Start-Sleep -Seconds $delay
}

Write-Host ""
Write-Host "=========================================="
Write-Host "실행 시작: $(Get-Date)"
Write-Host "=========================================="
Write-Host ""

py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

### 사용 방법:

1. 각 컴퓨터에 스크립트 저장
2. 1시47분 전에 실행:
   ```powershell
   .\run_at_1347.ps1
   ```
3. 자동으로 정확한 시간에 실행됨

---

## 요약

**Cursor AI에게 자동 실행을 요청하는 것은 불가능합니다.**

**대신:**
1. ✅ 스크립트를 사용한 지연 실행 (가장 정확)
2. ✅ 수동으로 동시에 Enter 키 누르기
3. ✅ 컴퓨터 1 먼저, 이후 2, 3 동시 실행

**추천: 스크립트를 사용한 지연 실행이 가장 정확합니다!**
