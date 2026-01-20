# 방화벽 포트 29500 설정 (관리자 모드 PowerShell)

## 포트 29500 열기 (PyTorch Distributed Training)

### 명령어:

```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed Training" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

### 실행 결과 확인:

```powershell
Get-NetFirewallRule -DisplayName "PyTorch Distributed Training" | Select-Object DisplayName, Enabled, Direction, Action
```

---

## 간단 요약

**포함 (필요한 명령어):**
```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed Training" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

**확인:**
```powershell
Get-NetFirewallRule -DisplayName "PyTorch Distributed Training"
```

---

**이 명령어 하나면 충분합니다!** ✅
