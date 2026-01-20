# 분산 학습 빠른 시작 가이드

## ✅ IP 주소 정보

- **컴퓨터 1 (메인)**: `210.93.16.37` (rank=0)
- **컴퓨터 2**: `210.93.16.36` (rank=1)
- **컴퓨터 3**: `210.93.16.35` (rank=2)
- **포트**: `29500`

---

## 🚀 빠른 설정 (각 컴퓨터에서)

### 컴퓨터 1 (메인)에서:

```powershell
cd C:\Users\<사용자명>\dev\ai-tr
.\setup_accelerate_computer1.ps1
```

### 컴퓨터 2에서:

```powershell
cd C:\Users\<사용자명>\dev\ai-tr
.\setup_accelerate_computer2.ps1
```

### 컴퓨터 3에서:

```powershell
cd C:\Users\comso-1407\dev\ai-tr
.\setup_accelerate_computer3.ps1
```

---

## 🔥 방화벽 설정 (모든 컴퓨터, 관리자 권한)

PowerShell을 **관리자 권한**으로 실행:

```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed Training" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

---

## 📦 데이터 파일 확인 (모든 컴퓨터)

```powershell
dir training_data_10000.csv
```

**파일이 없으면:**
```powershell
git pull origin main
```

---

## 🎯 분산 학습 시작 (3대 동시에!)

**중요: 3대 컴퓨터에서 거의 동시에 실행 (5초 이내 차이)**

### 모든 컴퓨터에서:

```powershell
cd C:\Users\<사용자명>\dev\ai-tr

# 환경 변수 설정 (선택사항)
$env:MASTER_ADDR='210.93.16.37'
$env:MASTER_PORT='29500'

# 분산 학습 시작
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

**실행 순서:**
1. **컴퓨터 1 먼저 실행** (메인 노드)
2. **5초 이내에 컴퓨터 2 실행**
3. **5초 이내에 컴퓨터 3 실행**

---

## ✅ 실행 확인

**정상 실행 시:**
- 각 컴퓨터에서 "분산 학습 초기화 완료" 메시지
- GPU 사용률 증가 (`nvidia-smi` 확인)
- 로그 파일 생성

**GPU 확인:**
```powershell
nvidia-smi
```

---

## 🆘 문제 해결

### 연결 오류:
1. IP 주소 확인: `ipconfig`
2. 방화벽 확인
3. 같은 네트워크 확인

### 동기화 오류:
1. 모든 컴퓨터에서 같은 `training_data_10000.csv` 확인
2. 패키지 버전 확인

---

## 📝 체크리스트

### 시작 전:
- [ ] 모든 컴퓨터에서 Accelerate 설정 완료
- [ ] 모든 컴퓨터에서 방화벽 설정 완료
- [ ] 모든 컴퓨터에서 `training_data_10000.csv` 확인
- [ ] 모든 컴퓨터가 같은 네트워크에 연결됨

### 실행:
- [ ] 컴퓨터 1에서 명령 실행 (먼저)
- [ ] 컴퓨터 2에서 명령 실행 (5초 이내)
- [ ] 컴퓨터 3에서 명령 실행 (5초 이내)
- [ ] 정상 작동 확인

---

**이제 준비 완료! 분산 학습을 시작하세요!** 🚀
