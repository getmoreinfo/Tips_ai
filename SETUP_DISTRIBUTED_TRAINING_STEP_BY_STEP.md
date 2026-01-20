# 분산 학습 시작 단계별 가이드 (3대 컴퓨터)

## ✅ 준비 사항 확인

모든 컴퓨터에서:
- [x] Python 3.11.x 설치 완료
- [x] 패키지 설치 완료
- [x] `training_data_10000.csv` 파일 존재
- [x] 같은 네트워크에 연결됨

---

## 1단계: 각 컴퓨터의 IP 주소 확인

**모든 컴퓨터에서 실행:**

```powershell
ipconfig
```

**IPv4 주소 확인** (이더넷 또는 무선 LAN 어댑터에서)

예시:
- 컴퓨터 1 (메인): `192.168.1.100`
- 컴퓨터 2: `192.168.1.101`
- 컴퓨터 3: `192.168.1.102`

**각 컴퓨터의 IP 주소를 메모하세요!**

---

## 2단계: Accelerate 설정 (각 컴퓨터마다)

### 컴퓨터 1 (메인 노드, rank=0)에서:

```powershell
py -3.11 -m accelerate.config
```

**질문에 답변 (예시 IP 사용):**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine? (0-based)
> 0

What is the IP address of the machine that will host the main process?
> 192.168.1.100  (컴퓨터 1의 실제 IP)

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 192.168.1.101,192.168.1.102  (컴퓨터 2, 3의 IP - 쉼표로 구분)

What is the IP address of this machine?
> 192.168.1.100  (컴퓨터 1의 IP)

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision? (yes/no)
> yes

What mixed precision mode should be used? (no/fp16/bf16)
> fp16
```

---

### 컴퓨터 2 (rank=1)에서:

```powershell
py -3.11 -m accelerate.config
```

**질문에 답변 (대부분 동일, 일부만 다름):**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine? (0-based)
> 1  ⬅️ 중요! 노드 2는 1

What is the IP address of the machine that will host the main process?
> 192.168.1.100  (컴퓨터 1의 IP - 메인 노드)

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 192.168.1.101,192.168.1.102  (컴퓨터 2, 3의 IP)

What is the IP address of this machine?
> 192.168.1.101  ⬅️ 중요! 컴퓨터 2의 실제 IP

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision? (yes/no)
> yes

What mixed precision mode should be used? (no/fp16/bf16)
> fp16
```

---

### 컴퓨터 3 (rank=2)에서:

```powershell
py -3.11 -m accelerate.config
```

**질문에 답변:**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine? (0-based)
> 2  ⬅️ 중요! 노드 3은 2

What is the IP address of the machine that will host the main process?
> 192.168.1.100  (컴퓨터 1의 IP - 메인 노드)

What is the port you will use to communicate with the main process?
> 29500

What are the IP addresses of the machines connected to the main process?
> 192.168.1.101,192.168.1.102  (컴퓨터 2, 3의 IP)

What is the IP address of this machine?
> 192.168.1.102  ⬅️ 중요! 컴퓨터 3의 실제 IP

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision? (yes/no)
> yes

What mixed precision mode should be used? (no/fp16/bf16)
> fp16
```

---

## 3단계: 방화벽 설정 (중요!)

**모든 컴퓨터에서 Windows 방화벽 포트 열기:**

### PowerShell을 관리자 권한으로 실행:

```powershell
# 포트 29500 인바운드 규칙 추가
New-NetFirewallRule -DisplayName "PyTorch Distributed Training" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

**또는 GUI 방법:**
1. **Windows 보안** → **방화벽 및 네트워크 보호**
2. **고급 설정** 클릭
3. **인바운드 규칙** → **새 규칙**
4. **포트** 선택 → **다음**
5. **TCP** 선택, **특정 로컬 포트**: `29500` → **다음**
6. **연결 허용** → **다음**
7. **모든 프로필** 체크 → **다음**
8. 이름: "PyTorch Distributed Training" → **완료**

---

## 4단계: 네트워크 연결 테스트 (선택사항)

**컴퓨터 1에서:**

```powershell
# 컴퓨터 2로 연결 테스트 (예시)
Test-NetConnection -ComputerName 192.168.1.101 -Port 29500

# 컴퓨터 3으로 연결 테스트 (예시)
Test-NetConnection -ComputerName 192.168.1.102 -Port 29500
```

**연결 성공하면:**
```
TcpTestSucceeded : True
```

---

## 5단계: 데이터 파일 확인

**모든 컴퓨터에서:**

```powershell
# training_data_10000.csv 파일이 있는지 확인
dir training_data_10000.csv

# 파일 크기 확인
(Get-Item training_data_10000.csv).Length
```

**파일이 없으면:**
- Git에서 받기: `git pull origin main`
- 또는 컴퓨터 1에서 복사

---

## 6단계: 분산 학습 시작! 🚀

**중요: 3대 컴퓨터에서 거의 동시에 실행해야 합니다! (5초 이내 차이)**

### 컴퓨터 1, 2, 3 모두에서:

```powershell
# 프로젝트 폴더로 이동
cd C:\Users\<사용자명>\dev\ai-tr

# 분산 학습 시작
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

**실행 순서:**
1. **컴퓨터 1 먼저 실행** (메인 노드)
2. **5초 이내에 컴퓨터 2 실행**
3. **5초 이내에 컴퓨터 3 실행**

**또는 3개의 PowerShell 창을 미리 열어두고:**
- 창 1 (컴퓨터 1): 명령어 입력 후 대기
- 창 2 (컴퓨터 2): 명령어 입력 후 대기
- 창 3 (컴퓨터 3): 명령어 입력 후 대기
- **Enter를 거의 동시에 누르기**

---

## 7단계: 정상 작동 확인

### 각 컴퓨터에서 확인:

**정상 실행 시 출력:**
```
[INFO] 분산 학습 초기화 완료
[INFO] Rank 0/1/2 (각 컴퓨터마다 다름)
[INFO] GPU 사용 가능: True
[INFO] 데이터 로드 중...
[INFO] 학습 시작...
```

**GPU 사용률 확인:**
```powershell
# 다른 PowerShell 창에서
nvidia-smi
```

**정상 작동 시:**
- GPU 사용률: 50-100%
- GPU 메모리 사용 중
- 프로세스 이름: python.exe

---

## 문제 해결

### 연결 오류 발생 시:

1. **IP 주소 확인:**
   ```powershell
   ipconfig
   ```

2. **방화벽 확인:**
   ```powershell
   Get-NetFirewallRule -DisplayName "PyTorch*"
   ```

3. **포트 사용 중 확인:**
   ```powershell
   netstat -an | findstr 29500
   ```

4. **모든 컴퓨터가 같은 네트워크에 있는지 확인**

### 동기화 오류 발생 시:

1. **모든 컴퓨터에서 같은 데이터 파일 확인:**
   ```powershell
   Get-FileHash training_data_10000.csv
   ```
   모든 컴퓨터에서 해시 값이 같아야 함

2. **패키지 버전 확인:**
   ```powershell
   py -3.11 -m pip list | findstr "torch transformers accelerate"
   ```
   모든 컴퓨터에서 버전이 같아야 함

### GPU 오류 발생 시:

1. **GPU 드라이버 확인:**
   ```powershell
   nvidia-smi
   ```

2. **CUDA 버전 확인:**
   ```powershell
   py -3.11 -c "import torch; print(torch.version.cuda)"
   ```

---

## 빠른 실행 체크리스트

### 시작 전:

- [ ] 모든 컴퓨터의 IP 주소 확인 및 메모
- [ ] 컴퓨터 1 (메인): Accelerate 설정 완료 (rank=0)
- [ ] 컴퓨터 2: Accelerate 설정 완료 (rank=1)
- [ ] 컴퓨터 3: Accelerate 설정 완료 (rank=2)
- [ ] 모든 컴퓨터에서 방화벽 설정 완료 (포트 29500)
- [ ] 모든 컴퓨터에서 `training_data_10000.csv` 파일 확인
- [ ] 모든 컴퓨터가 같은 네트워크에 연결됨

### 실행:

- [ ] 컴퓨터 1에서 명령 실행 (먼저)
- [ ] 컴퓨터 2에서 명령 실행 (5초 이내)
- [ ] 컴퓨터 3에서 명령 실행 (5초 이내)
- [ ] 정상 작동 확인

---

## 요약 명령어

### 모든 컴퓨터에서:

```powershell
# 1. IP 확인
ipconfig

# 2. Accelerate 설정
py -3.11 -m accelerate.config

# 3. 방화벽 설정 (관리자 권한)
New-NetFirewallRule -DisplayName "PyTorch Distributed Training" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow

# 4. 데이터 확인
dir training_data_10000.csv

# 5. 분산 학습 시작 (3대 동시에)
cd C:\Users\<사용자명>\dev\ai-tr
py -3.11 -m accelerate.commands.launch 02_finetune_local.py
```

---

**이제 준비 완료! 3대 컴퓨터에서 분산 학습을 시작하세요!** 🚀
