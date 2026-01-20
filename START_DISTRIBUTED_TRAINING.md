# 분산 학습 시작 가이드 (3대 컴퓨터)

## ✅ 준비 완료!

3대의 컴퓨터에 모두 Git 클론과 환경 설정이 완료되었습니다.

---

## 다음 단계: 분산 학습 시작

### 1단계: 각 컴퓨터의 IP 주소 확인

**각 컴퓨터에서 PowerShell 실행:**

```powershell
ipconfig
```

**IPv4 주소 확인** (예: 192.168.1.100)

각 컴퓨터의 IP 주소를 메모하세요:
- 컴퓨터 1 (메인): `________________`
- 컴퓨터 2: `________________`
- 컴퓨터 3: `________________`

---

### 2단계: Accelerate 설정 (가장 간단한 방법) ⭐

#### 컴퓨터 1 (메인)에서:

```bash
accelerate config
```

**질문에 답변:**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> multi-GPU

How many different machines will you use?
> 3

What is the rank of this machine?
> 0

What is the IP address of the machine that will host the main process?
> 192.168.1.100  (컴퓨터 1의 IP)

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

Do you want to use Mixed Precision?
> fp16
```

#### 컴퓨터 2에서:

```bash
accelerate config
```

**질문에 답변 (대부분 동일, 일부만 다름):**

```
What is the rank of this machine?
> 1  (노드 2는 1)

What is the IP address of this machine?
> 192.168.1.101  (컴퓨터 2의 IP)
```

#### 컴퓨터 3에서:

```bash
accelerate config
```

**질문에 답변:**

```
What is the rank of this machine?
> 2  (노드 3은 2)

What is the IP address of this machine?
> 192.168.1.102  (컴퓨터 3의 IP)
```

---

### 3단계: 방화벽 설정 (중요!)

**모든 컴퓨터에서 Windows 방화벽 설정:**

#### 포트 29500 열기:

1. **Windows 보안** → **방화벽 및 네트워크 보호**
2. **고급 설정** 클릭
3. **인바운드 규칙** → **새 규칙**
4. **포트** 선택 → **다음**
5. **TCP** 선택, **특정 로컬 포트**: `29500` → **다음**
6. **연결 허용** → **다음**
7. **모든 프로필** 체크 → **다음**
8. 이름: "PyTorch Distributed Training" → **완료**

---

### 4단계: 데이터 확인

**모든 컴퓨터에서 확인:**

```bash
# training_data_10000.csv 파일이 있는지 확인
dir training_data_10000.csv

# 또는
ls training_data_10000.csv
```

**파일이 없으면:** 데이터베이스에 접속 가능한 한 컴퓨터에서 실행:

```bash
python 01_export_sample_10000.py
```

그 후 생성된 `training_data_10000.csv`를 다른 컴퓨터로 복사합니다.

**또는 Git에 푸시 후 다른 컴퓨터에서 pull:**

```bash
# 컴퓨터 1에서
git add training_data_10000.csv
git commit -m "Add training data"
git push origin main

# 컴퓨터 2, 3에서
git pull origin main
```

---

### 5단계: 분산 학습 시작! 🚀

**모든 컴퓨터에서 거의 동시에 실행** (5초 이내 차이):

#### 컴퓨터 1, 2, 3 모두에서:

```bash
accelerate launch 02_finetune_local.py
```

**중요:** 
- 3개 명령을 **거의 동시에** 실행해야 합니다!
- 먼저 실행한 컴퓨터가 다른 컴퓨터를 기다립니다.
- 약간의 시간 차이는 괜찮지만 너무 오래 걸리면 오류가 날 수 있습니다.

---

## 실행 방법 2: Distributed 직접 사용 (원하는 경우)

### 각 컴퓨터에서 PowerShell 스크립트 실행:

#### 컴퓨터 1에서:

```powershell
# run_distributed_node1.ps1 파일 수정
# $MASTER_ADDR를 컴퓨터 1의 IP로 변경 후
.\run_distributed_node1.ps1
```

#### 컴퓨터 2에서:

```powershell
# run_distributed_node2.ps1 파일 수정
# $MASTER_ADDR를 컴퓨터 1의 IP로 변경 후
.\run_distributed_node2.ps1
```

#### 컴퓨터 3에서:

```powershell
# run_distributed_node3.ps1 파일 수정
# $MASTER_ADDR를 컴퓨터 1의 IP로 변경 후
.\run_distributed_node3.ps1
```

---

## 실행 중 확인 사항

### 정상 실행 시:

- 각 컴퓨터에서 "분산 학습 초기화 완료" 메시지
- GPU 사용률 증가 (Task Manager 또는 `nvidia-smi` 확인)
- 로그 파일 생성 (`./logs` 폴더)

### 오류 발생 시:

1. **연결 오류:**
   - IP 주소 확인
   - 방화벽 설정 확인
   - 모든 컴퓨터가 같은 네트워크에 있는지 확인

2. **동기화 오류:**
   - 모든 컴퓨터에서 같은 데이터 파일 확인
   - `training_data_10000.csv` 파일이 동일한지 확인

3. **GPU 오류:**
   - `nvidia-smi` 실행 확인
   - CUDA 버전 확인

---

## 학습 완료 후

### 모델 저장 위치:

**컴퓨터 1 (메인 노드)에서:**
```
./results/finetuned_e5_large/
```

- `config.json`
- `pytorch_model.bin`
- `tokenizer_config.json`
- `metadata.json`

### 모델 사용:

```bash
python 03_use_finetuned_model.py
```

---

## 요약 체크리스트

### 시작 전 확인:

- [ ] 모든 컴퓨터의 IP 주소 확인
- [ ] Accelerate 설정 완료 (각 컴퓨터에서)
- [ ] 방화벽 설정 완료 (포트 29500)
- [ ] `training_data_10000.csv` 파일 확인 (모든 컴퓨터)
- [ ] `.env` 파일 생성 완료 (필요한 경우)
- [ ] 모든 컴퓨터가 같은 네트워크에 연결됨

### 실행:

- [ ] 3대 컴퓨터에서 거의 동시에 `accelerate launch 02_finetune_local.py` 실행
- [ ] 정상 실행 확인
- [ ] 학습 진행 확인

---

## 빠른 시작 명령어

### 모든 컴퓨터에서:

```bash
# 1. IP 확인
ipconfig

# 2. Accelerate 설정 (한 번만)
accelerate config

# 3. 데이터 확인
dir training_data_10000.csv

# 4. 분산 학습 시작
accelerate launch 02_finetune_local.py
```

**이제 준비 완료! 분산 학습을 시작하세요!** 🚀
