# 멀티 노드 분산 학습 가이드

3대의 컴퓨터(각각 4060TI GPU 1개)를 연결하여 AI 모델 학습을 수행하는 방법입니다.

## 방법 1: PyTorch Distributed Training (추천)

### 준비 사항

1. **3대 컴퓨터 모두 필요한 것:**
   - Python 및 필요한 패키지 설치 (같은 버전)
   - 학습 데이터 파일 (`training_data_10000.csv`)
   - 프로젝트 폴더 동기화

2. **네트워크 설정:**
   - 3대 컴퓨터가 같은 네트워크에 연결되어 있어야 함
   - 각 컴퓨터의 IP 주소 확인
   - 방화벽에서 포트 오픈 (기본: 29500)

### 실행 방법

#### 노드 1 (메인 노드, IP: 예시 192.168.1.100)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=3 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    02_finetune_distributed.py
```

#### 노드 2 (IP: 예시 192.168.1.101)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=3 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    02_finetune_distributed.py
```

#### 노드 3 (IP: 예시 192.168.1.102)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=3 \
    --node_rank=2 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    02_finetune_distributed.py
```

**중요:** 3개 명령을 동시에 실행해야 합니다!

---

## 방법 2: HuggingFace Accelerate (더 간단)

### 1. Accelerate 설치
```bash
pip install accelerate
```

### 2. Accelerate 설정 (메인 노드에서만)
```bash
accelerate config
```
질문에 답변:
- Multi-node training: **yes**
- Main node IP address: **192.168.1.100** (메인 노드 IP)
- Main node port: **29500**
- Total number of nodes: **3**
- Current node rank: **0** (노드1: 0, 노드2: 1, 노드3: 2)
- Which GPU(s): **0** (각 노드에서 GPU 0번 사용)

### 3. 설정 파일 배포
`accelerate config`로 생성된 설정 파일을 다른 노드에도 복사하고, 각 노드에서 `node_rank`를 수정합니다.

### 4. 실행 (모든 노드에서 동시 실행)
```bash
accelerate launch 02_finetune_distributed.py
```

---

## 방법 3: 간단한 스크립트로 자동화

### Windows PowerShell 스크립트 생성

`run_distributed_node1.ps1` (메인 노드):
```powershell
$env:MASTER_ADDR="192.168.1.100"
$env:MASTER_PORT="29500"
$env:NODE_RANK="0"
$env:WORLD_SIZE="3"

python -m torch.distributed.launch `
    --nproc_per_node=1 `
    --nnodes=3 `
    --node_rank=0 `
    --master_addr=$env:MASTER_ADDR `
    --master_port=$env:MASTER_PORT `
    02_finetune_distributed.py
```

`run_distributed_node2.ps1`:
```powershell
$env:MASTER_ADDR="192.168.1.100"
$env:MASTER_PORT="29500"
$env:NODE_RANK="1"
$env:WORLD_SIZE="3"

python -m torch.distributed.launch `
    --nproc_per_node=1 `
    --nnodes=3 `
    --node_rank=1 `
    --master_addr=$env:MASTER_ADDR `
    --master_port=$env:MASTER_PORT `
    02_finetune_distributed.py
```

`run_distributed_node3.ps1`:
```powershell
$env:MASTER_ADDR="192.168.1.100"
$env:MASTER_PORT="29500"
$env:NODE_RANK="2"
$env:WORLD_SIZE="3"

python -m torch.distributed.launch `
    --nproc_per_node=1 `
    --nnodes=3 `
    --node_rank=2 `
    --master_addr=$env:MASTER_ADDR `
    --master_port=$env:MASTER_PORT `
    02_finetune_distributed.py
```

### 실행
각 노드에서 해당 스크립트를 **동시에** 실행합니다.

---

## 주의사항

1. **동시 실행:** 모든 노드에서 스크립트를 거의 동시에 실행해야 합니다.

2. **네트워크 연결:** 모든 컴퓨터가 같은 네트워크에 있고 서로 접근 가능해야 합니다.

3. **방화벽:** Windows 방화벽에서 포트 29500을 허용해야 합니다.

4. **파일 동기화:** 모든 노드에 같은 프로젝트 폴더와 데이터 파일이 있어야 합니다.

5. **Python 패키지 버전:** 모든 노드에서 같은 버전의 PyTorch, transformers 등을 사용해야 합니다.

---

## 문제 해결

### 연결 오류
- IP 주소 확인: `ipconfig` (Windows) 또는 `ip addr` (Linux)
- 방화벽 설정 확인
- 모든 노드가 같은 네트워크에 있는지 확인

### 동기화 오류
- 모든 노드에서 같은 데이터 파일 사용 확인
- random_state 설정 확인

### 성능
- 네트워크 속도가 중요합니다 (가능하면 유선 연결 권장)
- 로컬 네트워크(Gigabit 이상) 권장

---

## 대안: 단일 노드 멀티 GPU

만약 같은 컴퓨터에 GPU 여러 개가 있다면, 더 간단하게 사용할 수 있습니다:

```bash
# 같은 컴퓨터에 GPU 2개가 있는 경우
python -m torch.distributed.launch --nproc_per_node=2 02_finetune_distributed.py
```

HuggingFace Trainer는 자동으로 멀티 GPU를 감지하므로, 코드 수정 없이도 사용할 수 있습니다.
