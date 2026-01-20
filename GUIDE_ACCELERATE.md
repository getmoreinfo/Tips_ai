# HuggingFace Accelerate 완전 가이드

## Accelerate란?

**HuggingFace Accelerate**는 PyTorch 분산 학습을 간단하게 만들어주는 라이브러리입니다.

### 기존 방법의 문제점 (PyTorch Distributed)

```python
# 복잡한 수동 설정이 필요
import torch.distributed as dist
import os

# 환경 변수 수동 설정
os.environ['MASTER_ADDR'] = '192.168.1.100'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '3'

# 분산 초기화
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 모델을 각 GPU에 배치
model = model.to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model)
# ... 복잡한 코드 ...
```

### Accelerate를 사용하면

```python
from accelerate import Accelerator

# 이것만 하면 끝!
accelerator = Accelerator()

# 모델, 데이터로더 자동 준비
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 학습 루프는 동일하게 작성
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    
    # backward도 자동 처리
    accelerator.backward(loss)
    optimizer.step()
```

## Accelerate의 핵심 장점

### 1. **자동 설정**
- GPU 개수 자동 감지
- 멀티 노드 설정 자동 처리
- 메모리 최적화 자동 적용

### 2. **코드 변경 최소화**
- 기존 학습 코드를 거의 그대로 사용
- 단일 GPU 코드 → 멀티 GPU/노드로 쉽게 확장

### 3. **간단한 실행**
```bash
# 단일 GPU
python train.py

# 여러 GPU (같은 컴퓨터)
accelerate launch train.py

# 여러 컴퓨터 (3대)
accelerate launch --multi_gpu --num_machines=3 train.py
```

## 당신의 경우: 3대 컴퓨터에 적용

### 단계 1: Accelerate 설치
```bash
pip install accelerate
```

### 단계 2: 설정 파일 생성 (한 번만)

#### 메인 노드 (첫 번째 컴퓨터)에서:
```bash
accelerate config
```

질문에 답변:
```
- Multi-node training: **yes**
- Main node IP address: **192.168.1.100** (메인 노드 IP)
- Main node port: **29500**
- Total number of nodes: **3**
- Current node rank: **0** (0 = 첫 번째 노드)
- Which GPU(s): **0** (각 컴퓨터에서 GPU 0번 사용)
- Mixed precision: **fp16** (더 빠른 학습)
```

설정 파일이 `~/.cache/huggingface/accelerate/default_config.yaml`에 생성됩니다.

#### 다른 노드에도 설정 (간단히)
- 설정 파일을 복사하거나
- 각 노드에서 `accelerate config`를 다시 실행하고 `node_rank`만 변경 (노드2: 1, 노드3: 2)

### 단계 3: 코드 수정 (최소한)

#### 기존 코드 (`02_finetune_local.py`)
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

#### Accelerate 사용 (거의 변경 없음!)
```python
# Trainer는 이미 Accelerate를 자동 지원!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()  # 그대로 실행하면 자동으로 분산 학습!
```

### 단계 4: 실행 (매우 간단!)

#### 모든 노드에서 동시에 실행:
```bash
accelerate launch 02_finetune_local.py
```

끝! 🎉

## Accelerate vs 수동 Distributed

| 항목 | 수동 Distributed | Accelerate |
|------|-----------------|------------|
| **설정 코드** | 50+ 줄 | 1줄 |
| **환경 변수** | 수동 설정 | 자동 처리 |
| **실행 명령** | 복잡한 launch 옵션 | `accelerate launch` |
| **디버깅** | 어려움 | 쉬움 |
| **유지보수** | 복잡 | 간단 |

## 실제 사용 예시

### 예시 1: 단일 컴퓨터, GPU 여러 개
```bash
accelerate launch --num_processes=4 train.py  # GPU 4개 사용
```

### 예시 2: 여러 컴퓨터 (당신의 경우)
```bash
# 모든 컴퓨터에서 동시에 실행
accelerate launch train.py
```

### 예시 3: Trainer 사용 시 (가장 간단!)
```python
# HuggingFace Trainer는 Accelerate를 자동 지원!
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    ddp_find_unused_parameters=False,  # 분산 학습 최적화
)

trainer = Trainer(...)
trainer.train()  # 그냥 실행하면 자동으로 분산 처리!
```

## 당신의 프로젝트에 바로 적용하기

### 방법 A: 기존 코드 유지 (Trainer 자동 지원)

`02_finetune_local.py`를 거의 그대로 사용:
```bash
# 모든 노드에서
accelerate launch 02_finetune_local.py
```

Trainer가 자동으로 분산 학습을 처리합니다!

### 방법 B: Accelerate 직접 사용 (더 세밀한 제어)

`02_finetune_with_accelerate.py` 생성:
```python
from accelerate import Accelerator

accelerator = Accelerator()

# 모델, 데이터 준비
model, train_dataloader = accelerator.prepare(model, train_dataloader)

# 학습 루프
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # ... 학습 코드 ...
        accelerator.backward(loss)
```

## 주의사항

1. **모든 노드에서 동시 실행**해야 합니다
2. **같은 네트워크**에 연결되어 있어야 합니다
3. **같은 프로젝트 폴더와 데이터**가 있어야 합니다
4. **방화벽 설정**: 포트가 열려 있어야 합니다

## 요약

**Accelerate = 분산 학습을 쉽게!**

- ✅ 복잡한 설정 없음
- ✅ 코드 변경 최소화
- ✅ 자동 최적화
- ✅ 간단한 실행

당신의 3대 컴퓨터 환경에 최적의 솔루션입니다! 🚀
