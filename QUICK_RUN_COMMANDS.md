# 빠른 실행 명령어 (5초 내 동시 실행)

## 각 컴퓨터에서 PowerShell에 직접 입력할 명령어

### 컴퓨터 1 (메인 노드) - 먼저 실행:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

### 컴퓨터 2 - 컴퓨터 1 실행 직후 (1-2초 후):
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

### 컴퓨터 3 - 컴퓨터 2 실행 직후 (1-2초 후):
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

---

## 실행 순서

1. **컴퓨터 1**: 위 명령어 복사 → PowerShell에 붙여넣기 → Enter
2. **1-2초 후 컴퓨터 2**: 위 명령어 복사 → PowerShell에 붙여넣기 → Enter
3. **1-2초 후 컴퓨터 3**: 위 명령어 복사 → PowerShell에 붙여넣기 → Enter

**중요:** 
- 컴퓨터 1이 먼저 실행되어야 합니다 (메인 노드)
- 컴퓨터 2와 3은 컴퓨터 1 실행 후 5초 이내에 실행하세요
- 각 컴퓨터의 PowerShell에서 직접 명령어를 입력하면 됩니다
