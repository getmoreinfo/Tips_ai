# 빠른 실행 명령어 (5초 내 동시 실행)

## 각 컴퓨터에서 실행할 명령어

### 컴퓨터 1 (메인 노드) - 먼저 실행:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

### 컴퓨터 2 - 컴퓨터 1 실행 직후:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

### 컴퓨터 3 - 컴퓨터 1 실행 직후:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

---

## 실행 순서

1. **컴퓨터 1**: 위 명령어 실행 (Enter)
2. **1-2초 후 컴퓨터 2**: 위 명령어 실행 (Enter)
3. **1-2초 후 컴퓨터 3**: 위 명령어 실행 (Enter)

**중요:** 컴퓨터 1이 먼저 실행되어야 하고, 컴퓨터 2와 3은 거의 동시에 (5초 이내) 실행하세요!
