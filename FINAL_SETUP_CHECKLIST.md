# 최종 설정 체크리스트 (컴퓨터 2, 3)

## ✅ Git Pull 후 확인할 것들

### 1단계: Git Pull
```bash
git pull origin main
```

### 2단계: 필수 확인 사항

#### ✅ 패키지 설치 확인
```bash
pip list | findstr "torch transformers accelerate"
```
없으면:
```bash
.\setup_other_computer.ps1
```

#### ✅ GPU 확인
```bash
nvidia-smi
```

#### ✅ 데이터 파일 확인
```bash
dir training_data_10000.csv
```
없으면 Git에 있으면 pull로 받아짐

#### ✅ .env 파일 확인 (필요한 경우)
```bash
dir .env
```
없으면:
```bash
python create_env_file.py
```
그 후 실제 DB 정보로 수정

#### ✅ 방화벽 설정 (중요!)
포트 29500이 열려 있어야 함
관리자 PowerShell에서:
```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed Training" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

---

## 3단계: 실행 명령어

### 컴퓨터 1 (메인) - 먼저 실행:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

### 컴퓨터 2 - 컴퓨터 1 실행 직후:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

### 컴퓨터 3 - 컴퓨터 2 실행 직후:
```powershell
$env:USE_LIBUV="0"
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr=210.93.16.37 --master_port=29500 02_finetune_distributed.py
```

---

## 빠른 체크리스트

각 컴퓨터에서:
- [ ] `git pull origin main` 완료
- [ ] 패키지 설치 확인 (없으면 `.\setup_other_computer.ps1`)
- [ ] GPU 확인 (`nvidia-smi`)
- [ ] `training_data_10000.csv` 파일 확인
- [ ] 방화벽 설정 완료 (포트 29500)
- [ ] 명령어 실행 준비 완료

---

## 실행 순서

1. **컴퓨터 1**: 명령어 실행 (Enter)
2. **1-2초 후 컴퓨터 2**: 명령어 실행 (Enter)
3. **1-2초 후 컴퓨터 3**: 명령어 실행 (Enter)

**끝!** 🚀
