# Git 커밋 가이드

## Git 설정 (처음 한 번만)

Git 커밋을 위해 사용자 정보 설정이 필요합니다:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**또는 이 저장소에만 적용:**
```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

## 커밋할 파일들

다음 파일들이 Git에 추가되었습니다:

### Accelerate 설정 스크립트:
- ✅ `setup_accelerate_computer1.ps1` - 컴퓨터 1 (메인, rank=0)
- ✅ `setup_accelerate_computer2.ps1` - 컴퓨터 2 (rank=1)
- ✅ `setup_accelerate_computer3.ps1` - 컴퓨터 3 (rank=2)

### 가이드 문서:
- ✅ `START_DISTRIBUTED_QUICK.md` - 빠른 시작 가이드
- ✅ `accelerate_config_guide.md` - 상세 설정 가이드
- ✅ `INSTALL_PYTHON_3.11.7.md` - Python 설치 가이드
- ✅ `SYNC_PYTHON_VERSION.md` - Python 버전 동기화 가이드
- ✅ `MANUAL_INSTALL_GUIDE.md` - 수동 설치 가이드
- ✅ `SETUP_DISTRIBUTED_TRAINING_STEP_BY_STEP.md` - 단계별 분산 학습 가이드

### 기타:
- ✅ `requirements.txt` - 패키지 버전 목록
- ✅ `install_packages_for_python311.ps1` - Python 3.11 패키지 설치 스크립트

---

## 커밋 및 푸시 명령어

```powershell
# 1. Git 사용자 정보 설정 (처음 한 번만)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 2. 커밋
git commit -m "Add Accelerate setup scripts and distributed training guides

- Add Accelerate setup scripts for all 3 computers
- Add quick start guide for distributed training
- Add Python 3.11 installation and sync guides
- Add requirements.txt with exact package versions
- IP addresses: 210.93.16.37 (computer1), 210.93.16.36 (computer2), 210.93.16.35 (computer3)
- Port: 29500"

# 3. 푸시
git push origin main
```

---

## 커밋 후 다른 컴퓨터에서

컴퓨터 1과 2에서:

```powershell
git pull origin main
```

그 후 각 컴퓨터에서:
- 컴퓨터 1: `.\setup_accelerate_computer1.ps1`
- 컴퓨터 2: `.\setup_accelerate_computer2.ps1`

---

**Git 설정 후 커밋하면 모든 컴퓨터에서 동일한 설정을 사용할 수 있습니다!** ✅
