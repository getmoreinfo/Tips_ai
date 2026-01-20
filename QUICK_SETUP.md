# 빠른 설정 가이드 (다른 컴퓨터)

## 1단계: 필수 설치

### Python 확인
```powershell
python --version
```
Python 3.8+ 필요

### GPU 확인
```powershell
nvidia-smi
```
GPU 드라이버가 설치되어 있어야 함

## 2단계: 자동 설치 (추천!)

### 프로젝트 폴더에서 실행:
```powershell
.\setup_other_computer.ps1
```

이 스크립트가 자동으로:
- PyTorch 설치
- 필요한 패키지 설치
- 설치 확인

## 3단계: 수동 설치 (원하는 경우)

### 패키지 설치:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets scikit-learn pandas psycopg2-binary python-dotenv accelerate
```

## 4단계: 프로젝트 파일 복사

### 필요한 파일:
- 모든 `.py` 파일
- `training_data_10000.csv`
- `.env` 파일 (필요시)

### 복사 방법:
1. USB로 복사
2. 네트워크 공유 폴더 사용
3. Git 사용

## 5단계: 네트워크 설정

### IP 주소 확인:
```powershell
ipconfig
```

### 방화벽 설정:
- 포트 29500 열기 (Windows 방화벽)

## 6단계: Accelerate 설정

```bash
accelerate config
```

질문에 답변:
- Multi-node: **yes**
- Main node IP: **메인 노드 IP**
- Total nodes: **3**
- Current rank: **0** (노드1), **1** (노드2), **2** (노드3)

## 7단계: 실행!

```bash
accelerate launch 02_finetune_local.py
```

**모든 컴퓨터에서 동시에 실행**

---

## 체크리스트

각 컴퓨터에서:
- [ ] Python 설치됨
- [ ] GPU 드라이버 설치됨
- [ ] 패키지 설치됨 (`setup_other_computer.ps1` 실행)
- [ ] 프로젝트 파일 복사됨
- [ ] IP 주소 확인됨
- [ ] Accelerate 설정됨

---

## Cursor는 필요 없습니다!

**Cursor는 코드 편집용 IDE입니다.** 
- 학습 실행에는 필요 없음
- PowerShell 또는 명령 프롬프트로 충분
- VS Code, Notepad++ 등 다른 편집기 사용 가능

**중요한 것:**
- ✅ Python
- ✅ 패키지
- ✅ 프로젝트 파일
- ✅ 네트워크 연결
