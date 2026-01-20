# 수동 설치 가이드 (Python 3.11.7)

## 📥 Python 3.11.7 다운로드

### 방법 1: Python 공식 사이트 (추천)

**다운로드 링크:**
- **직접 다운로드:** https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
- **공식 릴리스 페이지:** https://www.python.org/downloads/release/python-3117/

**설치 단계:**
1. 위 링크 클릭하여 `python-3.11.7-amd64.exe` 다운로드
2. 다운로드한 파일 실행
3. ✅ **"Add Python 3.11 to PATH"** 반드시 체크!
4. "Install Now" 클릭
5. 설치 완료 대기

### 방법 2: Python 공식 다운로드 페이지

**웹사이트:** https://www.python.org/downloads/

1. 페이지 접속
2. "Python 3.11.7" 클릭 (또는 "Download Python 3.11.7" 버튼)
3. "Windows installer (64-bit)" 다운로드
4. 위와 동일하게 설치

---

## 🔧 설치 후 확인

**새로운 PowerShell 창 열기** (중요! 환경 변수 새로고침을 위해)

```powershell
python --version
```

**출력되어야 함:**
```
Python 3.11.7
```

**만약 Python 3.13.2가 나온다면:**
- PATH 환경 변수에서 Python 3.11 경로가 더 앞에 있어야 함
- 또는 `py -3.11 --version` 명령 사용

---

## 📦 패키지 설치 (Python 3.11로)

### 방법 1: 자동 스크립트 사용

```powershell
# 프로젝트 폴더로 이동
cd C:\Users\comso-1407\dev\ai-tr

# 자동 설치 스크립트 실행
.\install_packages_for_python311.ps1
```

### 방법 2: 수동 설치

```powershell
# 1. pip 업그레이드
python -m pip install --upgrade pip

# 2. PyTorch 설치 (CUDA 11.8)
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. 나머지 패키지 설치
python -m pip install transformers==4.57.6 datasets==4.5.0 accelerate==1.12.0 pandas==2.3.3 numpy==2.3.5 scikit-learn==1.8.0 psycopg2-binary==2.9.11 python-dotenv==1.2.1
```

---

## ⚠️ Python 3.11이 기본 버전이 아닐 때

**Python 3.11 경로 직접 사용:**
```powershell
$python311 = "C:\Users\comso-1407\AppData\Local\Programs\Python\Python311\python.exe"

# 버전 확인
& $python311 --version

# 패키지 설치
& $python311 -m pip install --upgrade pip
& $python311 -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
& $python311 -m pip install transformers==4.57.6 datasets==4.5.0 accelerate==1.12.0 pandas==2.3.3 numpy==2.3.5 scikit-learn==1.8.0 psycopg2-binary==2.9.11 python-dotenv==1.2.1
```

---

## ✅ 설치 확인

```powershell
python --version
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
pip list | findstr "torch transformers accelerate"
```

---

## 🔗 주요 다운로드 링크 요약

1. **Python 3.11.7 다운로드:**
   - 직접: https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
   - 페이지: https://www.python.org/downloads/release/python-3117/

2. **프로젝트 폴더:**
   - `C:\Users\comso-1407\dev\ai-tr`

3. **설치 스크립트:**
   - `.\install_packages_for_python311.ps1`

---

## 📝 설치 체크리스트

- [ ] Python 3.11.7 다운로드
- [ ] "Add Python to PATH" 체크하고 설치
- [ ] 새 PowerShell 창 열기
- [ ] `python --version` 확인 (3.11.7)
- [ ] `pip install --upgrade pip` 실행
- [ ] PyTorch 설치 (위 명령 참조)
- [ ] 나머지 패키지 설치
- [ ] `python test_training_quick.py` 테스트

---

**설치 중 문제가 있으면 알려주세요!**
