# Python 3.11.7 설치 가이드 (컴퓨터 2, 3)

## 목표
컴퓨터 1과 동일한 Python 버전(3.11.7)을 설치하여 호환성 문제 해결

---

## 방법 1: Python 공식 사이트에서 다운로드 (추천)

### 1단계: Python 3.11.7 다운로드

1. **Python 3.11.7 다운로드 페이지 방문:**
   - https://www.python.org/downloads/release/python-3117/
   - 또는 직접 링크: https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe

2. **"Windows installer (64-bit)" 다운로드**

### 2단계: 설치

1. **다운로드한 `python-3.11.7-amd64.exe` 실행**

2. **설치 옵션:**
   - ✅ **"Add Python 3.11 to PATH"** 반드시 체크!
   - ✅ "Install launcher for all users" (선택사항)
   - **"Install Now" 클릭**

3. **설치 완료 대기**

### 3단계: 설치 확인

**새로운 PowerShell 창에서:**
```powershell
python --version
```

**출력:**
```
Python 3.11.7
```

---

## 방법 2: Microsoft Store에서 설치

1. **Microsoft Store 열기**
2. **"Python 3.11" 검색**
3. **Python 3.11 설치** (버전이 3.11.7인지 확인)
4. **설치 후 확인:**
   ```powershell
   python --version
   ```

---

## 설치 후 필수 작업

### 1. 기존 Python 3.13 제거 (선택사항)

Python 3.13을 완전히 제거하고 싶다면:
- **설정 → 앱 → Python 3.13.2 제거**

**또는:**
- PATH 환경 변수에서 Python 3.13 경로 제거

### 2. Python 3.11.7이 기본 버전인지 확인

```powershell
# 어느 Python이 실행되는지 확인
where python
python --version

# Python 3.11.7이 나와야 함
```

---

## 패키지 재설치

### 1단계: pip 업그레이드

```powershell
python -m pip install --upgrade pip
```

### 2단계: requirements.txt로 정확한 버전 설치

```powershell
# 프로젝트 폴더로 이동
cd C:\Users\comso-1407\dev\ai-tr

# requirements.txt로 설치
pip install -r requirements.txt
```

**주의:** PyTorch는 별도 인덱스가 필요하므로:
```powershell
# PyTorch 먼저 설치
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
pip install transformers==4.57.6 datasets==4.5.0 accelerate==1.12.0 pandas==2.3.3 numpy==2.3.5 scikit-learn==1.8.0 psycopg2-binary==2.9.11 python-dotenv==1.2.1
```

### 3단계: 설치 확인

```powershell
python --version
pip list | findstr "torch transformers accelerate"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 테스트 실행

```powershell
python test_training_quick.py
```

**정상 작동하면:**
```
✅ 테스트 성공!
학습이 정상적으로 작동합니다!
```

---

## 문제 해결

### Python 3.11.7이 인식되지 않을 때

1. **새로운 PowerShell 창 열기** (환경 변수 새로고침)

2. **PATH 확인:**
   ```powershell
   $env:PATH -split ';' | Select-String "Python"
   ```

3. **직접 Python 3.11.7 경로로 실행:**
   ```powershell
   # 일반적인 설치 경로
   C:\Users\<사용자명>\AppData\Local\Programs\Python\Python311\python.exe --version
   ```

### 여러 Python 버전이 설치되어 있을 때

**특정 버전 실행:**
```powershell
# Python 3.11만 사용
py -3.11 --version

# 가상 환경 사용 권장
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 요약

1. ✅ Python 3.11.7 다운로드 및 설치
2. ✅ "Add Python to PATH" 체크 필수!
3. ✅ 새 PowerShell 창에서 `python --version` 확인
4. ✅ `pip install -r requirements.txt` 실행
5. ✅ `python test_training_quick.py` 테스트

**이제 컴퓨터 1, 2, 3 모두 Python 3.11.7로 통일됩니다!** ✅
