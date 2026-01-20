# Python 버전 동기화 가이드

## ⚠️ 중요: Python 버전 문제

컴퓨터 1에서 정상 작동하지만 컴퓨터 2, 3에서 오류가 발생한다면 **Python 버전 차이** 때문일 가능성이 높습니다.

---

## 해결 방법

### 1단계: 컴퓨터 1의 Python 버전 확인

**컴퓨터 1에서 실행:**
```bash
python --version
python -c "import sys; print(sys.version)"
```

**결과 예시:**
```
Python 3.11.5
3.11.5 (tags/v3.11.5, Dec  7 2023, 16:57:41) [MSC v.1937 64 bit (AMD64)]
```

---

### 2단계: 컴퓨터 2, 3에 동일한 Python 버전 설치

#### 옵션 A: Python 공식 사이트에서 다운로드 (추천)

1. **컴퓨터 1에서 확인한 Python 버전과 동일한 버전 다운로드:**
   - [Python 다운로드 페이지](https://www.python.org/downloads/)
   - 예: Python 3.11.5를 사용 중이면 Python 3.11.5 다운로드

2. **컴퓨터 2, 3에서 설치:**
   - 다운로드한 설치 파일 실행
   - ✅ **"Add Python to PATH"** 체크 필수!
   - "Install Now" 클릭

3. **설치 확인:**
   ```bash
   python --version
   ```

#### 옵션 B: Microsoft Store에서 설치

1. Microsoft Store 열기
2. "Python 3.11" 검색 (컴퓨터 1 버전과 동일하게)
3. 설치

---

### 3단계: 패키지 재설치 (정확한 버전)

**컴퓨터 2, 3에서 실행:**

```bash
# 프로젝트 폴더로 이동
cd C:\Users\<사용자명>\dev\ai-tr

# pip 업그레이드
python -m pip install --upgrade pip

# requirements.txt로 정확한 버전 설치
pip install -r requirements.txt
```

**또는 기존 패키지 삭제 후 재설치:**

```bash
# 기존 패키지 삭제 (선택사항)
pip uninstall torch torchvision torchaudio transformers datasets accelerate pandas numpy scikit-learn psycopg2-binary python-dotenv -y

# 정확한 버전으로 재설치
pip install -r requirements.txt
```

---

### 4단계: 버전 확인

**모든 컴퓨터에서 실행하여 비교:**

```bash
python --version
pip list | findstr "torch transformers accelerate pandas numpy"
```

**컴퓨터 1과 동일해야 합니다!**

---

## Python 버전 호환성 안내

### 권장 버전:

- ✅ **Python 3.10.x** (가장 안정적)
- ✅ **Python 3.11.x** (권장)
- ⚠️ **Python 3.12.x** (일부 패키지 호환성 확인 필요)
- ⚠️ **Python 3.13.x** (최신 버전, 호환성 문제 가능)

**컴퓨터 1이 Python 3.10 또는 3.11을 사용 중이면, 컴퓨터 2, 3도 동일하게 맞추는 것을 강력히 권장합니다.**

---

## 빠른 해결 체크리스트

### 컴퓨터 1에서:
- [ ] `python --version` 실행하여 버전 확인
- [ ] 버전 정보를 컴퓨터 2, 3에 전달

### 컴퓨터 2, 3에서:
- [ ] 컴퓨터 1과 동일한 Python 버전 설치
- [ ] `python --version` 확인
- [ ] 기존 패키지 삭제 (선택사항)
- [ ] `pip install -r requirements.txt` 실행
- [ ] `python test_training_quick.py` 테스트 실행
- [ ] 정상 작동 확인

---

## 문제가 계속되면

1. **컴퓨터 1의 모든 패키지 버전 확인:**
   ```bash
   pip freeze > requirements_computer1.txt
   ```

2. **requirements_computer1.txt를 컴퓨터 2, 3에 복사**

3. **컴퓨터 2, 3에서:**
   ```bash
   pip install -r requirements_computer1.txt
   ```

---

## 요약

**가장 간단한 해결 방법:**
1. 컴퓨터 1의 Python 버전 확인
2. 컴퓨터 2, 3에 동일한 버전 설치
3. `pip install -r requirements.txt` 실행
4. 테스트 실행

**이렇게 하면 모든 컴퓨터에서 동일한 환경이 구성됩니다!** ✅
