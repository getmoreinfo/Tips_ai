# 다른 컴퓨터 설정 단계별 가이드

## ✅ 네, 맞습니다!

다른 두 컴퓨터에서 다음 순서로 진행하시면 됩니다:

---

## 📋 체크리스트 (각 컴퓨터마다)

### 1단계: Cursor 설치 (선택사항)

**Cursor는 필수가 아닙니다!** 하지만 대화 내용을 이어서 사용하려면 설치하는 것이 좋습니다.

- [Cursor 공식 사이트](https://cursor.sh/)에서 다운로드
- 설치 후 실행

**참고:** Cursor 없이도 PowerShell이나 명령 프롬프트로 실행 가능합니다.

---

### 2단계: Git 클론

#### PowerShell 또는 명령 프롬프트에서:

```bash
# 원하는 위치로 이동 (예: C:\Users\<사용자명>\dev)
cd C:\Users\<사용자명>\dev

# 프로젝트 클론
git clone https://github.com/getmoreinfo/ai-tr.git

# 프로젝트 폴더로 이동
cd ai-tr
```

**완료!** 모든 파일이 복사됩니다.

---

### 3단계: Python 및 패키지 설치

#### 자동 설치 스크립트 사용 (가장 쉬움):

```powershell
.\setup_other_computer.ps1
```

이 스크립트가 자동으로:
- PyTorch 설치
- 필요한 패키지 설치
- 설치 확인

#### 수동 설치 (원하는 경우):

```bash
# PyTorch (CUDA 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 필요한 패키지
pip install transformers datasets scikit-learn pandas psycopg2-binary python-dotenv accelerate
```

---

### 4단계: .env 파일 생성

```bash
python create_env_file.py
```

그 후 `.env` 파일을 열어서 실제 데이터베이스 정보로 수정하세요.

---

### 5단계: Cursor에서 프로젝트 열기 (Cursor 설치한 경우)

1. Cursor 실행
2. **File** → **Open Folder**
3. 클론한 `ai-tr` 폴더 선택

**완료!** 이제 이 컴퓨터에서도 대화를 이어서 할 수 있습니다.

---

## 🚀 실행 테스트

### 데이터베이스 연결 테스트:

```bash
python 00_db_smoke_test.py
```

### 샘플 추출 (이미 있으면 생략):

```bash
python 01_export_sample_10000.py
```

### 파인튜닝 실행:

```bash
python 02_finetune_local.py
```

---

## 📝 요약

각 컴퓨터에서:

1. ✅ **Cursor 설치** (선택사항)
2. ✅ **Git 클론**: `git clone https://github.com/getmoreinfo/ai-tr.git`
3. ✅ **패키지 설치**: `.\setup_other_computer.ps1`
4. ✅ **.env 파일 생성**: `python create_env_file.py`
5. ✅ **Cursor에서 프로젝트 열기** (선택사항)

**끝!** 🎉

---

## 💡 팁

### Cursor 대화 내용 동기화

Cursor의 대화 내용은 자동으로 동기화되지 않습니다. 하지만:
- 같은 프로젝트 폴더를 열면 Cursor가 인식합니다
- 중요한 내용은 프로젝트의 `.md` 파일로 저장되어 있습니다

### Git으로 최신 파일 받기

다른 컴퓨터에서 업데이트된 파일을 받으려면:

```bash
git pull origin main
```

### 파일 수정 후 푸시

다른 컴퓨터에서 파일을 수정했다면:

```bash
git add .
git commit -m "Update from computer 2"
git push origin main
```

---

## ⚠️ 주의사항

1. **.env 파일**: 각 컴퓨터에서 별도로 생성해야 합니다 (Git에 올라가지 않음)
2. **패키지 버전**: 모든 컴퓨터에서 같은 버전 사용 권장
3. **네트워크**: 분산 학습을 하려면 같은 네트워크에 연결되어 있어야 합니다

---

## 🎯 빠른 명령어 모음

```bash
# 1. 클론
git clone https://github.com/getmoreinfo/ai-tr.git
cd ai-tr

# 2. 패키지 설치
.\setup_other_computer.ps1

# 3. .env 생성
python create_env_file.py

# 4. 테스트
python 00_db_smoke_test.py
```

**이제 준비 완료!** 🚀
