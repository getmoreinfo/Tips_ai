# Google Colab + VSCode 연동 가이드

> **최신:** Colab ↔ Cursor 워크플로우(코드 작성 → Colab 학습)는 **`COLAB_CURSOR_WORKFLOW.md`**를 참고하세요.

## 연동 방법 개요

Google Colab과 VSCode를 연동하는 방법은 여러 가지가 있습니다:

### 방법 1: Google Colab (공식 확장) - 추천
- 확장명: **Google Colab** (`Google.colab`). 구 "Colab Code" 후속.
- VSCode/Cursor에서 `.ipynb` 열고 **Select Kernel → Colab** 선택 후 실행
- 가장 공식적이고 안정적

### 방법 2: Remote SSH (간접적)
- Colab의 SSH 기능을 통해 원격 접속
- 완전한 VSCode 환경 사용 가능

### 방법 3: Colab에서 직접 파일 편집
- Colab 내장 코드 에디터 사용
- 간단하지만 VSCode 기능은 제한적

---

## 방법 1: Google Colab 확장 사용 (가장 추천)

### Step 1: VSCode/Cursor 확장 설치
1. VSCode 또는 Cursor 열기
2. Extensions (Ctrl+Shift+X / Cmd+Shift+X)
3. **"Google Colab"** 검색 및 설치
4. 또는 직접 설치: https://marketplace.visualstudio.com/items?itemName=Google.colab

### Step 2: Colab 노트북 사용
1. `.ipynb` 파일 열기 (로컬 또는 Drive)
2. 우측 상단 **Select Kernel** 클릭 → **Colab** 선택
3. Google 로그인 후 런타임(CPU/GPU/TPU) 선택
4. 또는 `Ctrl+Shift+P` → "Colab: Open Notebook"으로 Drive 노트 열기

### Step 3: 사용 방법
- VSCode/Cursor에서 노트북 편집
- 셀 실행은 **Colab 클라우드 런타임**에서 동작
- 파일 편집·Git·AI 등은 에디터 기능 활용

### 장점
- ✅ VSCode의 모든 기능 사용 가능
- ✅ IntelliSense, 디버깅 등 지원
- ✅ Git 통합 가능
- ✅ Cursor AI도 함께 사용 가능

---

## 방법 2: Remote SSH (고급)

### Step 1: Colab에서 SSH 활성화
Colab 노트북에서:
```python
# SSH 연결 설정
!pip install colab_ssh --upgrade
from colab_ssh import setup_ssh, init_git
setup_ssh(ngrok_region="us")
```

### Step 2: VSCode Remote SSH 설정
1. VSCode에서 "Remote - SSH" 확장 설치
2. SSH 연결 정보 입력
3. Colab 서버에 연결

### 장점
- ✅ 완전한 원격 개발 환경
- ✅ 모든 VSCode 기능 사용
- ✅ 터미널 접근 가능

### 단점
- ⚠️ 설정이 복잡함
- ⚠️ 세션이 끊기면 재연결 필요

---

## 방법 3: Cursor에서 Colab 연동

### Cursor는 VSCode 기반이므로 동일하게 작동합니다!

### Step 1: Cursor에서 Google Colab 확장 설치
1. Cursor 열기
2. Extensions (Ctrl+Shift+X / Cmd+Shift+X)
3. **"Google Colab"** 검색 및 설치

### Step 2: Colab 노트북 사용
1. `.ipynb` 열기 → **Select Kernel → Colab**
2. 또는 `Ctrl+Shift+P` → "Colab: Open Notebook" → Drive에서 선택

### Step 3: Cursor AI 기능 활용
- Cursor의 AI 기능으로 코드 작성
- Colab 런타임에서 실행
- 결과 확인 및 수정

---

## 실제 워크플로우 예시

### 시나리오: Qwen 7B 학습

#### 1. Colab에서 노트북 생성
- Google Colab에서 새 노트북 생성
- 기본 설정 코드 작성

#### 2. Cursor에서 열기
```bash
# Cursor에서
Ctrl+Shift+P → "Colab: Open Notebook"
# Google Drive의 .ipynb 파일 선택
```

#### 3. Cursor AI로 코드 작성
- Cursor의 AI 기능으로 학습 코드 작성/수정
- IntelliSense로 자동완성
- 디버깅 기능 활용

#### 4. Colab에서 실행
- 셀 실행은 Colab 런타임 사용
- GPU 할당 확인
- 학습 진행 상황 모니터링

#### 5. 결과 확인 및 수정
- Cursor에서 결과 파일 확인
- 필요시 코드 수정
- Git으로 버전 관리

---

## Cursor AI와의 연동

### Cursor의 장점
1. **AI 코드 작성**: Colab 코드를 Cursor AI로 작성
2. **코드 리뷰**: AI가 코드를 검토하고 개선 제안
3. **디버깅**: AI가 오류를 찾고 수정
4. **문서화**: AI가 코드 설명 생성

### 사용 예시
```python
# Cursor AI에게 요청:
# "Qwen 7B 모델 학습 코드를 작성해줘"
# → Cursor가 코드 생성
# → Colab에서 실행
```

---

## 추천 워크플로우

### 옵션 A: Google Colab 확장 (간단)
1. Cursor에 **Google Colab** 확장 설치
2. `.ipynb` 열고 Select Kernel → Colab 선택
3. Cursor AI로 코드 작성
4. Colab 런타임에서 셀 실행

### 옵션 B: 파일 동기화 (유연함)
1. Google Drive에 파일 업로드
2. Cursor에서 로컬 파일 편집
3. Drive 동기화로 Colab에 반영
4. Colab에서 실행

### 옵션 C: Git 사용 (프로페셔널)
1. GitHub에 코드 저장
2. Cursor에서 로컬로 클론
3. Cursor AI로 개발
4. Git push
5. Colab에서 Git pull 후 실행

---

## 주의사항

1. **세션 관리**: Colab 세션이 끊기면 재연결 필요
2. **파일 경로**: Colab과 로컬의 경로가 다름
3. **동기화**: 파일 변경 시 동기화 확인 필요
4. **권한**: Google Drive 접근 권한 필요

---

## 빠른 시작 체크리스트

- [ ] Cursor (또는 VS Code) 설치 완료
- [ ] **Google Colab** 확장 설치
- [ ] Google 로그인, Colab GPU 런타임 선택 가능 확인
- [ ] (선택) Google Drive에 프로젝트 파일 업로드
- [ ] Colab 노트북 생성 또는 기존 `.ipynb` 사용
- [ ] Cursor에서 노트북 열기 → Kernel → Colab
- [ ] 학습 코드 작성 및 Colab에서 실행

---

## 문제 해결

### Google Colab 확장이 작동 안 할 때
- VSCode/Cursor 재시작 (또는 Reload Window)
- **Google Colab** 확장 재설치
- Google 계정 재로그인

### 파일을 찾을 수 없을 때
- Google Drive 경로 확인
- 파일 권한 확인
- Drive 동기화 확인

### 런타임 연결 안 될 때
- Colab에서 런타임 연결 확인
- 새 런타임 할당
- 노트북 새로고침

---

이제 Cursor에서 Colab을 사용할 준비가 되었습니다! 🚀
