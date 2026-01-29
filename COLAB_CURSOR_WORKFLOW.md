# Colab + VS Code + Cursor 연동 가이드

**목표:** Cursor AI로 AI 학습 코드를 만들고, Colab GPU에서 실행하는 워크플로우

---

## 1. 전체 구조 요약

```
┌─────────────────┐     연동      ┌─────────────────┐     동일 에디터    ┌─────────────────┐
│  Google Colab   │ ◄──────────►  │  VS Code        │ ◄──────────────►  │  Cursor         │
│  (GPU 런타임)   │   공식 확장    │  (에디터)       │   Cursor = VS Code 기반  │  (AI 보조 코딩)  │
└────────┬────────┘               └────────┬────────┘                    └────────┬────────┘
         │                                 │                                     │
         │  .ipynb 실행                    │  .py / .ipynb 편집                   │  AI로 코드 생성
         │  GPU 학습                       │  Git, 디버깅 등                      │  리팩터·문서화
         └────────────────────────────────┴─────────────────────────────────────┘
                                    같은 프로젝트 폴더 / Git repo
```

- **Colab**: 클라우드 GPU/TPU, `.ipynb` 실행
- **VS Code / Cursor**: 로컬에서 편집, Git, AI 지원. **Cursor는 VS Code 포크**라 확장·설정 호환
- **연동**: VS Code/Cursor에 **Google Colab 확장** 설치 → Colab을 **커널**로 선택해 노트북 실행

---

## 2. Colab ↔ VS Code 연동 (공식 확장)

### 2-1. 확장 설치

1. **VS Code** 또는 **Cursor** 실행
2. `Ctrl+Shift+X` (Mac: `Cmd+Shift+X`) → Extensions
3. **"Google Colab"** 검색
4. **Google** 공식 확장 설치  
   - Marketplace: https://marketplace.visualstudio.com/items?itemName=Google.colab

### 2-2. 노트북에서 Colab 커널 사용

1. `.ipynb` 파일 열기 (또는 새 노트북 생성)
2. 우측 상단 **"Select Kernel"** 클릭
3. **"Colab"** 선택
4. Google 로그인 후 **런타임 선택** (CPU / GPU / TPU 등)
5. 셀 실행 시 **Colab 클라우드**에서 돌아감 (로컬 PC 아님)

### 2-3. 정리

- Colab ↔ VS Code 연동 = **같은 에디터에서 Colab을 커널로 쓰는 것**
- VS Code 대신 **Cursor**를 써도 동일 (같은 확장 사용)

---

## 3. VS Code ↔ Cursor 연동

**Cursor는 VS Code 기반**이라:

- VS Code 확장 대부분 사용 가능 (Google Colab 포함)
- 설정(`settings.json`), 키 바인딩, 테마 호환
- **VS Code “연동”이라기보다, Cursor를 VS Code 대신 쓰면 됨**

### 3-1. Cursor에서 Colab 쓰기

1. Cursor에 **Google Colab** 확장 설치 (위 2-1과 동일)
2. `Ctrl+Shift+P` → **"Colab: Open Notebook"**  
   또는 `.ipynb` 열고 **Select Kernel → Colab**
3. Colab 런타임 선택 후 셀 실행

### 3-2. Cursor만 쓸 때

- Colab 연동 없이 **로컬에서만** 개발할 때는:
  - Cursor로 `.py` / `.ipynb` 편집
  - 로컬 Python / Jupyter 커널로 실행
- Colab에서 **학습**만 할 때는:
  - Cursor에서 Colab 커널 선택 후 동일 파일 실행

---

## 4. 목표 워크플로우: Cursor AI → Colab 학습

**“Cursor로 코드 작성 → Colab에서 학습”** 흐름입니다.

### 4-1. 방법 A: Colab 확장 + 노트북 (가장 직접적)

1. **프로젝트 폴더**를 Cursor로 연다 (예: `ai-tr`)
2. **학습용 노트북** 만들거나 기존 `.ipynb` 사용  
   - 예: `colab_train_qwen7b.ipynb`  
   - 또는 `colab_train_qwen7b.py`를 `%run` 하는 노트북
3. **Select Kernel → Colab** 선택 후 GPU 런타임 연결
4. **Cursor AI**로:
   - 학습 코드 작성·수정
   - 에러 해결, 리팩터링, 주석/문서화
5. **Colab 런타임**에서 셀 실행 → GPU 학습
6. 결과·체크포인트는 **Google Drive** 등에 저장 (기존 Colab 사용법과 동일)

### 4-2. 방법 B: Git + Colab (코드 버전 관리)

1. **Cursor**에서 로컬 프로젝트 편집 (`.py` / `.ipynb`)
2. **Git**으로 `git push` (GitHub 등)
3. **Colab** 노트북에서:
   - `!git clone ...` 후 `!python 23_train_report_summary_lora.py ...`  
   - 또는 `COLAB_QUICK_START.md` / `COLAB_SETUP_GUIDE.md` 방식으로 실행
4. 학습 코드 변경 시: Cursor에서 수정 → push → Colab에서 pull 후 재실행

### 4-3. 방법 C: Drive 동기화

1. **Google Drive**에 프로젝트 폴더 동기화 (예: `tips_ai_colab`)
2. **Cursor**에서 Drive 폴더 열거나, 로컬 복사본 편집 후 Drive에 반영
3. **Colab**에서 `drive.mount` 후 해당 폴더 경로 사용  
   - `COLAB_QUICK_START.md`의 `cp /content/drive/MyDrive/...` 방식과 동일

---

## 5. 추천: Cursor + Colab 확장 + Git

| 단계 | 도구 | 할 일 |
|------|------|--------|
| 1 | Cursor | 프로젝트 열기, AI로 학습 코드 작성/수정 |
| 2 | Git | `commit` / `push`로 버전 관리 |
| 3 | Cursor | `.ipynb` 열고 **Kernel → Colab** 선택 |
| 4 | Colab | GPU 런타임에서 셀 실행, 학습 |
| 5 | Drive 등 | 체크포인트·결과 저장 |

- **에디터**: Cursor (VS Code 대체)
- **실행 환경**: Colab (GPU)
- **동기화**: Git 또는 Drive

---

## 6. 체크리스트

- [ ] Cursor (또는 VS Code) 설치
- [ ] **Google Colab** 확장 설치
- [ ] Google 계정 로그인
- [ ] Colab에서 GPU 런타임 선택 가능 확인
- [ ] 프로젝트 폴더를 Cursor로 열기
- [ ] `.ipynb`에서 **Select Kernel → Colab** 후 셀 실행 테스트
- [ ] (선택) Git repo 연결, Colab에서 `git clone` / `git pull` 사용

---

## 7. 문제 해결

### Colab 커널이 안 보일 때

- 확장 **Google Colab** 설치 여부 확인
- Cursor/VS Code 재시작
- `Ctrl+Shift+P` → "Developer: Reload Window"

### Colab 로그인 실패

- 팝업 차단 해제
- 시크릿 창에서 Google 로그인 상태 확인
- 확장 재설치 후 재시도

### Colab에서 GPU 안 잡힐 때

- Colab 메뉴: **런타임 → 런타임 유형 변경 → GPU**
- 무료 한도 소진 시 대기 또는 Pro 사용

### Cursor에서 확장이 VS Code와 다르게 동작할 때

- Cursor도 VS Code 호환이라 대부분 동일.  
- 특정 확장만 문제면 Cursor 포럼/이슈 검색.

---

## 8. 이 프로젝트에서의 사용 예

- **학습 스크립트**: `23_train_report_summary_lora.py`  
- **Colab용**: `COLAB_QUICK_START.md`, `colab_train_qwen7b.py` 등 참고  
- **Cursor**에서 `23_*.py` 또는 노트북 수정 → **Kernel = Colab** 선택 → Colab에서 학습 실행.

이 구조면 **Cursor AI로 학습 코드를 만들고, Colab에서 학습하는 메커니즘**을 그대로 사용할 수 있습니다.
