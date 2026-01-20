# 다른 컴퓨터로 파일 및 대화 내용 옮기기

## 방법 1: Git 사용 (가장 추천! ⭐)

### 장점
- ✅ 파일 자동 동기화
- ✅ 버전 관리
- ✅ 여러 컴퓨터에서 동일한 상태 유지
- ✅ 대화 내용도 문서로 저장 가능

### 단계별 가이드

#### 1. Git 저장소 초기화 (현재 컴퓨터)

```bash
git init
git add .
git commit -m "Initial commit: AI training project"
```

#### 2. GitHub/GitLab에 업로드 (선택사항)

```bash
# GitHub에서 새 저장소 생성 후
git remote add origin https://github.com/<username>/tips_ai.git
git push -u origin main
```

#### 3. 다른 컴퓨터에서 클론

```bash
git clone https://github.com/<username>/tips_ai.git
cd tips_ai
```

#### 4. 대화 내용도 문서로 저장 (권장)

현재 대화 내용을 README나 별도 파일로 저장:
- `CONVERSATION_HISTORY.md` 파일 생성
- 중요한 대화 내용 정리

---

## 방법 2: 프로젝트 폴더 직접 복사

### 복사할 폴더 위치
```
C:\Users\comso-1407\dev\tips_ai
```

### 복사 방법

#### 방법 A: USB/외장 하드
1. `C:\Users\comso-1407\dev\tips_ai` 폴더 전체 복사
2. 다른 컴퓨터의 같은 위치에 붙여넣기

#### 방법 B: 네트워크 공유
1. 현재 컴퓨터에서 폴더 공유 설정
2. 다른 컴퓨터에서 네트워크 드라이브로 접근
3. 폴더 복사

#### 방법 C: 클라우드 (OneDrive, Google Drive 등)
1. 폴더를 클라우드에 업로드
2. 다른 컴퓨터에서 다운로드

---

## 방법 3: Cursor 대화 히스토리 복사 (선택사항)

### Cursor 설정 및 히스토리 위치

#### Windows:
```
%APPDATA%\Cursor
C:\Users\<사용자명>\AppData\Roaming\Cursor
```

### 복사할 폴더들

1. **설정 파일:**
   ```
   %APPDATA%\Cursor\User\settings.json
   ```

2. **채팅 히스토리** (만약 저장되어 있다면):
   ```
   %APPDATA%\Cursor\storage\
   %APPDATA%\Cursor\User\workspaceStorage\
   ```

### 대화 히스토리 복사 방법

#### 자동 복사 스크립트 사용:

```powershell
# copy_cursor_history.ps1 실행
```

#### 수동 복사:

1. **현재 컴퓨터에서:**
   ```powershell
   # Cursor 설정 폴더 위치 확인
   cd $env:APPDATA\Cursor
   # 이 폴더를 USB나 네트워크로 복사
   ```

2. **다른 컴퓨터에서:**
   ```powershell
   # 같은 위치에 붙여넣기
   C:\Users\<새사용자명>\AppData\Roaming\Cursor
   ```

### 주의사항
- ⚠️ Cursor의 대화 히스토리는 로컬에만 저장됩니다
- ⚠️ 복사해도 완전히 동일하지 않을 수 있습니다
- ✅ **중요한 대화는 문서로 저장하는 것을 권장합니다**

---

## 추천 방법: Git + 문서화

### 1. Git 저장소 만들기

```bash
# 현재 프로젝트 폴더에서
git init
git add .
git commit -m "Initial commit"
```

### 2. 대화 내용 정리 (중요한 것만)

프로젝트에 `PROJECT_NOTES.md` 파일 만들기:
```markdown
# 프로젝트 노트

## 주요 작업 내용
- PostgreSQL 연결 설정
- 10,000개 샘플 추출
- 파인튜닝 설정
- 멀티 노드 분산 학습 설정

## 주요 결정 사항
- HuggingFace Accelerate 사용
- FP16 자동 설정
- dataloader_num_workers=0 (Windows 안정성)
```

### 3. 다른 컴퓨터에서 Git 클론

```bash
git clone <repository-url>
# 또는 폴더 직접 복사
```

---

## 빠른 복사 스크립트

### 현재 컴퓨터에서 실행:

```powershell
# copy_project_to_backup.ps1 실행
```

이 스크립트가:
1. 프로젝트 폴더를 압축
2. USB나 특정 위치로 복사
3. Cursor 설정도 함께 복사 (선택사항)

---

## 요약

### 가장 간단한 방법:
1. ✅ **프로젝트 폴더 전체 복사** (USB/네트워크/클라우드)
2. ✅ **다른 컴퓨터에 붙여넣기**
3. ✅ **Cursor 설치 후 프로젝트 폴더 열기**

### 가장 좋은 방법:
1. ✅ **Git 사용** (버전 관리 + 동기화)
2. ✅ **중요한 대화 내용 문서화**
3. ✅ **GitHub/GitLab에 업로드** (백업 + 공유)

---

## 체크리스트

다른 컴퓨터로 옮기기 전:
- [ ] 모든 `.py` 파일 확인
- [ ] `training_data_10000.csv` 포함 확인
- [ ] `.env` 파일 확인 (비밀번호는 따로 전달)
- [ ] 중요한 대화 내용 문서화
- [ ] Git 저장소 생성 (선택사항)

다른 컴퓨터에서:
- [ ] 프로젝트 폴더 복사 완료
- [ ] Python 설치 확인
- [ ] 패키지 설치 (`setup_other_computer.ps1`)
- [ ] Cursor 설치 (선택사항)
- [ ] 프로젝트 폴더 열기

---

**중요:** Cursor의 대화 히스토리는 클라우드 동기화가 되지 않습니다. 
**중요한 내용은 별도 문서로 저장하는 것을 강력히 권장합니다!**
