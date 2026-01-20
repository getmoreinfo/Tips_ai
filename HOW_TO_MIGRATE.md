# 다른 컴퓨터로 옮기기 - 빠른 가이드

## ✅ 네, 가능합니다!

Cursor를 설치하고 파일을 옮길 수 있습니다. 방법은 다음과 같습니다:

---

## 방법 1: Git 사용 (가장 쉬움! ⭐)

### 현재 컴퓨터에서:

```bash
# 1. 새로 만든 파일들을 Git에 추가
git add .
git commit -m "Add distributed training setup and migration files"

# 2. GitHub에 푸시 (이미 원격 저장소가 있다면)
git push origin main
```

### 다른 컴퓨터에서:

1. **Cursor 설치**
2. **프로젝트 클론:**
   ```bash
   git clone <your-repository-url>
   cd tips_ai
   ```
3. **Cursor에서 프로젝트 폴더 열기**

끝! 🎉

---

## 방법 2: 폴더 직접 복사

### 자동 백업 스크립트 사용:

```powershell
# 현재 컴퓨터에서
.\copy_project_to_backup.ps1 -IncludeCursorHistory
```

이 스크립트가:
- ✅ 프로젝트 파일 모두 복사
- ✅ Cursor 설정 복사
- ✅ 대화 히스토리 포함 (선택사항)

### 다른 컴퓨터에서:

1. **Cursor 설치**
2. **백업 폴더를 원하는 위치에 복사**
3. **Cursor에서 프로젝트 폴더 열기**

---

## 방법 3: Cursor 설정만 따로 복사

### Cursor 대화 히스토리 위치:

```
C:\Users\<사용자명>\AppData\Roaming\Cursor\User\workspaceStorage\
```

### 수동 복사:

1. **현재 컴퓨터에서:**
   - 위 폴더를 USB로 복사

2. **다른 컴퓨터에서:**
   - Cursor 설치 후 같은 위치에 붙여넣기

---

## 추천 방법: Git + 문서화

### 1. 현재 파일들을 Git에 추가

```bash
git add .
git commit -m "Complete project setup with distributed training"
git push  # GitHub 등에 업로드
```

### 2. 다른 컴퓨터에서

```bash
git clone <repository-url>
cd tips_ai
```

### 3. Cursor 설치 후 프로젝트 열기

Cursor가 설치되어 있으면:
- File → Open Folder
- 클론한 `tips_ai` 폴더 선택

---

## 중요 참고사항

### ✅ 포함되는 것:
- 모든 Python 파일
- 데이터 파일 (training_data_10000.csv)
- 설정 파일
- 문서 파일

### ⚠️ 주의사항:
- `.env` 파일: 민감 정보이므로 Git에 올리지 말 것
- Cursor 대화 히스토리: 로컬에만 저장되므로 수동 복사 필요
- 비밀번호/API 키: 안전하게 전달

---

## 체크리스트

현재 컴퓨터:
- [ ] Git에 파일 추가 (`git add .`)
- [ ] 커밋 및 푸시 (`git commit`, `git push`)
- [ ] 또는 백업 스크립트 실행

다른 컴퓨터:
- [ ] Cursor 설치
- [ ] Git 클론 또는 폴더 복사
- [ ] Cursor에서 프로젝트 폴더 열기
- [ ] `.env` 파일 생성 (`create_env_file.py`)
- [ ] 패키지 설치 (`setup_other_computer.ps1`)

---

**결론:** Cursor를 설치하고 파일을 옮기면 대화 내용까지 이어서 사용할 수 있습니다!
