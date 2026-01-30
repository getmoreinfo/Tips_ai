# Cursor 프로젝트를 노트북으로 옮기는 방법

## 방법 1: Git 사용 (가장 추천) ⭐

### 현재 컴퓨터에서
```powershell
# 변경사항 커밋
git add .
git commit -m "프로젝트 정리 완료 - 리포트 요약 파이프라인만 유지"

# 원격 저장소에 푸시 (GitHub/GitLab 등)
git push origin main
```

### 노트북에서
```powershell
# 프로젝트 클론
git clone <your-repo-url>
cd tips_ai

# 또는 기존 폴더가 있다면
cd tips_ai
git pull origin main
```

**장점:**
- ✅ 버전 관리
- ✅ 변경사항 추적
- ✅ 여러 기기에서 동기화
- ✅ 백업 역할

---

## 방법 2: USB/외장하드 복사 (간단함)

### 현재 컴퓨터에서
1. 프로젝트 폴더 전체 복사
   ```
   C:\Users\comso-1407\dev\tips_ai
   ```
2. USB/외장하드에 붙여넣기

### 노트북에서
1. USB/외장하드 연결
2. 프로젝트 폴더를 원하는 위치에 복사
   ```
   예: C:\Users\<your-name>\dev\tips_ai
   ```
3. Cursor에서 폴더 열기

**장점:**
- ✅ 인터넷 불필요
- ✅ 빠른 전송
- ✅ 간단함

---

## 방법 3: 클라우드 동기화 (Google Drive/OneDrive)

### Google Drive 사용
1. **현재 컴퓨터:**
   - Google Drive 폴더에 프로젝트 복사
   - 예: `C:\Users\<your-name>\Google Drive\tips_ai`

2. **노트북:**
   - Google Drive 동기화 대기
   - 또는 Google Drive 웹에서 다운로드
   - Cursor에서 폴더 열기

### OneDrive 사용
1. **현재 컴퓨터:**
   - OneDrive 폴더에 프로젝트 복사
   - 자동 동기화

2. **노트북:**
   - OneDrive 동기화 대기
   - Cursor에서 폴더 열기

**장점:**
- ✅ 자동 동기화
- ✅ 여러 기기에서 접근
- ✅ 백업 역할

---

## 방법 4: 네트워크 공유 (같은 네트워크)

### 현재 컴퓨터에서 공유 폴더 설정
1. 프로젝트 폴더 우클릭 → 속성 → 공유
2. 네트워크 공유 설정

### 노트북에서 접근
1. 네트워크 드라이브 연결
2. Cursor에서 폴더 열기

**장점:**
- ✅ 실시간 동기화 가능
- ✅ 빠른 전송

---

## 방법 5: 압축 파일 전송 (이메일/메신저)

### 현재 컴퓨터에서
```powershell
# ZIP 파일 생성
Compress-Archive -Path .\* -DestinationPath .\tips_ai.zip -Force
```

### 전송 방법
- 이메일 첨부 (크기 제한 확인)
- 메신저 전송 (카카오톡, 슬랙 등)
- 클라우드 업로드 후 링크 공유

### 노트북에서
1. ZIP 파일 다운로드
2. 압축 해제
3. Cursor에서 폴더 열기

---

## 추천: Git 사용 (방법 1)

가장 안전하고 관리하기 쉬운 방법입니다.

### 빠른 설정 (처음 한 번만)

#### 현재 컴퓨터에서
```powershell
# Git 저장소 초기화 (아직 안 했다면)
git init
git add .
git commit -m "Initial commit - 리포트 요약 파이프라인"

# GitHub에 새 저장소 생성 후
git remote add origin https://github.com/<your-username>/tips_ai.git
git push -u origin main
```

#### 노트북에서
```powershell
git clone https://github.com/<your-username>/tips_ai.git
cd tips_ai
```

---

## Cursor에서 프로젝트 열기

### 노트북에서
1. Cursor 실행
2. File → Open Folder
3. 프로젝트 폴더 선택
4. 완료!

---

## 주의사항

### 환경 변수 파일
- `.env` 파일은 Git에 포함되지 않음 (`.gitignore`에 있음)
- 노트북에서 새로 생성 필요:
  ```powershell
  python create_env_file.py
  # 또는 직접 .env 파일 생성
  ```

### 데이터베이스 연결
- `db_category_loader.py`가 DB 연결 필요
- 노트북에서도 DB 접근 가능한지 확인

### 패키지 설치
```powershell
pip install -r requirements_lora.txt
```

---

## 가장 빠른 방법 (USB 사용)

1. **현재 컴퓨터:**
   - 프로젝트 폴더 전체 복사 (Ctrl+C)
   - USB에 붙여넣기

2. **노트북:**
   - USB 연결
   - 프로젝트 폴더를 원하는 위치에 복사
   - Cursor에서 폴더 열기

**소요 시간: 약 5분**

---

원하는 방법을 알려주시면 더 자세히 안내해드리겠습니다!
