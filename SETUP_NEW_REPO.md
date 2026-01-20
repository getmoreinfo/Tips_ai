# 새 개인 Git 저장소 설정 가이드

## 현재 상태
- 기존 원격 저장소: `https://gitlab.com/gouse2042/tips_ai.git`
- 이것이 본인의 GitLab 계정이라면 이미 본인 레포지토리입니다!

## 옵션 1: 기존 저장소 그대로 사용 (가장 간단)

이미 본인의 레포지토리라면 그대로 사용하면 됩니다:

```bash
git add .
git commit -m "Add distributed training and migration files"
git push origin main
```

## 옵션 2: 새로운 GitHub 저장소 만들기

### 1. GitHub에서 새 저장소 생성

1. [GitHub](https://github.com) 로그인
2. 우측 상단 **+** → **New repository** 클릭
3. 저장소 이름 입력 (예: `tips_ai`)
4. **Private** 선택 (본인만 접근)
5. **Create repository** 클릭

### 2. 현재 프로젝트에 새 원격 저장소 연결

```bash
# 기존 원격 저장소 제거
git remote remove origin

# 새 GitHub 저장소 추가
git remote add origin https://github.com/<your-username>/tips_ai.git

# 확인
git remote -v

# 파일 추가 및 커밋
git add .
git commit -m "Initial commit: AI training project"

# 새 저장소에 푸시
git push -u origin main
```

## 옵션 3: 새로운 GitLab 저장소 만들기

### 1. GitLab에서 새 저장소 생성

1. [GitLab](https://gitlab.com) 로그인
2. 우측 상단 **+** → **New project/repository** 클릭
3. **Create blank project** 선택
4. Project name 입력 (예: `tips_ai_personal`)
5. **Visibility Level**: Private 선택
6. **Initialize repository with a README** 체크 해제
7. **Create project** 클릭

### 2. 새 GitLab 저장소에 연결

```bash
# 기존 원격 저장소 제거
git remote remove origin

# 새 GitLab 저장소 추가
git remote add origin https://gitlab.com/<your-username>/tips_ai_personal.git

# 확인
git remote -v

# 파일 추가 및 커밋
git add .
git commit -m "Initial commit: AI training project"

# 새 저장소에 푸시
git push -u origin main
```

## 옵션 4: 로컬 저장소만 사용 (원격 저장소 없이)

원격 저장소 없이 로컬에서만 관리:

```bash
# 기존 원격 저장소 제거
git remote remove origin

# 로컬에서만 작업
git add .
git commit -m "Add all project files"

# 다른 컴퓨터로 옮길 때는 폴더를 직접 복사
```

---

## 권장 방법: GitHub Private 저장소

### 빠른 설정:

```bash
# 1. 기존 원격 저장소 제거
git remote remove origin

# 2. GitHub에서 새 Private 저장소 생성 (웹에서)

# 3. 새 저장소 URL로 연결 (본인의 GitHub 계정 이름으로 변경)
git remote add origin https://github.com/<your-username>/tips_ai.git

# 4. 파일 추가
git add .

# 5. 커밋
git commit -m "Complete AI training project setup"

# 6. 푸시
git push -u origin main
```

---

## .gitignore 확인

민감한 정보는 저장소에 올리지 않도록 `.gitignore` 확인:

```bash
# .env 파일은 Git에 올라가지 않음 (이미 .gitignore에 포함됨)
# training_data_10000.csv는 데이터 파일이므로 필요시 제외 가능
```

필요시 `.gitignore`에 추가:
```
training_data_10000.csv  # 큰 데이터 파일은 제외할 수 있음
```

---

## 다음 단계

1. ✅ 새 저장소 설정 완료
2. ✅ 파일 푸시 완료
3. ✅ 다른 컴퓨터에서 클론: `git clone <new-repository-url>`
4. ✅ Cursor 설치 후 프로젝트 열기
