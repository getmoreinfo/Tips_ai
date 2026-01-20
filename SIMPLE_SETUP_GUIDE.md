# 다른 컴퓨터에서 간단 설정 가이드

## Git Pull 완료 후

### 빠른 설정 (3단계)

#### 1. 패키지 설치
```bash
.\setup_other_computer.ps1
```

#### 2. .env 파일 생성
```bash
python create_env_file.py
```
그 후 `.env` 파일을 열어서 실제 DB 정보로 수정

#### 3. 테스트 실행
```bash
python test_training_quick.py
```

---

## Cursor AI에게 할 말

**가장 간단한 버전:**

```
이 컴퓨터는 AI 학습 프로젝트의 두 번째 컴퓨터입니다.
Git pull 완료했고, 이제 환경 설정을 시작하려고 합니다.
setup_other_computer.ps1를 실행하는 방법을 알려주세요.
```

**자세한 버전:**

```
안녕하세요. 

이 컴퓨터는 3대의 분산 학습 컴퓨터 중 두 번째 컴퓨터입니다.
Git pull을 완료했고, 이제 환경 설정을 진행하려고 합니다.

필요한 작업:
1. Python 패키지 설치
2. .env 파일 생성 및 설정
3. 학습 테스트
4. Accelerate 멀티 노드 설정

각 단계를 순서대로 알려주세요.
```

---

**이렇게 말하면 AI가 모든 것을 도와드립니다!** ✅
