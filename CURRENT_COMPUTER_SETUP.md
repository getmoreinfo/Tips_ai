# 현재 컴퓨터 설정 단계별 가이드

## 진행 순서

### 1단계: Accelerate 설정 (단일 컴퓨터용)

먼저 단일 컴퓨터로 테스트하기 위해 설정:

**PowerShell에서 실행:**

```bash
python -m accelerate.commands.config
```

**또는 직접 설정 파일 생성:**

다음 명령으로 기본 설정 파일 생성:

```bash
python -c "from accelerate import Accelerator; a = Accelerator(); print('Accelerate initialized')"
```

**또는 간단하게:**

단일 컴퓨터 테스트이므로 Accelerate 없이도 실행 가능합니다:

```bash
python 02_finetune_local.py
```

---

### 2단계: 데이터 확인

```bash
dir training_data_10000.csv
```

파일이 있으면 다음 단계로, 없으면:

```bash
python 01_export_sample_10000.py
```

---

### 3단계: 단일 컴퓨터 테스트 실행

**방법 1: 일반 실행 (추천)**

```bash
python 02_finetune_local.py
```

**방법 2: Accelerate 사용**

```bash
python -m accelerate.commands.launch 02_finetune_local.py
```

---

### 4단계: 정상 작동 확인

다음이 정상적으로 실행되어야 합니다:

- ✅ GPU 확인 메시지 출력
- ✅ 데이터 로드 성공
- ✅ 모델 로드 성공
- ✅ 토크나이징 완료
- ✅ 학습 시작
- ✅ 에러 없이 실행

**학습은 시간이 걸리므로, 처음 몇 스텝이 정상적으로 실행되는지 확인하면 됩니다.**

학습을 중단하려면: `Ctrl + C`

---

### 5단계: Git 커밋 및 푸시

정상 작동 확인 후:

```bash
git add .
git commit -m "Complete setup: tested single computer training successfully"
git push origin main
```

---

### 6단계: 나머지 두 컴퓨터에서 Pull

각 컴퓨터에서:

```bash
git pull origin main
```

---

## 다음 단계: 멀티 노드 설정

현재 컴퓨터에서 테스트 성공 후:

1. **현재 컴퓨터:** Accelerate 재설정 (멀티 노드용, rank=0)
2. **컴퓨터 2:** Accelerate 설정 (rank=1)
3. **컴퓨터 3:** Accelerate 설정 (rank=2)
4. **3대 동시 실행:** `python -m accelerate.commands.launch 02_finetune_local.py`

---

## 빠른 체크리스트

- [ ] Accelerate 설치 확인
- [ ] 데이터 파일 확인 (`training_data_10000.csv`)
- [ ] 테스트 실행 (`python 02_finetune_local.py`)
- [ ] 정상 작동 확인
- [ ] Git 커밋 및 푸시
- [ ] 나머지 두 컴퓨터에서 pull
