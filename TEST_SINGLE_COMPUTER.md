# 단일 컴퓨터 테스트 가이드

## 현재 컴퓨터에서 먼저 테스트

### 1단계: Accelerate 설정 (단일 컴퓨터용)

먼저 단일 컴퓨터로 테스트하기 위한 설정:

```bash
accelerate config
```

**질문에 답변:**

```
In which compute environment are you running?
> This machine

Which type of machine are you using?
> single-GPU  (또는 multi-GPU인데 GPU가 1개면 single-GPU)

How many GPUs are available on this machine?
> 1

Which GPU(s) should be used?
> 0

Do you want to use Mixed Precision?
> fp16  (GPU가 지원하는 경우)
```

**또는 간단하게:**

```
accelerate config --config_file default_config.yaml
```

그리고 다음 내용으로 생성:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
num_processes: 1
use_cpu: false
```

---

### 2단계: 설정 확인

```bash
accelerate env
```

---

### 3단계: 데이터 확인

```bash
dir training_data_10000.csv
```

파일이 없으면:

```bash
python 01_export_sample_10000.py
```

---

### 4단계: 단일 컴퓨터 테스트 실행

```bash
python 02_finetune_local.py
```

또는 Accelerate 사용:

```bash
accelerate launch 02_finetune_local.py
```

---

### 5단계: 정상 작동 확인

다음이 정상적으로 실행되어야 합니다:

- ✅ GPU 확인 메시지
- ✅ 데이터 로드 성공
- ✅ 모델 로드 성공
- ✅ 학습 시작
- ✅ 에러 없이 실행

---

### 6단계: Git 커밋 및 푸시

정상 작동 확인 후:

```bash
git add .
git commit -m "Complete distributed training setup - tested on single computer"
git push origin main
```

---

### 7단계: 나머지 두 컴퓨터에서 Pull

각 컴퓨터에서:

```bash
git pull origin main
```

그 후 각 컴퓨터에서 Accelerate 설정만 다시 하면 됩니다!

---

## 다음 단계: 멀티 노드 설정

현재 컴퓨터에서 테스트 성공 후:

1. **현재 컴퓨터에서:** Accelerate 재설정 (멀티 노드용)
2. **컴퓨터 2, 3에서:** Accelerate 설정 (각각 rank 1, 2)
3. **3대 동시 실행:** `accelerate launch 02_finetune_local.py`
