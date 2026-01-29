# Google Colab에서 Qwen 7B 학습 설정 가이드

## 1. 필요한 파일 목록

### 필수 파일 (반드시 필요)
1. `training_report_summary_sft_500.jsonl` - 학습 데이터 (500 examples)
2. `23_train_report_summary_lora.py` - 학습 스크립트
3. `report_summary_lib.py` - 리포트 요약 라이브러리
4. `ai_report_bullets_lib.py` - 카테고리 메트릭 계산 라이브러리
5. `requirements_lora.txt` - 패키지 의존성

### 선택 파일 (평가용)
6. `26_evaluate_report_summary.py` - 모델 평가 스크립트
7. `db_category_loader.py` - DB 로더 (평가 시 필요)

---

## 2. Google Colab 설정 단계

### Step 1: 새 노트북 생성
1. https://colab.research.google.com 접속
2. "새 노트북" 클릭
3. 런타임 → 런타임 유형 변경
   - 하드웨어 가속기: **GPU** 선택
   - GPU 유형: **A100** 또는 **V100** (가능하면 A100)

### Step 2: Google Drive 마운트
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: 프로젝트 폴더 생성 및 파일 업로드
```python
# 작업 디렉토리 생성
!mkdir -p /content/tips_ai
%cd /content/tips_ai

# Google Drive에서 파일 복사 (업로드 후)
# 또는 직접 파일 업로드
```

---

## 3. 파일 업로드 방법

### 방법 1: Google Drive에 업로드 후 복사 (추천)

#### 3-1. 로컬에서 Google Drive에 업로드
1. Google Drive 접속 (https://drive.google.com)
2. 새 폴더 생성: `tips_ai_colab`
3. 다음 파일들을 업로드:
   - `training_report_summary_sft_500.jsonl`
   - `23_train_report_summary_lora.py`
   - `report_summary_lib.py`
   - `ai_report_bullets_lib.py`
   - `requirements_lora.txt`

#### 3-2. Colab에서 Drive 마운트 및 복사
```python
from google.colab import drive
drive.mount('/content/drive')

# 파일 복사
!cp -r /content/drive/MyDrive/tips_ai_colab/* /content/tips_ai/
%cd /content/tips_ai
```

### 방법 2: 직접 업로드 (작은 파일용)
```python
from google.colab import files
uploaded = files.upload()  # 브라우저에서 파일 선택
```

---

## 4. Colab 노트북 전체 코드

```python
# ============================================
# Google Colab Qwen 7B 학습 설정
# ============================================

# 1. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 2. 작업 디렉토리 설정
import os
os.makedirs('/content/tips_ai', exist_ok=True)
%cd /content/tips_ai

# 3. 파일 복사 (Google Drive에서)
# 방법 A: Drive에 업로드한 경우
!cp -r /content/drive/MyDrive/tips_ai_colab/* /content/tips_ai/

# 방법 B: 직접 업로드
# from google.colab import files
# uploaded = files.upload()

# 4. 필수 패키지 설치
!pip install -q transformers peft datasets accelerate torch bitsandbytes

# 5. Hugging Face 토큰 설정 (필요시)
import os
# os.environ['HF_TOKEN'] = 'your_huggingface_token_here'  # 필요시 설정

# 6. GPU 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 7. 학습 데이터 확인
!head -n 1 training_report_summary_sft_500.jsonl

# 8. 학습 실행
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_500.jsonl \
  --out_dir /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary \
  --epochs 5 \
  --lr 1e-4 \
  --max_length 2048 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 1 \
  --grad_accum 8

# 9. 학습 완료 후 결과 확인
!ls -lh /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary/
```

---

## 5. Qwen 7B 학습 설정 변경사항

### 23_train_report_summary_lora.py 수정 필요사항

기본 모델이 `Qwen/Qwen2.5-3B-Instruct`로 되어있으므로, 
명령어에서 `--base_model Qwen/Qwen2.5-7B-Instruct`로 지정하면 됩니다.

### 메모리 최적화 설정 (7B 모델용)

```python
# 8GB GPU 환경에서는 bitsandbytes 8bit 로딩 필요
# Colab A100/V100은 충분한 메모리이므로 기본 설정으로도 가능
```

---

## 6. 학습 실행 명령어 (최종)

```bash
python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_500.jsonl \
  --out_dir /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary \
  --epochs 5 \
  --lr 1e-4 \
  --max_length 2048 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 1 \
  --grad_accum 8
```

---

## 7. 예상 학습 시간 및 비용

### A100 GPU 기준
- 예상 시간: 1-2시간 (500 examples, 5 epochs)
- Compute Units 소비: 약 10-20 units
- 월간 할당 대비: 약 10-20%

### V100 GPU 기준
- 예상 시간: 2-3시간
- Compute Units 소비: 약 12-18 units
- 월간 할당 대비: 약 12-18%

---

## 8. 학습 완료 후

### 결과 파일 다운로드
```python
# Google Drive에 자동 저장됨
# /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary/

# 또는 로컬로 다운로드
from google.colab import files
files.download('/content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary/adapter_model.safetensors')
```

---

## 9. 문제 해결

### GPU 할당 안 될 때
- 런타임 → 런타임 유형 변경 → GPU 재선택
- 또는 런타임 → 런타임 연결 해제 후 재연결

### 메모리 부족 시
- `--batch_size 1` 유지
- `--grad_accum` 값 증가
- `--max_length` 값 감소 (2048 → 1024)

### 학습 중단 시
- Colab은 세션이 끊어져도 체크포인트 저장 가능
- `--save_steps` 옵션으로 주기적 저장

---

## 10. 체크리스트

- [ ] Google Colab Pro 구독 완료
- [ ] 필수 파일 5개 준비 완료
- [ ] Google Drive에 파일 업로드 완료
- [ ] Colab 노트북에서 GPU 할당 확인
- [ ] 학습 데이터 파일 확인 (500 examples)
- [ ] 학습 스크립트 실행 준비 완료

준비되면 위의 코드를 Colab 노트북에 복사해서 실행하세요!
