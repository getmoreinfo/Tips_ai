# Google Colab Qwen 7B 학습 빠른 시작 가이드

## 필수 파일 목록 (5개만 필요!)

1. ✅ `training_report_summary_sft_500.jsonl` - 학습 데이터
2. ✅ `23_train_report_summary_lora.py` - 학습 스크립트  
3. ✅ `report_summary_lib.py` - 리포트 요약 라이브러리
4. ✅ `ai_report_bullets_lib.py` - 카테고리 메트릭 라이브러리
5. ✅ `requirements_lora.txt` - 패키지 목록 (선택사항)

---

## 빠른 시작 (3단계)

### Step 1: Google Drive에 파일 업로드

1. Google Drive 접속: https://drive.google.com
2. 새 폴더 생성: `tips_ai_colab`
3. 위 5개 파일을 이 폴더에 업로드

### Step 2: Colab 노트북 생성

1. https://colab.research.google.com 접속
2. "새 노트북" 클릭
3. 런타임 → 런타임 유형 변경 → **GPU (A100 또는 V100)** 선택

### Step 3: 아래 코드를 Colab에 복사 & 실행

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
!cp -r /content/drive/MyDrive/tips_ai_colab/* /content/tips_ai/

# 4. 파일 확인
!ls -lh

# 5. 필수 패키지 설치
!pip install -q transformers peft datasets accelerate torch bitsandbytes

# 6. GPU 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 7. 학습 데이터 확인
!head -n 1 training_report_summary_sft_500.jsonl

# 8. 학습 실행 (Qwen 7B)
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
  --grad_accum 8 \
  --save_steps 100 \
  --logging_steps 10

# 9. 학습 완료 확인
print("\n✅ 학습 완료!")
print("결과 위치: /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary/")
```

---

## 예상 소요 시간 및 비용

- **A100 GPU**: 1-2시간, 약 10-20 compute units
- **V100 GPU**: 2-3시간, 약 12-18 compute units
- **월간 할당 대비**: 약 10-20%

---

## 주의사항

1. **Google Drive 마운트**: 첫 실행 시 권한 요청이 나옵니다. 승인하세요.
2. **GPU 할당**: A100이 안 되면 V100으로 시도하세요.
3. **세션 시간**: Colab Pro는 세션이 길어도 괜찮지만, 중간에 끊기면 체크포인트에서 재시작 가능합니다.
4. **결과 저장**: Google Drive에 자동 저장되므로 안전합니다.

---

## 문제 해결

### GPU가 할당 안 될 때
- 런타임 → 런타임 연결 해제 → 재연결
- 또는 런타임 → 런타임 유형 변경 → GPU 재선택

### 메모리 부족 시
- `--batch_size 1` 유지
- `--grad_accum` 값을 16으로 증가
- `--max_length`를 1024로 감소

### 파일을 찾을 수 없을 때
- Google Drive 경로 확인: `/content/drive/MyDrive/tips_ai_colab/`
- 파일 이름 확인: 대소문자 구분 주의

---

준비되면 위 코드를 복사해서 Colab에 붙여넣고 실행하세요! 🚀
