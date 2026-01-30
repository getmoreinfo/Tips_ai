# Colab 전체 파이프라인 — 학습부터 검증까지

Colab **새 노트북**에서 아래 셀을 **순서대로** 실행하면, Drive 마운트 → Git pull → CSV 준비 → SFT 데이터 생성 → LoRA 학습 → 결과 저장 → 리포트 생성 → (선택) 검증까지 한 번에 진행할 수 있습니다.

---

## 파이프라인 개요

| 단계 | 스크립트 | 설명 | Colab |
|------|----------|------|-------|
| 0 | (로컬) 21 | DB → CSV 추출 | 로컬만 (.env 필요) |
| 1 | - | Drive 마운트, Git pull, CSV 복사 | ✅ |
| 2 | - | 패키지 설치, GPU 확인 | ✅ |
| 3 | 22 | SFT 학습 데이터(JSONL) 생성 | ✅ |
| 4 | 23 | LoRA 학습 (Qwen 7B) | ✅ |
| 5 | - | 학습 결과 → Drive 복사 | ✅ |
| 6 | 24 | 모델로 리포트 요약 생성 | ✅ |
| 7 | 25 | (선택) CSV만으로 카테고리 리포트 JSON 생성 | ✅ |
| 8 | 26 | (선택) 품질 평가 | DB 필요, Colab에서는 24 출력으로 수동 검증 |

- **21**은 로컬에서 `.env` 설정 후 한 번 실행해 `products_all.csv`, `reviews_all.csv`를 만들고, 이 파일들을 **Drive `tips_ai_colab`**에 넣어 두면 됩니다.
- **26**은 DB(`load_category_from_db`)를 쓰므로, Colab(CSV만 있는 환경)에서는 **24**로 여러 카테고리 돌려 본 뒤 출력 JSON을 보면서 검증하면 됩니다.

---

## Colab 새 노트북에서 실행할 코드 (순서대로)

아래 각 블록을 **새 셀**에 넣고, **위에서부터 순서대로** 실행하세요.  
런타임은 **런타임 → 런타임 유형 변경 → GPU** 로 설정한 뒤 진행하는 것을 권장합니다.

---

### 셀 1. Drive 마운트

```python
# Google Drive 마운트 (팝업에서 권한 허용)
from google.colab import drive
drive.mount('/content/drive')
```

**설명:** Colab에서 Drive를 `/content/drive`에 붙입니다. CSV·학습 결과 저장에 사용합니다.

---

### 셀 2. Git pull (이미 클론된 ai-tr 기준)

```python
# 저장소가 없으면 클론, 있으면 pull
import os
%cd /content
if not os.path.exists('ai-tr'):
  !git clone https://github.com/getmoreinfo/ai-tr.git ai-tr
%cd /content/ai-tr
!git pull origin main
!pwd
!ls -la
```

**설명:** 최신 코드를 가져옵니다. 비공개 저장소면 URL에 토큰을 넣어 클론하세요.

---

### 셀 3. CSV 준비 (Drive → 작업 디렉터리)

```python
# Drive의 tips_ai_colab에 있는 CSV를 /content/ai-tr로 복사
%cd /content/ai-tr
!cp /content/drive/MyDrive/tips_ai_colab/products_all.csv /content/ai-tr/ 2>/dev/null || echo "products_all.csv 없음"
!cp /content/drive/MyDrive/tips_ai_colab/reviews_all.csv /content/ai-tr/ 2>/dev/null || echo "reviews_all.csv 없음"
!ls -lh products_all.csv reviews_all.csv
```

**설명:** 학습·추론에 쓸 CSV를 작업 디렉터리로 가져옵니다. 없으면 21을 로컬에서 실행해 CSV를 만든 뒤 Drive에 올리세요.

---

### 셀 4. 패키지 설치 및 GPU 확인

```python
!pip install -q transformers peft datasets accelerate torch bitsandbytes pandas
import torch
print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

**설명:** LoRA 학습·추론에 필요한 라이브러리를 설치하고, GPU가 켜져 있는지 확인합니다.

---

### 셀 5. SFT 학습 데이터 생성 (22)

```python
# products_all.csv + reviews_all.csv → training_report_summary_sft_reviews.jsonl
%cd /content/ai-tr
!python 22_prepare_report_summary_sft.py \
  --input_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --out_jsonl training_report_summary_sft_reviews.jsonl \
  --samples_per_category 50 \
  --min_products 10 \
  --use_review_text \
  --review_top_keywords 10 \
  --review_pos_examples 3 \
  --review_neg_examples 2

!wc -l training_report_summary_sft_reviews.jsonl
!head -n 1 training_report_summary_sft_reviews.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); u=json.loads(d['messages'][1]['content']); print('reviewInsights:', 'reviewInsights' in u)"
```

**설명:** 카테고리별 시장/성장 요약 학습용 JSONL을 만들고, 리뷰 인사이트(`reviewInsights`)가 포함됐는지 확인합니다.

---

### 셀 6. LoRA 학습 (23)

```python
# Qwen 7B LoRA 학습 (GPU 필요, 수 분~수십 분 소요)
%cd /content/ai-tr
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_reviews.jsonl \
  --out_dir /content/ai-tr/results/qwen2.5-7b-lora-report-summary-reviews \
  --epochs 5 --lr 1e-4 --max_length 2048 \
  --lora_r 8 --lora_alpha 16 --batch_size 1 --grad_accum 8 \
  --save_steps 100 --logging_steps 10
```

**설명:** SFT JSONL로 Qwen2.5-7B-Instruct를 LoRA로 학습합니다. 결과는 `/content/ai-tr/results/...`에 저장됩니다.

---

### 셀 7. 학습 결과 Drive에 복사 (필수)

```python
# 런타임이 끊기면 로컬 결과가 사라지므로 반드시 Drive로 복사
!mkdir -p /content/drive/MyDrive/tips_ai_colab/results
!cp -r /content/ai-tr/results/qwen2.5-7b-lora-report-summary-reviews \
      /content/drive/MyDrive/tips_ai_colab/results/
!ls -lh /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary-reviews
```

**설명:** 학습된 LoRA 어댑터·메타데이터를 Drive에 백업합니다. 이후 24에서 이 경로를 `--model_dir`로 쓰면 됩니다.

---

### 셀 8. 리포트 요약 생성 (24) — 한 카테고리

```python
# 학습된 모델로 카테고리별 시장/성장 요약 JSON 생성
%cd /content/ai-tr
!python 24_generate_report_summary.py \
  --model_dir /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary-reviews \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "그림/동화/놀이책"
```

**설명:** 지정한 카테고리에 대해 `marketOverviewSummary`, `growthSummary`가 포함된 JSON을 출력합니다. 다른 카테고리는 `--category_contains` 값을 바꿔 실행하면 됩니다.

---

### 셀 9. (선택) 카테고리 리포트 전체 JSON 생성 (25)

```python
# 모델 없이 CSV만으로 차트+요약이 포함된 카테고리 리포트 JSON 생성
%cd /content/ai-tr
!python 25_generate_category_report_from_csv.py \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "그림/동화/놀이책" \
  --out report_category.json
!head -c 500 report_category.json
```

**설명:** 리포트 뷰어용 전체 JSON(차트 데이터 + 요약)을 만듭니다. 요약은 템플릿 기반이라 24와 다를 수 있습니다.

---

### 셀 10. (선택) 검증 — 여러 카테고리로 24 실행 후 확인

```python
# 여러 카테고리에 대해 24를 돌려 출력이 정상인지 확인 (26은 DB 필요로 Colab에서 대체)
%cd /content/ai-tr
for cat in "그림/동화/놀이책" "유모차"; do
  echo "=== $cat ==="
  !python 24_generate_report_summary.py \
    --model_dir /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary-reviews \
    --products_csv products_all.csv \
    --reviews_csv reviews_all.csv \
    --category_contains "$cat" 2>&1 | tail -20
done
```

**설명:** 26은 DB 기반이라 Colab(CSV만 있는 환경)에서는 사용하지 않고, 24를 여러 카테고리로 돌려 JSON 형식·내용을 눈으로 검증하는 용도입니다. (노트북에서는 `for` + `!python` 대신 카테고리별로 셀을 나눠 실행해도 됩니다.)

---

## 요약: Colab에서의 실행 순서

1. **Drive 마운트** (셀 1)  
2. **Git pull** (셀 2)  
3. **CSV 복사** (셀 3)  
4. **패키지 설치 + GPU** (셀 4)  
5. **22** SFT JSONL 생성 (셀 5)  
6. **23** LoRA 학습 (셀 6)  
7. **학습 결과 Drive 복사** (셀 7)  
8. **24** 리포트 요약 생성 (셀 8)  
9. **(선택) 25** 카테고리 리포트 JSON (셀 9)  
10. **(선택)** 여러 카테고리로 24 반복 실행 후 수동 검증 (셀 10)

---

## 오류 시 확인 사항

- **products_all.csv / reviews_all.csv 없음:** 로컬에서 21 실행 후 CSV를 Drive `tips_ai_colab`에 올리고, 셀 3 다시 실행.
- **리뷰 수=0, reviewInsights 없음:** 셀 3에서 CSV가 `/content/ai-tr`에 제대로 복사됐는지, 해당 CSV에 `review_text` 컬럼과 해당 카테고리 리뷰가 있는지 확인.
- **모델 로딩 실패:** 셀 7을 실행했는지, Drive 경로가 `tips_ai_colab/results/qwen2.5-7b-lora-report-summary-reviews`와 일치하는지 확인.
- **26 실행 불가:** 26은 DB 연결이 필요하므로 Colab(CSV만)에서는 24 출력으로 검증하면 됩니다.
