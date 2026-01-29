# Colab에 코드 직접 붙여넣기 — Drive/Git 없이

Drive·Git 없이 **Colab에서 파일 업로드 → 아래 셀만 순서대로 붙여넣어 실행**하는 방식입니다.

---

## 준비 (로컬에서 할 일)

Colab에 올릴 파일들을 **한 폴더에 모아두거나**, 압축해서 하나의 zip으로 만듭니다.

**필수 파일:**
- `products_all.csv`
- `reviews_all.csv`
- `22_prepare_report_summary_sft.py`
- `23_train_report_summary_lora.py`
- `24_generate_report_summary.py`
- `report_summary_lib.py`
- `ai_report_bullets_lib.py`

(선택: `25_generate_category_report_from_csv.py`, `26_evaluate_report_summary.py`, `db_category_loader.py` 등)

**방법 A:** 위 파일들을 zip으로 압축 → `ai-tr.zip`  
**방법 B:** Colab 왼쪽 **파일** 아이콘 → 업로드로 위 파일들을 하나씩 올림

---

## Colab에서 실행할 셀 (순서대로 복붙)

---

### 셀 1. 작업 폴더 만들기 + 파일 업로드

**방법 A: zip 업로드할 때**

```python
# zip 업로드 (실행 후 브라우저에서 ai-tr.zip 선택)
from google.colab import files
import zipfile
import os

uploaded = files.upload()  # ai-tr.zip 선택
for name in uploaded:
  with zipfile.ZipFile(name, 'r') as z:
    z.extractall('/content')
  print('압축 해제:', name)

%cd /content
# 압축 안 폴더 이름이 ai-tr가 아니면 아래에서 경로만 바꾸세요
if os.path.exists('ai-tr'):
  %cd /content/ai-tr
else:
  !ls /content
!pwd
!ls -la
```

**방법 B: zip 없이 파일만 업로드할 때**

```python
# CSV + 필요한 .py 한 번에 업로드 (실행 후 파일 여러 개 선택)
from google.colab import files
import os

os.makedirs('/content/ai-tr', exist_ok=True)
%cd /content/ai-tr
uploaded = files.upload()  # products_all.csv, reviews_all.csv, 22_...py, 23_...py, 24_...py, report_summary_lib.py, ai_report_bullets_lib.py 선택
print('업로드된 파일:', list(uploaded.keys()))
!ls -la
```

---

### 셀 2. 패키지 설치 + GPU 확인

```python
!pip install -q transformers peft datasets accelerate torch bitsandbytes pandas
import torch
print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

---

### 셀 3. SFT JSONL 생성 (22)

```python
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
```

---

### 셀 4. LoRA 학습 (23)

```python
%cd /content/ai-tr
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_reviews.jsonl \
  --out_dir /content/ai-tr/results/qwen2.5-7b-lora-report-summary-reviews \
  --epochs 5 --lr 1e-4 --max_length 2048 \
  --lora_r 8 --lora_alpha 16 --batch_size 1 --grad_accum 8 \
  --save_steps 100 --logging_steps 10
```

---

### 셀 5. 리포트 요약 생성 (24)

```python
%cd /content/ai-tr
!python 24_generate_report_summary.py \
  --model_dir /content/ai-tr/results/qwen2.5-7b-lora-report-summary-reviews \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "그림/동화/놀이책"
```

---

### 셀 6. (선택) 학습 결과 다운로드

```python
# 로컬로 받으려면: results 폴더를 zip으로 만들어 다운로드
%cd /content/ai-tr
!zip -r results.zip results/
files.download('results.zip')
```

---

## 요약

1. **셀 1:** 작업 폴더 만들기 + zip 또는 파일 업로드  
2. **셀 2:** 패키지 설치, GPU 확인  
3. **셀 3:** 22 실행 → SFT JSONL 생성  
4. **셀 4:** 23 실행 → LoRA 학습  
5. **셀 5:** 24 실행 → 리포트 요약 생성  
6. **셀 6:** (선택) results zip 다운로드  

Drive·Git 없이 **코드(셀)만 Colab에 붙여넣고**, CSV·스크립트는 **업로드**로만 준비하면 됩니다.
