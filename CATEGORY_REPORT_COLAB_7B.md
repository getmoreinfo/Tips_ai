# 상품·리뷰 CSV → 카테고리 리포트 + Qwen 7B Colab 학습 가이드

상품 상세 데이터와 리뷰 데이터로 카테고리 리포트를 만들고, **Qwen 7B**로 Colab에서 LoRA 학습한 뒤 리포트 생성까지 하는 전체 흐름을 정리했습니다.

> **Colab에서 뭘 쳐야 하는지·오류 나면 뭘 해야 하는지:** **`COLAB_RUN_ORDER.md`** 에 순서별 복붙용 코드와 오류별 조치를 정리해 두었습니다. Colab 실행 시 해당 문서를 우선 참고하세요.

---

## 1. 전체 흐름 요약

```
[DB] + .env (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGSCHEMA, PGTABLE, PGREVIEWSTABLE)
       │
       ▼
  21_export_products_reviews_from_db.py   ← products_all.csv, reviews_all.csv 추출
       │
       ▼
[상품 CSV] + [리뷰 CSV]
       │
       ▼
  22_prepare_report_summary_sft.py   ← SFT 학습용 JSONL 생성
       │
       ▼
  training_report_summary_sft_*.jsonl
       │
       ▼
  23_train_report_summary_lora.py    ← Qwen 7B LoRA 학습 (Colab GPU)
       │
       ▼
  LoRA 어댑터 (adapter_model.safetensors 등)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
  25_generate_category_report_from_csv.py   24_generate_report_summary.py
  (템플릿 기반 리포트, CSV만 사용)            (학습된 7B + CSV로 요약 생성)
       │                                      │
       └──────────────────┬───────────────────┘
                          ▼
              카테고리 리포트 JSON (차트 + 요약)
```

- **21**: DB + `.env` → `products_all.csv`, `reviews_all.csv` 추출  
- **22**: CSV → SFT JSONL  
- **23**: JSONL → Colab에서 Qwen 7B LoRA 학습  
- **25**: CSV → **템플릿 기반** 카테고리 리포트 (모델 불필요)  
- **24**: CSV 또는 DB + **학습된 7B** → **모델 기반** 요약 생성  

---

## 2. CSV 형식 (필수 컬럼)

### 2-1. 상품 CSV (`products_all.csv` 등)

| 컬럼 | 설명 | 필수 |
|------|------|------|
| `id` | 상품 고유 ID (정수) | ✅ |
| `category_id` | 카테고리 ID (숫자) | ✅ |
| `category` | 카테고리 경로 (예: `유아용품 > 유모차`) | ✅ |
| `manufacturer` | 브랜드/제조사 | ✅ |
| `review_count` | 리뷰 수 | ✅ |
| `average_rating` | 평균 평점 | ✅ |
| `min_price` | 최저가 (선택, 가격/성장 분석용) | |
| `price_trend` | 가격 추이 (선택) | |
| `specs` | 스펙 JSON (선택, 세그먼트 추출) | |
| `review_tags` | 리뷰 태그 리스트 (선택) | |

### 2-2. 리뷰 CSV (`reviews_all.csv` 등)

| 컬럼 | 설명 | 필수 |
|------|------|------|
| `product_id` | 상품 `id`와 매칭 | ✅ |
| `review_date` | 리뷰 날짜 (연도별 성장률용) | ✅ 권장 |
| `first_seen_at` / `last_seen_at` / `created_at` / `updated_at` | `review_date` 없을 때 대체 | |

날짜 컬럼은 `review_date` → `first_seen_at` → `last_seen_at` → `created_at` → `updated_at` 순으로 사용합니다.

---

## 3. 단계별 실행 방법

### 3-0. Step 0: DB에서 CSV 추출 (21) — .env 사용

**로컬**에서 DB 접속이 가능할 때, `.env`에 DB 설정을 두고:

```bash
# .env 예시 (프로젝트 루트)
# PGHOST=localhost
# PGPORT=5432
# PGUSER=your_user
# PGPASSWORD=your_password
# PGSCHEMA=public1
# PGTABLE=products
# PGREVIEWSTABLE=reviews

pip install pandas psycopg2-binary python-dotenv   # 필요 시
python 21_export_products_reviews_from_db.py
```

- 기본: `min_products` 이상인 카테고리만 추출. `--all` 이면 상품/리뷰 전부.
- `--out_dir`로 저장 위치 지정 가능 (기본: 현재 디렉터리).
- 출력 `products_all.csv`, `reviews_all.csv`를 22 / 25 / 24 등에서 사용.

**이미 CSV가 있는 경우** 이 단계는 건너뛰면 됩니다.

---

### 3-1. Step 1: SFT 학습 데이터 생성 (22)

**로컬** 또는 **Colab**에서 CSV 준비 후:

```bash
python 22_prepare_report_summary_sft.py \
  --input_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --out_jsonl training_report_summary_sft.jsonl \
  --samples_per_category 50 \
  --min_products 10
```

- `--samples_per_category`: 카테고리당 샘플 수 (예: 50 → 4개 카테고리면 200줄 JSONL).
- 출력 `training_report_summary_sft.jsonl`을 23 학습에 사용합니다.

**이미 JSONL이 있는 경우** (예: `training_report_summary_sft_500.jsonl`) 이 단계는 건너뛰고, 해당 파일을 23에 넘기면 됩니다.

---

### 3-2. Step 2: Colab에서 Qwen 7B LoRA 학습 (23)

#### A. Colab에서 할 작업

1. **Google Drive**에 폴더 생성 (예: `tips_ai_colab`).
2. 아래 **필수 파일** 업로드:
   - `23_train_report_summary_lora.py`
   - `report_summary_lib.py`
   - `ai_report_bullets_lib.py`
   - `training_report_summary_sft_500.jsonl` (또는 22에서 만든 JSONL)
3. Colab 노트북에서 **런타임 → 런타임 유형 변경 → GPU (A100 또는 V100)** 선택.
4. `colab_train.ipynb` 열고 **Select Kernel → Colab** 선택 후, 셀 순서대로 실행.

#### B. Colab 셀 구성 요약

- **Drive 마운트** → `tips_ai_colab` 내용을 `/content/tips_ai` 등으로 복사.
- **패키지 설치**:  
  `pip install transformers peft datasets accelerate torch bitsandbytes`
- **학습 실행** (Qwen 7B):

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
  --grad_accum 8 \
  --save_steps 100 \
  --logging_steps 10
```

- `--out_dir`을 Drive 경로로 두면 학습 결과가 Drive에 저장됩니다.

#### C. 7B 학습 시 참고

- **GPU**: A100 40GB 권장. V100 16GB면 `--batch_size 1`, `--grad_accum 8` 유지.
- **OOM** 나면: `--max_length 1024`, `--lora_r 4`, `--lora_alpha 8` 로 줄여보세요.
- **학습 시간**: 500샘플 기준 A100에서 대략 1~2시간.

---

### 3-3. Step 3: 카테고리 리포트 생성

#### A. 템플릿만 사용 (모델 없음): 25

CSV만 있으면 됩니다. 학습 결과 불필요.

```bash
python 25_generate_category_report_from_csv.py \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "유모차" \
  --out_json report_stroller.json
```

- `--category_id` 로 특정 카테고리만 지정할 수도 있습니다.
- 출력 `report_stroller.json` 에 차트용 데이터 + 템플릿 기반 `marketOverviewSummary`, `growthSummary` 가 들어갑니다.

#### B. 학습된 7B 사용 (모델 기반): 24

CSV와 **LoRA 어댑터 디렉터리**가 필요합니다.

```bash
python 24_generate_report_summary.py \
  --model_dir /path/to/qwen2.5-7b-lora-report-summary \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "유모차"
```

- `--model_dir`: 23에서 `--out_dir`로 저장한 경로 (Colab이면 Drive 안 경로).
- `--category_id` / `--category_contains` 중 하나 지정.
- DB 없이 CSV만으로 모델 기반 요약을 생성할 때 사용합니다.

---

## 4. Colab 노트북 (`colab_train.ipynb`) 사용 순서

1. **Drive 마운트** → `tips_ai_colab` 복사.
2. **(선택)** CSV가 Drive에 있으면 **22** 실행해서 JSONL 생성.
3. **23** 실행 (위 7B 학습 명령).
4. **(선택)** Colab 또는 로컬에서 **25**로 템플릿 리포트 생성  
   또는 **24**로 학습된 7B + CSV 리포트 생성.

노트북에 22·25용 셀을 넣어 두었으면, 해당 셀을 순서대로 실행하면 됩니다.

---

## 5. 파일 정리

| 파일 | 용도 |
|------|------|
| `21_export_products_reviews_from_db.py` | DB + .env → `products_all.csv`, `reviews_all.csv` |
| `env.example` | .env 예시 (복사 후 `.env`로 사용) |
| `22_prepare_report_summary_sft.py` | CSV → SFT JSONL |
| `23_train_report_summary_lora.py` | JSONL → LoRA 학습 (7B 포함) |
| `24_generate_report_summary.py` | LoRA + CSV/DB → 모델 기반 요약 |
| `25_generate_category_report_from_csv.py` | CSV → 템플릿 리포트 |
| `report_summary_lib.py` | 요약/성장률 계산 공통 |
| `ai_report_bullets_lib.py` | 카테고리 메트릭·불릿 공통 |

---

## 6. 트러블슈팅

- **21 실행 시 `DB 연결 실패` / `.env 필요`**  
  → 프로젝트 루트에 `.env` 생성 후 `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD` 등 설정. `env.example` 참고.  
  → `pip install psycopg2-binary python-dotenv pandas` 후 재실행.

- **22 실행 시 `필수 컬럼 누락`**  
  → 상품 CSV에 `id`, `category_id`, `category`, `manufacturer`, `review_count`, `average_rating` 있는지 확인.  
  → 리뷰 CSV에 `product_id`, `review_date`(또는 대체 날짜 컬럼) 확인.

- **23 Colab에서 OOM**  
  → `--max_length 1024`, `--lora_r 4`, `--lora_alpha 8`, `--batch_size 1`, `--grad_accum 16` 등으로 조정.

- **학습 완료했는데 Drive `qwen2.5-7b-lora-report-summary` 폴더만 있고 비어 있음**  
  → Drive 직저장 시 동기화/경로 문제로 비는 경우 있음. **colab_train.ipynb** 기준으로:  
  - 4번: `--out_dir`을 **로컬** `/content/tips_ai/results/qwen2.5-7b-lora-report-summary` 로 두고 학습.  
  - 학습 끝난 뒤 **4-1 Drive 복사** 셀 실행 (`cp -r ... /content/drive/MyDrive/tips_ai_colab/results/`).  
  → 로컬 저장 후 복사하면 Drive에 정상 반영됨.

- **24에서 `model_dir` 오류**  
  → `--model_dir`이 4-1 복사한 **Drive 경로** (`.../tips_ai_colab/results/qwen2.5-7b-lora-report-summary`)와 일치하는지 확인.  
  → `adapter_config.json`, `adapter_model.safetensors` 등 LoRA 파일이 들어 있는지 확인.

- **25/24에서 `category_contains`에 해당 없음**  
  → `products_all.csv`의 `category` 값에 해당 문자열이 포함되는지 확인.  
  → 또는 `--category_id`로 직접 지정.

---

이 순서대로 하면 **상품·리뷰 CSV → 카테고리 리포트 생성**과 **Qwen 7B Colab 학습**을 한 흐름으로 진행할 수 있습니다.
