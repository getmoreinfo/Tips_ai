# 파이프라인 코드 상세 설명 (21 ~ 26번)

각 스크립트의 **역할**, **입·출력**, **핵심 로직**, **옵션**을 정리한 문서입니다.

---

## 1. 21_export_products_reviews_from_db.py

### 역할
- **PostgreSQL DB**에서 상품·리뷰 데이터를 읽어 **CSV**로 추출합니다.
- `.env`의 DB 설정으로 `danawa_crawlingdb`에 접속하고, `products_all.csv`, `reviews_all.csv`를 생성합니다.
- 이후 22번(SFT 데이터 생성), 24번(추론), 25번(리포트 생성)에서 이 CSV를 입력으로 사용합니다.

### 입력
- **환경 변수** (`.env`):  
  `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGSCHEMA`, `PGTABLE`, `PGREVIEWSTABLE`
- **DB**: `danawa_crawlingdb`, 스키마·테이블명은 환경 변수로 지정

### 출력
- `products_all.csv`: 상품 테이블 전체(또는 조건에 맞는 카테고리만)
- `reviews_all.csv`: 해당 상품들의 리뷰 테이블

### 핵심 로직
1. **`--all` 없을 때**:  
   카테고리당 상품 수가 `--min_products` 이상인 카테고리만 조회하고, 그 수를 `--max_categories`로 제한.  
   → 상품 수가 많은 카테고리부터 최대 N개만 추출.
2. **`--all` 있을 때**:  
   카테고리 필터 없이 상품·리뷰 전부 추출.
3. 상품 ID 목록을 구한 뒤, 해당 `product_id`를 가진 리뷰만 `reviews` 테이블에서 조회해 CSV로 저장.

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--all` | - | 카테고리 필터 없이 전부 추출 |
| `--min_products` | 10 | 카테고리당 최소 상품 수 (--all 미사용 시) |
| `--max_categories` | 10000 | 추출할 최대 카테고리 수 |
| `--out_dir` | . | CSV 저장 디렉터리 |
| `--products_csv` | products_all.csv | 상품 CSV 파일명 |
| `--reviews_csv` | reviews_all.csv | 리뷰 CSV 파일명 |

---

## 2. 22_prepare_report_summary_sft.py

### 역할
- **products_all.csv**와 **reviews_all.csv**를 읽어, **SFT(지도 미세조정)용 JSONL**을 만듭니다.
- 각 행은 `{"messages": [system, user, assistant]}` 형태이며,  
  **user**: 카테고리 메트릭·성장 데이터·(선택) 리뷰 인사이트 JSON,  
  **assistant**: `marketOverviewSummary`·`growthSummary` JSON입니다.
- 23번 LoRA 학습의 **학습 데이터**로 사용됩니다.

### 입력
- `products_all.csv`: 상품 데이터 (category_id, category, manufacturer, review_count, average_rating 등 필수)
- `reviews_all.csv`: 리뷰 데이터 (review_date, review_text 등)

### 출력
- `training_report_summary_sft.jsonl`: 한 줄에 하나의 대화형 예제(JSON)

### 핵심 로직
1. **카테고리 필터링**:  
   `min_products` 이상 상품을 가진 카테고리만 대상.  
   상품 수가 큰 카테고리부터 `max_categories`개까지 선택.
2. **카테고리당 여러 샘플**:  
   `samples_per_category`만큼, 서브샘플링(`subsample_ratio`)으로 상품 집합을 바꿔가며 같은 카테고리에 대해 여러 줄 생성.
3. **메트릭 계산**:  
   `ai_report_bullets_lib.aggregate_category_with_reviews`로 브랜드 점유·리뷰 수·평점 분포·HHI·세그먼트 등 집계.
4. **연도별 성장**:  
   `report_summary_lib.calculate_yearly_growth`로 리뷰 수 기반 가상 매출/판매량·성장률 계산 (proxy 전제).
5. **템플릿 요약**:  
   `build_market_overview_summary`, `build_growth_summary`로 **정답(assistant)** 문단 생성.
6. **리뷰 인사이트(선택)**:  
   `--use_review_text`(기본 True)일 때, 리뷰 텍스트에서 키워드 추출·긍정/부정 샘플을 뽑아 **user**의 `reviewInsights`에 넣음.
7. **SYSTEM_PROMPT**:  
   “커머스 시장 리포트 작성”, “리뷰 수 = 대리 지표(proxy)” 전제, 출력 형식(marketOverviewSummary, growthSummary) 등이 정의되어 있음.

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input_csv` | products_all.csv | 상품 CSV |
| `--reviews_csv` | reviews_all.csv | 리뷰 CSV |
| `--out_jsonl` | training_report_summary_sft.jsonl | 출력 JSONL |
| `--min_products` | 10 | 카테고리당 최소 상품 수 |
| `--max_categories` | 3000 | 최대 카테고리 수(상한) |
| `--samples_per_category` | 100 | 카테고리당 생성할 샘플 수 |
| `--subsample_ratio` | 0.9 | 서브샘플링 비율 |
| `--use_review_text` / `--no_use_review_text` | True | 리뷰 인사이트 포함 여부 |
| `--review_top_keywords` | 10 | topKeywords 개수 |
| `--review_pos_examples` | 3 | 긍정 리뷰 샘플 수(평점≥4) |
| `--review_neg_examples` | 2 | 부정 리뷰 샘플 수(평점≤2) |
| `--seed` | 42 | 재현용 랜덤 시드 |

---

## 3. 23_train_report_summary_lora.py

### 역할
- **22번에서 만든 JSONL(SFT)**을 사용해 **Qwen2.5 Instruct** 계열 모델에 **LoRA(PEFT)**를 적용해 학습합니다.
- 학습 목표: “user 입력(메트릭·성장·리뷰 인사이트 JSON)” → “assistant 출력(marketOverviewSummary, growthSummary JSON)”을 생성하도록 미세조정합니다.

### 입력
- `training_report_summary_sft.jsonl`: 22번 출력
- Hugging Face에서 **베이스 모델** 자동 다운로드 (기본: Qwen/Qwen2.5-7B-Instruct)

### 출력
- `--out_dir` 아래:  
  LoRA adapter 가중치, `adapter_config.json`, 저장된 토크나이저, **training_metadata.json**(base_model, 학습 옵션 기록)

### 핵심 로직
1. **예제 변환** (`_build_train_example`):  
   - `messages`에서 마지막 `assistant` 내용만 **정답(라벨)**으로 두고, 그 앞까지를 **프롬프트**로 붙여 하나의 시퀀스로 만듦.  
   - 라벨: 프롬프트 구간은 `-100`(손실 제외), 패딩도 `-100`.  
   - `max_length`까지 패딩·잘라내기.
2. **LoRA 설정**:  
   `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj` 등에 LoRA 적용.  
   `r`(rank), `lora_alpha`, dropout 등은 인자로 지정.
3. **학습**:  
   Hugging Face `Trainer` + `TrainingArguments`.  
   A100 40GB 기준으로 OOM을 피하기 위해 `max_length=2048`, `batch_size=2`, `grad_accum=16`, `gradient_checkpointing` 사용.  
   bf16/fp16는 GPU capability에 따라 자동 선택.
4. **메타데이터**:  
   학습에 쓴 `base_model`, `train_jsonl`, `max_length`, `epochs`, `lr`, `batch_size`, `grad_accum`, `lora_r`, `lora_alpha`를 `training_metadata.json`에 저장.  
   → 24번·26번에서 토크나이저/베이스 모델 경로 확인 시 사용.

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--base_model` | Qwen/Qwen2.5-7B-Instruct | 베이스 모델 |
| `--train_jsonl` | training_report_summary_sft.jsonl | 학습용 JSONL |
| `--out_dir` | results_report/qwen2.5-7b-lora-report-summary | 저장 디렉터리 |
| `--max_length` | 2048 | 최대 시퀀스 길이(40GB GPU 권장) |
| `--epochs` | 10 | 에폭 |
| `--lr` | 1e-4 | 학습률 |
| `--batch_size` | 2 | 디바이스당 배치 크기 |
| `--grad_accum` | 16 | 그래디언트 누적(effective batch = batch_size × grad_accum) |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--save_steps` | 500 | 체크포인트 저장 주기 |
| `--save_total_limit` | 3 | 유지할 체크포인트 개수 |
| `--warmup_ratio` | 0.05 | 워밍업 비율 |

---

## 4. 24_generate_report_summary.py

### 역할
- **학습된 LoRA 모델**을 사용해 **특정 카테고리**에 대한 커머스 시장 리포트 요약(marketOverviewSummary, growthSummary)을 **한 번** 생성합니다.
- 데이터 소스: **DB** 또는 **CSV**(products_all.csv, reviews_all.csv).
- **템플릿 전용** 모드(`--template_only`)도 지원해, 모델 없이 템플릿만으로 요약을 뽑을 수 있습니다.

### 입력
- **모델**: `--model_dir`(LoRA adapter가 있는 디렉터리).  
  `training_metadata.json`에서 `base_model`을 읽어 토크나이저·베이스 모델 로드.
- **카테고리 지정**: `--category_id` 또는 `--category_contains` 중 하나 필수.
- **CSV 사용 시**: `--products_csv`, `--reviews_csv` 지정.

### 출력
- stdout: JSON 형태  
  `categoryId`, `categoryName`, `marketOverviewSummary`, `growthSummary` (및 필요 시 정규화된 다른 필드).

### 핵심 로직
1. **카테고리 선택**:  
   CSV일 때는 `_pick_category_from_df`(25번과 동일 로직).  
   DB일 때는 `db_category_loader.load_category_from_db`.
2. **메트릭·성장**:  
   `aggregate_category_with_reviews`, `calculate_yearly_growth`로 22번·23번과 동일한 방식으로 user 입력용 JSON 구성.
3. **리뷰 인사이트**:  
   `_build_review_insights_for_inference`에서 리뷰 텍스트 컬럼(`review_text`, `review_content`, `content`, `text`, `body` 등 유연 인식)으로 키워드·긍정/부정 샘플 생성해 user 입력에 포함.
4. **모델 추론**:  
   system + user 메시지로 프롬프트 구성 → `model.generate`(temperature=0, do_sample=False) → 생성 구간만 잘라서 파싱.
5. **출력 정규화**:  
   - `_extract_json`: 생성 텍스트에서 첫 번째 `{ ... }` JSON 추출, `<|im_end|>` 제거.  
   - `_normalize_result_obj`: 중첩 JSON 문자열 풀기, `<|im_end|>`·중국어 구간 제거, `_clean_summary_text`로 문장 정리.  
   - JSON 파싱 실패 시 `_coerce_to_result_json`으로 마크다운/불릿 형태를 최소한의 JSON으로 복구.

### 주요 옵션
| 옵션 | 설명 |
|------|------|
| `--model_dir` | LoRA 모델 디렉터리 (template_only가 아니면 필수) |
| `--category_id` | 카테고리 ID (category_contains와 둘 중 하나) |
| `--category_contains` | 카테고리 경로에 포함된 문자열 (예: 유모차, 그림/동화/놀이책) |
| `--products_csv` | 상품 CSV (CSV 모드) |
| `--reviews_csv` | 리뷰 CSV (CSV 모드) |
| `--template_only` | 모델 없이 템플릿만으로 요약 생성 |
| `--max_length` | 최대 생성 토큰 수(기본 4096) |

---

## 5. 25_generate_category_report_from_csv.py

### 역할
- **products_all.csv**와 **reviews_all.csv**만으로, **카테고리 리포트 화면용 전체 JSON**을 한 번에 만듭니다.
- 24번은 “요약 텍스트(marketOverviewSummary, growthSummary)” 위주이고, 25번은 **차트 데이터(charts)** + **요약(summaries)** + **가정(assumptions)**까지 포함한 **프론트/백엔드 연동용** 구조입니다.
- **모델을 사용하지 않습니다.**  
  요약은 `report_summary_lib`의 `build_market_overview_summary`, `build_growth_summary`로 템플릿 기반 생성.

### 입력
- `products_all.csv`, `reviews_all.csv`
- 카테고리 지정: `--category_id` 또는 `--category_contains` 중 하나

### 출력
- stdout 또는 `--out_json` 파일.  
  예시 구조:
  - `categoryId`, `categoryName`
  - `charts`: `brandTop10Donut`(items), `growthLine`(unit, points)
  - `summaries`: `marketOverviewSummary`, `growthSummary`
  - `assumptions`: `proxy`, `unitsPerReview`, `priceRef`

### 핵심 로직
1. **카테고리 선택**:  
   `_pick_category`로 24번과 동일하게 category_id 또는 category_contains로 한 카테고리 선택.
2. **해당 카테고리 리뷰만 필터**:  
   상품 ID 집합에 맞는 `product_id` 리뷰만 사용.
3. **메트릭·성장**:  
   `aggregate_category_with_reviews`, `calculate_yearly_growth`(units_per_review, start_year, end_year 옵션 사용).
4. **요약**:  
   `build_market_overview_summary`, `build_growth_summary`로 템플릿 요약 생성.
5. **차트용 데이터**:  
   `top_brands` → donut items (name, value, share).  
   `growth_metrics` → growthLine points (year, reviewCount, estimatedUnits, estimatedRevenue, growthRate).
6. **assumptions**:  
   proxy=review_count, unitsPerReview, priceRef 기록 → 리포트 해석 시 “대리 지표” 전제를 명시.

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--products_csv` | products_all.csv | 상품 CSV |
| `--reviews_csv` | reviews_all.csv | 리뷰 CSV |
| `--category_id` | - | 카테고리 ID |
| `--category_contains` | - | 카테고리 경로 검색 문자열 |
| `--units_per_review` | 3.0 | 리뷰 1건당 가상 판매량 계수 |
| `--start_year` | 2020 | 성장 계산 시작 연도 |
| `--end_year` | 2025 | 성장 계산 종료 연도 |
| `--out_json` | - | 저장 경로(미지정 시 stdout) |

---

## 6. 26_evaluate_report_summary.py

### 역할
- **학습된 LoRA 모델**의 리포트 요약 **품질을 평가**합니다.
- **같은 카테고리**에 대해 **템플릿 기반 출력**과 **모델 출력**을 모두 만들고,  
  JSON 유효성·요약 길이·문장 수·대리 지표 언급·핵심 수치 포함·템플릿 유사도 등을 계산합니다.
- **데이터 소스는 DB**입니다. (`db_category_loader.load_category_from_db`)  
  Colab처럼 CSV만 있는 환경에서는 24번을 여러 카테고리로 돌려 수동으로 비교하는 방식으로 대체할 수 있습니다.

### 입력
- `--model_dir`: LoRA 모델 디렉터리 (필수)
- `--categories`: 평가할 카테고리 이름 목록 (`category_contains`로 검색, 기본: 유모차, 그림/동화/놀이책)
- DB 접속 정보(환경 변수 등)로 카테고리별 상품·리뷰 로드

### 출력
- stdout: 카테고리별 품질 요약(JSON 유효, 요약 길이, 대리 지표 언급, 템플릿 유사도 등)
- `--out_json`(기본: evaluation_report_summary.json):  
  카테고리별 `template`/`model` 출력, `quality` 지표, 전체 `summary` 통계

### 핵심 로직
1. **모델 로드**:  
   `training_metadata.json`에서 base_model 확인.  
   토크나이저는 adapter 디렉터리에 없으면 base_model에서 로드.  
   베이스 모델 + PeftModel.from_pretrained(adapter).
2. **카테고리별 평가** (`evaluate_category`):  
   - DB에서 해당 카테고리 상품·리뷰 로드.  
   - 메트릭·성장 계산 후 **템플릿 출력**(build_market_overview_summary, build_growth_summary) 생성.  
   - **모델 출력**: 24번과 동일한 system/user 프롬프트로 generate → JSON 추출(실패 시 최소 복구).
   - **품질 계산** (`evaluate_quality`):  
     - JSON 유효 여부, marketOverviewSummary/growthSummary 존재 여부  
     - 요약 길이(문자 수), 문장 수  
     - "대리 지표" 또는 "proxy" 언급 여부  
     - 핵심 수치(%·점수 등) 개수  
     - 템플릿과의 간단한 키워드 기반 유사도(Jaccard 비슷한 공통 단어 비율)
3. **전체 통계**:  
   JSON 형식 준수율, 대리 지표 언급율, 평균 요약 길이, 평균 핵심 수치 개수, 평균 템플릿 유사도 등을 요약해 출력·JSON에 기록.

### 주요 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model_dir` | (필수) | LoRA 모델 디렉터리 |
| `--categories` | 유모차, 그림/동화/놀이책 | 평가할 카테고리(공백으로 여러 개) |
| `--max_length` | 4096 | 최대 생성 토큰 수 |
| `--out_json` | evaluation_report_summary.json | 평가 결과 저장 경로 |

---

## 파이프라인 흐름 요약

```
[DB] ── 21번 ──► products_all.csv, reviews_all.csv
                      │
                      ├──► 22번 ──► training_report_summary_sft.jsonl
                      │                  │
                      │                  └──► 23번 ──► LoRA adapter (results_report/...)
                      │                                        │
                      │                                        ├──► 24번 (단일 카테고리 요약 생성)
                      │                                        └──► 26번 (DB 기반 품질 평가)
                      │
                      └──► 25번 ──► 카테고리 리포트 전체 JSON (차트+요약+assumptions)
```

- **21**: 데이터 준비 (DB → CSV)  
- **22**: SFT 데이터 준비 (CSV → JSONL)  
- **23**: LoRA 학습 (JSONL → adapter)  
- **24**: 단일 카테고리 요약 생성 (CSV 또는 DB + adapter)  
- **25**: CSV만으로 리포트 전체 JSON 생성 (모델 미사용)  
- **26**: DB + adapter로 여러 카테고리 품질 평가  

*최종 업데이트: 2026-01-29*
