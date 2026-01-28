# 실행 순서 가이드

## 0. 사전 확인
- `.env` 에 **PGHOST, PGPORT, PGUSER, PGPASSWORD, PGSCHEMA, PGTABLE, PGREVIEWSTABLE** 이 올바르게 설정되어 있어야 합니다.

---

## 경로 A: DB에서 바로 불릿 생성 (CSV 없음)

카테고리 불릿만 필요할 때, CSV 추출 없이 DB에서 바로 조회 후 불릿 생성합니다.

| 순서 | 실행할 코드 | 역할 |
|------|-------------|------|
| 1 | `python 00_db_smoke_test.py` | (선택) DB 연결·테이블 조회 테스트 |
| 2 | `python generate_category_bullets_from_db.py --category_contains 유모차` | DB에서 유모차 카테고리 상품+리뷰 조회 후 불릿 생성 |
| 또는 | `python generate_category_bullets_from_db.py --category_id 4` | 카테고리 ID로 조회 후 불릿 생성 |

**옵션**
- `--category_contains "유모차"` : 카테고리 경로에 "유모차"가 포함된 상품만 사용
- `--category_id 4` : `category_id == 4` 인 상품만 사용
- `--max_bullets 8` : 최대 불릿 개수 (기본 8)

---

## 경로 B: CSV 추출 후 불릿 생성

DB → CSV 먼저 뽑고, 그다음에 불릿을 만들 때 사용합니다.

| 순서 | 실행할 코드 | 역할 |
|------|-------------|------|
| 1 | `python 00_db_smoke_test.py` | (선택) DB 연결·테이블 조회 테스트 |
| 2 | `python export_all_products.py` | danawa_crawlingdb → **products_all.csv** 생성 |
| 3 | `python export_reviews_all.py` | danawa_crawlingdb → **reviews_all.csv** 생성 |
| 4 | `python 21_generate_category_bullets.py --template_only --category_contains 유모차` | CSV 기반으로 유모차 카테고리 불릿 생성 (템플릿 모드) |
| 또는 | `python 21_generate_category_bullets.py --template_only --category_id 4` | CSV 기반으로 category_id=4 불릿 생성 |

**참고**  
- 21번은 `--csv products_all.csv`, `--reviews_csv reviews_all.csv` 가 기본값입니다.  
- `--template_only` 없이 쓰려면 `--model_dir` 에 LoRA 디렉터리 지정이 필요합니다.

---

## 경로 C: 카테고리 불릿 LoRA 학습까지 (전체 파이프라인)

학습용 데이터 만들기 ~ 학습 ~ 불릿 생성까지 한 번에 보고 싶을 때입니다.

| 순서 | 실행할 코드 | 역할 |
|------|-------------|------|
| 1 | 경로 B의 1~3 | DB 연결 테스트, products_all.csv, reviews_all.csv 생성 |
| 2 | `python 19_prepare_category_bullet_sft.py --input_csv products_all.csv --out_jsonl training_category_bullets_sft.jsonl` | 불릿 SFT용 JSONL 생성 |
| 3 | `python 20_train_category_bullet_lora.py` (해당 스크립트 인자/설정 확인) | LoRA 학습 |
| 4 | `python 21_generate_category_bullets.py --model_dir <LoRA출력경로> --category_id 4` | 학습된 모델로 불릿 생성 |

---

## 요약: 지금 당장 불릿만 보고 싶을 때

1. **DB만 쓰고 싶다**  
   ```bash
   python generate_category_bullets_from_db.py --category_contains 유모차
   ```

2. **이미 CSV가 있다**  
   ```bash
   python 21_generate_category_bullets.py --template_only --category_contains 유모차
   ```

3. **DB/CSV 둘 다 처음부터**  
   ```bash
   python 00_db_smoke_test.py
   python export_all_products.py
   python export_reviews_all.py
   python generate_category_bullets_from_db.py --category_contains 유모차
   ```
