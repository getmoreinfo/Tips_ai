# AI-TR: 카테고리 리포트 요약 (LoRA SFT)

**다나와 스타일 상품·리뷰 데이터**를 입력으로, **카테고리별 시장 개요 요약**(marketOverviewSummary)을 생성하는 파이프라인입니다.  
Qwen2.5-7B LoRA SFT로 리뷰 텍스트 기반 인사이트(키워드·세그먼트·실행 시사점)까지 반영한 요약을 생성합니다.

---

## 1. 파이프라인 (실행 순서)

| 단계 | 스크립트 | 역할 |
|------|----------|------|
| 1 | `21_export_products_reviews_from_db.py` | DB → CSV 추출 (`products_all.csv`, `reviews_all.csv`) |
| 2 | `22_prepare_report_summary_sft.py` | SFT용 JSONL 생성 (카테고리·리뷰 인사이트 포함) |
| 3 | `23_train_report_summary_lora.py` | Qwen2.5-7B LoRA 학습 |
| 4 | `24_generate_report_summary.py` | 학습된 adapter로 카테고리 리포트 요약 생성 |
| 5 | `25_generate_category_report_from_csv.py` | CSV 기반 전체 카테고리 리포트 생성 (백엔드 연동용) |
| 6 | `26_evaluate_report_summary.py` | 요약 품질 평가 (DB/파일 입력) |

**Colab에서 실행 시:** 2 → 3 → 4 순서로 실행. 코드는 로컬에서 Git push → Colab에서 Git pull 후 동일 스크립트 실행.  
상세 명령·오류 대응은 **docs/COLAB_RUN_ORDER.md** 참고.

---

## 2. 설계·결정 요약

- **7B 선택 이유**  
  수치 해석 중심 요약은 1.5B/3B로 충분했으나, **리뷰 텍스트 기반 인사이트**(사용 시나리오·맥락 해석)를 넣을 때 문장 논리·표현 다양성 한계가 있어 **모델 규모 상승 = 기능 확장** 목표로 7B로 확장. Colab A100 40GB 기준으로 학습 가능하도록 파라미터 조정.

- **리뷰 기반 proxy 전제**  
  리포트의 매출·점유·성장 수치는 **실제 거래 데이터가 아니라**, **리뷰 수를 대리 지표(proxy)**로 쓴 **상대적 구조 해석** 결과입니다. 소비자에게 이 전제를 명시해 오해를 방지합니다.

- **학습·추론 환경**  
  로컬 대신 **Google Colab + A100 GPU**에서 22→23→24를 End-to-End 실행. Git으로 코드·CSV 동기화.

- **운영 전략**  
  기본값은 **템플릿 기반 요약**, 모델 요약은 **선택 옵션**. JSON 오류·언어 혼입 등이 잦거나 품질 우위가 없으면 템플릿-only로 유지.


---

## 3. 주요 파일

| 구분 | 파일 |
|------|------|
| **진입** | `README.md` (본 문서) |
| **파이프라인** | `21_export_...` ~ `26_evaluate_...` (번호 순) |
| **공통 라이브러리** | `report_summary_lib.py`, `ai_report_bullets_lib.py`, `db_category_loader.py` |
| **설정 예시** | `env.example`, `requirements_db.txt`, `requirements_lora.txt` |
| **Colab 실행** | `colab_run_all.py` / `colab_run_all.ipynb` (선택) |

**파이프라인 각 코드 상세 설명**은 **docs/PIPELINE_CODE_GUIDE.md** 에 정리해 두었습니다.  
**상세 가이드(Colab 설정, GPU, Mac 설정 등)**는 **docs/** 폴더에 정리해 두었습니다. → **docs/README.md** 에서 목록 확인.

---

## 4. 데이터·환경

- **입력:** `products_all.csv`, `reviews_all.csv` (21번으로 DB에서 추출하거나 기존 CSV 사용)
- **환경 변수:** DB 사용 시 `.env` 참고. 형식은 `env.example` 참고.
- **학습 결과:** `results/` 등은 `.gitignore` 대상. Colab에서는 Drive 복사 또는 로컬 다운로드로 보관.

---

*최종 업데이트: 2026-01-29.*
