# AI-TR: 카테고리 리포트 요약 (LoRA SFT)

**다나와 스타일 상품·리뷰 데이터**를 입력으로, **카테고리별 시장 개요 요약**(marketOverviewSummary)을 생성하는 파이프라인입니다.  
Qwen2.5-7B LoRA SFT로 리뷰 텍스트 기반 인사이트(키워드·세그먼트·실행 시사점)까지 반영한 요약을 생성합니다.

---

## 사용 모델 (상세)

### 모델을 채택한 이유

**1) Qwen2.5-7B-Instruct**

- **규모(7B)**: 수치만 나열하는 단순 요약은 1.5B/3B로도 가능했으나, **리뷰 텍스트 기반 인사이트**(사용 시나리오·맥락 해석, 실행 시사점)까지 반영하려면 문장 논리와 표현 다양성이 필요해 **7B로 확장**했습니다. 즉, 모델 규모 상승 = 기능 확장(리뷰 맥락까지 해석하는 요약)을 목표로 한 선택입니다.
- **Qwen 계열**: 인스트럭트/채팅 형식을 기본 지원하고, 한국어·다국어 성능이 좋으며, Hugging Face에서 바로 로드해 오픈소스 기반으로 학습·추론할 수 있습니다.
- **실행 환경**: Colab A100 40GB 도입으로 7B 전체 로드 + LoRA 학습이 가능해졌고, OOM을 피하기 위해 max_length·batch·LoRA rank 등은 40GB 기준으로 조정했습니다.

**2) LoRA(PEFT)**

- **전체 파인튜닝 대비**: 7B 전체를 다시 학습하면 GPU 메모리·학습 시간·저장 용량이 크게 듭니다. **LoRA**는 베이스 모델은 고정하고 **adapter(저랭크 행렬)만 학습**하므로, 메모리와 저장량을 줄이면서도 태스크(리포트 요약)에 맞게 조정할 수 있습니다.
- **배포**: 학습 결과는 adapter만 저장하면 되고, 추론 시에도 베이스 모델 + adapter 조합만 있으면 되어 관리와 배포가 단순합니다.

---

### 1) 베이스 모델

| 항목 | 내용 |
|------|------|
| **모델 ID** | `Qwen/Qwen2.5-7B-Instruct` |
| **출처** | Hugging Face Hub |
| **역할** | Causal LM(자동회귀 언어 모델), 인스트럭트/채팅 형식 지원 |
| **규모** | 7B 파라미터 |

- **학습 시**: `23_train_report_summary_lora.py` 에서 `--base_model` 인자로 지정 (기본값: `Qwen/Qwen2.5-7B-Instruct`).  
  Hugging Face에서 다운로드 후 `AutoModelForCausalLM.from_pretrained(base_model)` 로 로드하며, GPU 사용 시 `float16` 으로 로드합니다.
- **추론 시**: `24_generate_report_summary.py`, `26_evaluate_report_summary.py` 는 **학습 결과 디렉터리** 안의 `training_metadata.json` 에서 `base_model` 키를 읽어, 그 ID로 베이스 모델을 다시 로드합니다.  
  따라서 23번에서 7B로 학습했다면 추론 시에도 동일한 7B 베이스가 사용됩니다.

### 2) LoRA (PEFT)

| 항목 | 내용 |
|------|------|
| **의미** | Low-Rank Adaptation. 베이스 모델 전체를 다시 학습하지 않고, **저랭크 행렬(adapter)** 만 학습하여 특정 태스크에 맞춤. |
| **적용 레이어** | `q_proj`, `k_proj`, `v_proj`, `o_proj` (Attention) + `up_proj`, `down_proj`, `gate_proj` (MLP) |
| **rank (r)** | 16 (기본값). OOM 시 8로 낮춤. |
| **alpha** | 32 (기본값, 보통 2×r) |
| **dropout** | 0.05 |
| **task_type** | `CAUSAL_LM` |

- **학습 시**: `23_train_report_summary_lora.py` 에서 `LoraConfig` 로 위 설정을 넣고 `get_peft_model(model, lora)` 로 베이스 모델에 LoRA를 붙인 뒤, SFT 데이터(JSONL)로 학습합니다.  
  학습이 끝나면 **adapter 가중치만** `out_dir` 에 저장되고, 베이스 모델 본체는 저장하지 않습니다.
- **추론 시**: 베이스 모델을 `from_pretrained(base_model)` 로 로드한 뒤, `PeftModel.from_pretrained(base_model_obj, model_dir)` 로 학습된 adapter를 덧붙여 사용합니다.

### 3) 토크나이저

- **학습·추론 모두** 토크나이저는 **베이스 모델**에서만 로드합니다.  
  (`AutoTokenizer.from_pretrained(base_model)`)  
  LoRA adapter 디렉터리에는 `vocab` 등이 없을 수 있어, 24번·26번에서는 adapter 경로가 아닌 `training_metadata.json` 의 `base_model` 값으로 토크나이저를 로드합니다.

### 4) 학습/추론 파라미터 (참고)

| 구분 | 주요 설정 | 스크립트·위치 |
|------|-----------|----------------|
| **학습** | `max_length=2048`, `batch_size=2`, `grad_accum=16`, `epochs=10`, `lr=1e-4`, `warmup_ratio=0.05` | `23_train_report_summary_lora.py` (인자 및 `TrainingArguments`) |
| **추론** | `max_new_tokens=4096`(기본), `temperature=0`, `do_sample=False` | `24_generate_report_summary.py` (`model.generate`) |

- A100 40GB 기준으로 OOM을 피하기 위해 `max_length` 2048, `batch_size` 2, `lora_r` 16을 사용합니다.  
  학습량은 `epochs`·`grad_accum`으로 보완합니다.

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

## 실행 명령어

아래는 프로젝트 루트에서 실행하는 예시입니다. CSV·모델 경로는 환경에 맞게 바꿉니다.

```bash
# 1) DB → CSV 추출 (.env 설정 후)
python 21_export_products_reviews_from_db.py
# python 21_export_products_reviews_from_db.py --all --out_dir ./data

# 2) SFT용 JSONL 생성
python 22_prepare_report_summary_sft.py --input_csv products_all.csv --reviews_csv reviews_all.csv --out_jsonl training_report_summary_sft.jsonl

# 3) LoRA 학습 (GPU 필요, Colab A100 권장)
python 23_train_report_summary_lora.py --train_jsonl training_report_summary_sft.jsonl --out_dir results_report/qwen2.5-7b-lora-report-summary

# 4) 단일 카테고리 요약 생성 (학습된 adapter 사용)
python 24_generate_report_summary.py --model_dir results_report/qwen2.5-7b-lora-report-summary --products_csv products_all.csv --reviews_csv reviews_all.csv --category_contains "아기띠"

# 5) 카테고리 리포트 전체 JSON 생성 (모델 없이 템플릿)
python 25_generate_category_report_from_csv.py --products_csv products_all.csv --reviews_csv reviews_all.csv --category_contains "아기띠" --out_json report_stroller.json

# 6) 품질 평가 (DB 연결 필요)
python 26_evaluate_report_summary.py --model_dir results_report/qwen2.5-7b-lora-report-summary --categories "유모차" "그림/동화/놀이책" --out_json evaluation_report_summary.json
```

**PM 데모 UI**

```bash
# 로컬 서버로 HTML 뷰어 실행 (프로젝트 루트에서)
python serve_report_viewer.py --port 8000
# 브라우저: http://127.0.0.1:8000/pm_demo.html (한 페이지 데모)
#         http://127.0.0.1:8000/report_viewer.html?json=report_stroller.json (리포트 뷰어)
```

서버 없이 **pm_demo.html** 만 열어도 됩니다. (파일 더블클릭 또는 `open pm_demo.html`)

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
