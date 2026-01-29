# Colab 실행 순서 — 복붙용 명령어 정리

Colab에서 **새 노트북** 만들고, 아래 **순서대로** 각 코드 블록을 **새 셀에 붙여넣은 뒤 실행**하세요.  
오류가 나면 **오류별 조치** 섹션에서 해당 메시지로 이동해 안내된 명령만 실행하면 됩니다.

---

## 🚀 자동 실행 (한 셀에 모두)

**수동으로 1→2→3→4→4-1 실행하기 싫으면** 아래만 쓰면 된다.

1. **`colab_run_all.ipynb`** 를 Colab에서 연다 (Drive에 올리거나 Colab → 파일 → 노트북 업로드).
2. **런타임 → 런타임 유형 변경** → **GPU** 선택 → 저장.
3. **코드 셀 하나**만 있다. 그 셀 **실행**.
4. Drive 마운트 **권한 허용** (팝업) 한 번만 하면, 이후 **복사 → 설치 → 학습 → Drive 복사**까지 자동 진행.

또는 **스크립트로:** Drive `tips_ai_colab`에 **`colab_run_all.py`** 를 넣어 두고, Colab에서:

```python
from google.colab import drive
drive.mount('/content/drive')
%run /content/drive/MyDrive/tips_ai_colab/colab_run_all.py
```

위 두 셀만 순서대로 실행해도 2→3→4→4-1 이 자동으로 돌아간다.

---

## 사전 준비

- **Colab:** https://colab.research.google.com → 새 노트북
- **런타임:** 런타임 → 런타임 유형 변경 → **GPU** (T4 / A100 등) 선택 → 저장
- **Drive:** `내 드라이브` 안에 `tips_ai_colab` 폴더 생성 후 아래 **필수 파일** 업로드  
  - `22_prepare_report_summary_sft.py`  
  - `23_train_report_summary_lora.py`  
  - `report_summary_lib.py`  
  - `ai_report_bullets_lib.py`  
  - `training_report_summary_sft_500.jsonl` (또는 22로 만든 JSONL)  
  - (선택) `products_all.csv`, `reviews_all.csv`, `24_generate_report_summary.py`, `25_generate_category_report_from_csv.py`

---

## 1번. Drive 마운트

**새 코드 셀**에 아래만 넣고 실행:

```python
from google.colab import drive
drive.mount('/content/drive')
```

- 브라우저에서 **Drive 권한 허용** 안 나오면 팝업 차단 해제 후 다시 실행.
- 출력에 `Mounted at /content/drive` 나오면 성공.

---

## 2번. 작업 디렉터리 + 파일 복사

**새 코드 셀**에 아래만 넣고 실행:

```python
import os
os.makedirs('/content/tips_ai', exist_ok=True)
%cd /content/tips_ai
!cp -r /content/drive/MyDrive/tips_ai_colab/* /content/tips_ai/
!ls -lh
```

- `ls` 결과에 `23_train_report_summary_lora.py`, `training_report_summary_sft_500.jsonl` 등이 보여야 함.
- **안 보이면:** Drive `tips_ai_colab` 경로·파일명 확인 후 1번부터 다시.

---

## 3번. 패키지 설치 + GPU 확인

**새 코드 셀**에 아래만 넣고 실행:

```python
!pip install -q transformers peft datasets accelerate torch bitsandbytes pandas
import torch
print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

- `GPU: True GPU이름` 나오면 GPU 사용 가능.

---

## (선택) 3-1. SFT JSONL 생성 — CSV 쓸 때만

`products_all.csv`, `reviews_all.csv`가 `tips_ai_colab`에 **있을 때만** 실행.

**새 코드 셀**에 아래만 넣고 실행:

```python
%cd /content/tips_ai
!python 22_prepare_report_summary_sft.py \
  --input_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --out_jsonl training_report_summary_sft.jsonl \
  --samples_per_category 50 \
  --min_products 10
!head -n 1 training_report_summary_sft.jsonl
```

- **이걸 실행했으면** 4번에서 `--train_jsonl training_report_summary_sft.jsonl` 로 바꿔서 사용.

---

## 4번. LoRA 학습 (Qwen 7B)

**새 코드 셀**에 아래만 넣고 실행:

```python
%cd /content/tips_ai
!ls -la 23_train_report_summary_lora.py training_report_summary_sft_500.jsonl
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_500.jsonl \
  --out_dir /content/tips_ai/results/qwen2.5-7b-lora-report-summary \
  --epochs 5 --lr 1e-4 --max_length 2048 \
  --lora_r 8 --lora_alpha 16 --batch_size 1 --grad_accum 8 \
  --save_steps 100 --logging_steps 10
```

- **3-1 실행했을 때:** `--train_jsonl training_report_summary_sft_500.jsonl` 를  
  `--train_jsonl training_report_summary_sft.jsonl` 로 **바꿔서** 실행.
- `ls` 에서 `No such file` 나오면 **2번** 다시 실행한 뒤 4번 재실행.
- 학습 끝나면 **반드시 4-1** 실행.

---

## 4-1. 학습 결과 → Drive 복사 (필수)

**새 코드 셀**에 아래만 넣고 실행:

```python
!ls -la /content/tips_ai/results/qwen2.5-7b-lora-report-summary
!mkdir -p /content/drive/MyDrive/tips_ai_colab/results
!cp -r /content/tips_ai/results/qwen2.5-7b-lora-report-summary /content/drive/MyDrive/tips_ai_colab/results/
!ls -la /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary
print("\n✅ Drive 저장 완료: 내 드라이브 → tips_ai_colab → results → qwen2.5-7b-lora-report-summary")
```

- 4번 **끝난 뒤** 곧바로 실행. 런타임 끊기기 전에 반드시 실행.
- Drive `tips_ai_colab/results/` 안에 `adapter_model.safetensors` 등이 있어야 함.

---

## (선택) 5. 템플릿 리포트 (25) — CSV 있을 때

**새 코드 셀**에 아래만 넣고 실행 (`유모차` 대신 원하는 카테고리 문자열로 변경):

```python
%cd /content/tips_ai
!python 25_generate_category_report_from_csv.py \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "유모차" \
  --out_json /content/drive/MyDrive/tips_ai_colab/report_category.json
!head -c 500 /content/drive/MyDrive/tips_ai_colab/report_category.json
```

---

## (선택) 6. 모델 리포트 (24) — 4-1 끝난 뒤, CSV 있을 때

**새 코드 셀**에 아래만 넣고 실행:

```python
%cd /content/tips_ai
!python 24_generate_report_summary.py \
  --model_dir /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary \
  --products_csv products_all.csv \
  --reviews_csv reviews_all.csv \
  --category_contains "유모차"
```

- `--category_contains` 를 원하는 카테고리로 바꿔도 됨.

---

## 오류별 조치 — 나온 메시지에 맞는 것만 실행

아래는 **에러 메시지** 기준으로, **그때 쳐야 할 명령어**만 정리한 것이다.  
순서대로 **1번 → 2번 → …** 다시 돌리는 게 좋은 경우도 함께 적어두었다.

---

### `No such file or directory` / `can't open file ... 23_train_report_summary_lora.py`

**원인:** 2번(복사) 안 했거나, 복사 실패. `/content/tips_ai` 에 스크립트 없음.

**할 일:**  
1) 1번 Drive 마운트 셀 **다시 실행**  
2) 2번 복사 셀 **다시 실행**  
3) 아래로 **파일 있는지 확인**:

```python
%cd /content/tips_ai
!ls -la 23_train_report_summary_lora.py training_report_summary_sft_500.jsonl
```

- 두 파일 다 보이면 → **4번** 학습 셀 다시 실행.  
- 안 보이면 → Drive `tips_ai_colab` 안에 위 파일들이 있는지 확인 후 2번 다시.

---

### `FileNotFoundError` / `No such file` (복사할 때 `tips_ai_colab` 관련)

**원인:** Drive에 `tips_ai_colab` 없거나, 경로가 `MyDrive` 기준이 아님.

**할 일:**  
1) Drive에서 `내 드라이브` → `tips_ai_colab` 폴더 있는지 확인.  
2) 없으면 만들고, 필수 파일 넣은 뒤 **1번 → 2번** 순서로 다시 실행.

---

### `qwen2.5-7b-lora-report-summary` 폴더만 있고 **안이 비어 있음**

**원인:** 4번에서 Drive 직저장 쓰지 않고 로컬 저장 쓰는 구조라, Drive로 복사(4-1)를 안 한 상태.

**할 일:**  
1) **로컬에** 결과가 있는지 확인:

```python
!ls -la /content/tips_ai/results/qwen2.5-7b-lora-report-summary
```

- 여기 `adapter_model.safetensors`, `training_metadata.json` 등이 보이면 → **4-1** 셀 **지금 실행**:

```python
!mkdir -p /content/drive/MyDrive/tips_ai_colab/results
!cp -r /content/tips_ai/results/qwen2.5-7b-lora-report-summary /content/drive/MyDrive/tips_ai_colab/results/
!ls -la /content/drive/MyDrive/tips_ai_colab/results/qwen2.5-7b-lora-report-summary
```

- 로컬 폴더도 비어 있으면 → 런타임 끊긴 뒤라 결과 날아간 것. **4번 학습부터 다시** 돌리고, 끝나자마자 **4-1** 실행.

---

### `ModuleNotFoundError: No module named 'google.colab'`

**원인:** Colab이 아니라 **로컬**(Cursor 등)에서 노트북 실행 중.

**할 일:** 이 순서는 **Colab 웹** (colab.research.google.com) 에서만 동작함.  
Colab에서 노트북 열고 **1번부터** 다시 실행.

---

### GPU / CUDA OOM (Out of Memory)

**원인:** GPU 메모리 부족.

**할 일:** 4번 `!python 23_train_report_summary_lora.py ...` 에서 아래처럼 바꿔서 실행:

```python
%cd /content/tips_ai
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --train_jsonl training_report_summary_sft_500.jsonl \
  --out_dir /content/tips_ai/results/qwen2.5-7b-lora-report-summary \
  --epochs 5 --lr 1e-4 --max_length 1024 \
  --lora_r 4 --lora_alpha 8 --batch_size 1 --grad_accum 16 \
  --save_steps 100 --logging_steps 10
```

- `--max_length 1024`, `--lora_r 4`, `--lora_alpha 8`, `--grad_accum 16` 로 줄인 것.  
- 끝나면 마찬가지로 **4-1** 실행.

---

### 기타 `pip` / `transformers` 등 패키지 에러

**할 일:** 3번 셀 **다시 실행** (패키지 재설치):

```python
!pip install -q transformers peft datasets accelerate torch bitsandbytes pandas
import torch
print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
```

---

## 체크리스트 (실행 전)

- [ ] Colab **GPU** 런타임 선택
- [ ] Drive `tips_ai_colab` 에 **필수 파일** 모두 업로드
- [ ] **1 → 2 → 3 → 4 → 4-1** 순서로 실행
- [ ] 4번 끝난 뒤 **반드시 4-1** 실행 (Drive 복사)

---

이 문서는 `colab_train.ipynb` 수정·버그 수정할 때마다 같이 갱신한다.  
**무엇을 쳐야 하는지**만 보려면 여기서 복붙해서 쓰면 된다.
