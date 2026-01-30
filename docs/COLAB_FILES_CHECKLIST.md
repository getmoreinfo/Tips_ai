# Google Colab 업로드 파일 체크리스트

## ✅ 필수 파일 (반드시 업로드)

### 1. 학습 데이터
- [ ] `training_report_summary_sft_500.jsonl`
  - 위치: 프로젝트 루트
  - 크기: 약 수십 MB
  - 확인: `!head -n 1 training_report_summary_sft_500.jsonl` 실행 시 JSON 출력 확인

### 2. 학습 스크립트
- [ ] `23_train_report_summary_lora.py`
  - 위치: 프로젝트 루트
  - 역할: LoRA 학습 메인 스크립트

### 3. 리포트 요약 라이브러리
- [ ] `report_summary_lib.py`
  - 위치: 프로젝트 루트
  - 역할: 템플릿 기반 요약 생성 함수

### 4. 카테고리 메트릭 라이브러리
- [ ] `ai_report_bullets_lib.py`
  - 위치: 프로젝트 루트
  - 역할: 카테고리 지표 계산 함수

### 5. 패키지 의존성 (선택사항)
- [ ] `requirements_lora.txt`
  - 위치: 프로젝트 루트
  - 역할: 패키지 목록 (Colab에서 직접 설치 가능)

---

## 📁 Google Drive 업로드 구조

```
Google Drive/
└── tips_ai_colab/
    ├── training_report_summary_sft_500.jsonl  ✅
    ├── 23_train_report_summary_lora.py         ✅
    ├── report_summary_lib.py                  ✅
    ├── ai_report_bullets_lib.py              ✅
    └── requirements_lora.txt                   ✅ (선택)
```

---

## 🚫 업로드 불필요한 파일

다음 파일들은 **업로드하지 않아도 됩니다**:
- `products_all.csv` (학습 데이터 생성용, 학습에는 불필요)
- `reviews_all.csv` (학습 데이터 생성용, 학습에는 불필요)
- `db_category_loader.py` (평가용, 학습에는 불필요)
- `26_evaluate_report_summary.py` (평가용, 학습에는 불필요)
- 기타 CSV 파일들
- 결과 디렉토리 (`results_report/` 등)

---

## ✅ 업로드 전 확인사항

1. **파일 크기 확인**
   ```bash
   # 로컬에서 확인
   ls -lh training_report_summary_sft_500.jsonl
   ```

2. **파일 내용 확인**
   ```bash
   # 첫 번째 줄 확인
   head -n 1 training_report_summary_sft_500.jsonl
   ```

3. **필수 파일 존재 확인**
   ```bash
   ls -1 training_report_summary_sft_500.jsonl \
         23_train_report_summary_lora.py \
         report_summary_lib.py \
         ai_report_bullets_lib.py
   ```

---

## 📤 Google Drive 업로드 방법

### 방법 1: 웹 브라우저에서 직접 업로드 (추천)
1. https://drive.google.com 접속
2. 새 폴더 생성: `tips_ai_colab`
3. 폴더 열기
4. 파일 드래그 앤 드롭 또는 "업로드" 버튼 클릭

### 방법 2: Google Drive 데스크톱 앱 사용
1. Google Drive 데스크톱 앱 설치
2. 로컬 폴더 동기화
3. 파일 복사

---

## 🔍 Colab에서 파일 확인

업로드 후 Colab에서 확인:
```python
from google.colab import drive
drive.mount('/content/drive')

# 파일 확인
!ls -lh /content/drive/MyDrive/tips_ai_colab/

# 파일 복사
!cp -r /content/drive/MyDrive/tips_ai_colab/* /content/tips_ai/
!ls -lh /content/tips_ai/
```

---

## ⚠️ 주의사항

1. **파일 이름**: 대소문자 구분 주의
2. **경로**: Google Drive 경로는 `/content/drive/MyDrive/`로 시작
3. **권한**: 첫 Drive 마운트 시 권한 승인 필요
4. **용량**: 학습 데이터 파일이 크면 업로드 시간이 걸릴 수 있음

---

준비 완료되면 `COLAB_QUICK_START.md`의 코드를 실행하세요!
