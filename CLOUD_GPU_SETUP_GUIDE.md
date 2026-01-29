# 클라우드 GPU 환경 구성 가이드

## 1. Google Colab Pro (가장 추천)

### 장점
- **간단한 설정**: 브라우저만으로 시작 가능
- **무료/유료 옵션**: 무료는 T4, Pro는 V100/A100
- **사전 설치된 라이브러리**: PyTorch, Transformers 등
- **무료 저장공간**: Google Drive 연동

### 사용 방법

#### 1단계: Colab 노트북 생성
```python
# 새 노트북 생성 후 GPU 활성화
# 런타임 > 런타임 유형 변경 > GPU 선택 (T4/V100/A100)
```

#### 2단계: 프로젝트 파일 업로드
```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 파일 복사
!cp -r /content/drive/MyDrive/tips_ai /content/
!cd /content/tips_ai
```

#### 3단계: 학습 실행
```python
# 패키지 설치
!pip install transformers peft datasets accelerate torch

# 학습 실행
!python 23_train_report_summary_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_jsonl training_report_summary_sft_500.jsonl \
  --out_dir results_report/qwen2.5-1_5b-lora-report-summary-500 \
  --epochs 5 --lr 1e-4 --max_length 2048
```

### 비용
- 무료: T4 GPU (제한적)
- Pro: 월 $10 (V100/A100, 더 많은 사용 시간)
- Pro+: 월 $50 (더 많은 리소스)

---

## 2. Kaggle Notebooks (무료, 추천)

### 장점
- **완전 무료**: P100 GPU, 주간 30시간
- **데이터셋 호스팅**: 학습 데이터 업로드 가능
- **커뮤니티**: 많은 예제와 공유 노트북

### 사용 방법

#### 1단계: Kaggle 계정 생성 및 노트북 생성
- https://www.kaggle.com 접속
- 새 노트북 생성
- GPU 옵션 활성화

#### 2단계: 데이터셋 업로드
- Datasets 탭에서 학습 데이터 업로드
- 또는 Google Drive에서 직접 로드

#### 3단계: 학습 코드 작성
```python
# Kaggle 노트북 예시
import os
os.chdir('/kaggle/working')

# 데이터 로드
import pandas as pd
# ... 학습 코드 ...
```

### 제한사항
- 주간 30시간 GPU 사용 제한
- 세션당 최대 9시간
- 인터넷 접근 제한 (설치 시 `--enable-internet` 필요)

---

## 3. RunPod / Vast.ai (저렴한 GPU 렌탈)

### 장점
- **저렴한 비용**: 시간당 $0.1~$0.5
- **다양한 GPU**: RTX 3090, A100 등 선택 가능
- **전체 서버 접근**: 완전한 제어

### 사용 방법

#### 1단계: 계정 생성 및 포드 생성
- https://www.runpod.io 또는 https://vast.ai 접속
- GPU 포드 생성 (RTX 3090 24GB 추천)

#### 2단계: SSH 접속
```bash
ssh root@<pod-ip>
```

#### 3단계: 환경 설정
```bash
# Git 클론 또는 파일 업로드
git clone <your-repo>
cd tips_ai

# 가상환경 설정
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 학습 실행
python 23_train_report_summary_lora.py ...
```

### 비용 예시
- RTX 3090 24GB: 시간당 약 $0.3
- A100 40GB: 시간당 약 $1.5
- 학습 5시간 예상: 약 $1.5~$7.5

---

## 4. AWS SageMaker / GCP Vertex AI (엔터프라이즈급)

### 장점
- **확장성**: 필요에 따라 리소스 조정
- **통합 도구**: 모델 배포, 모니터링 등
- **안정성**: 프로덕션 환경에 적합

### 단점
- **비용**: 상대적으로 비쌈
- **복잡한 설정**: 초기 설정이 복잡

### AWS SageMaker 예시
```python
from sagemaker.huggingface import HuggingFace

# 학습 작업 정의
estimator = HuggingFace(
    entry_point='train.py',
    instance_type='ml.g4dn.xlarge',  # T4 GPU
    role='your-role',
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
)

estimator.fit({'training': 's3://your-bucket/data'})
```

---

## 5. 로컬 환경 개선 (장기적)

### GPU 업그레이드
- **RTX 4060 Ti 16GB**: 약 $500, 16GB VRAM
- **RTX 4070 Ti**: 약 $800, 12GB VRAM
- **RTX 4080**: 약 $1,200, 16GB VRAM
- **RTX 4090**: 약 $1,600, 24GB VRAM

### RAM 증설
- 32GB 이상 권장
- DDR4/DDR5 3200MHz 이상

### SSD
- NVMe SSD 권장 (학습 데이터 로딩 속도)

---

## 추천 순서

1. **즉시 시작**: Google Colab Pro ($10/월)
   - 가장 빠르게 시작 가능
   - V100/A100으로 빠른 학습

2. **무료 옵션**: Kaggle Notebooks
   - 완전 무료
   - 주간 30시간 제한

3. **저렴한 렌탈**: RunPod/Vast.ai
   - 필요할 때만 사용
   - 시간당 $0.3~$0.5

4. **장기 사용**: 로컬 GPU 업그레이드
   - 자주 학습한다면 투자 가치 있음

---

## 학습 시간 비교 (500 examples 기준)

| 환경 | GPU | 예상 시간 | 비용 |
|------|-----|----------|------|
| 로컬 (현재) | RTX 3060 8GB | 2-3시간 | 무료 |
| Colab Pro | V100 | 30-60분 | $10/월 |
| Colab Pro | A100 | 15-30분 | $10/월 |
| Kaggle | P100 | 1-2시간 | 무료 |
| RunPod | RTX 3090 | 45-90분 | $0.3/시간 |

---

## 다음 단계

1. **Google Colab Pro 시작하기** (가장 추천)
   - 빠른 설정
   - 좋은 성능
   - 합리적인 비용

2. **학습 데이터 준비**
   - `training_report_summary_sft_500.jsonl` 파일 준비
   - Google Drive에 업로드

3. **학습 스크립트 수정** (필요시)
   - 경로 조정
   - 출력 디렉토리 설정

원하는 옵션을 알려주시면 더 자세한 설정 가이드를 제공하겠습니다!
