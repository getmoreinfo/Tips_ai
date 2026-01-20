# 02_finetune_local.py
# 전체 코드

import torch
import pandas as pd
import numpy as np
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# GPU 확인
print("=" * 60)
print("GPU 상태 확인")
print("=" * 60)
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("경고: GPU를 사용할 수 없습니다. CPU로 학습하면 매우 느립니다.")
print()

# 1. 데이터 로드
print("=" * 60)
print("데이터 로드")
print("=" * 60)

df = pd.read_csv('training_data_10000.csv')

# category_name이 NaN인 경우 제거
df = df.dropna(subset=['category_name'])
df = df[df['category_name'].notna()]

print(f"총 데이터: {len(df):,}개")
print(f"카테고리 수: {df['category_name'].nunique()}개")

# 카테고리별 최소/최대 샘플 수
category_counts = df['category_name'].value_counts()
print(f"최소 샘플 수: {category_counts.min()}개 (카테고리: {category_counts.idxmin()})")
print(f"최대 샘플 수: {category_counts.max()}개 (카테고리: {category_counts.idxmax()})")
print(f"평균 샘플 수: {category_counts.mean():.1f}개")

# 2. Train/Validation/Test 분할
print("\n" + "=" * 60)
print("데이터 분할")
print("=" * 60)

# Train 80%, Temp 20%
train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    #stratify=df['category_name'], 
    random_state=42
)

# Temp를 Validation 50%, Test 50%로 분할
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    #stratify=temp_df['category_name'],
    random_state=42
)

print(f"학습 데이터: {len(train_df):,}개 (80%)")
print(f"검증 데이터: {len(val_df):,}개 (10%)")
print(f"테스트 데이터: {len(test_df):,}개 (10%)")

# Dataset 변환
train_dataset = Dataset.from_pandas(train_df[['text', 'category_name']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['text', 'category_name']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['text', 'category_name']].reset_index(drop=True))

# 3. 모델 및 토크나이저 로드
print("\n" + "=" * 60)
print("모델 로드")
print("=" * 60)

model_name = "intfloat/multilingual-e5-large"
num_labels = df['category_name'].nunique()

print(f"모델: {model_name}")
print(f"분류 라벨 수: {num_labels}개")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="single_label_classification"
)

# Gradient checkpointing (메모리 절약 기법)
model.gradient_checkpointing_enable()
print("Gradient Checkpointing 활성화 (메모리 절약)")

# 4. 토크나이징
print("\n" + "=" * 60)
print("토크나이징")
print("=" * 60)

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=128
    )

print("학습 데이터 토크나이징 중...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
print("검증 데이터 토크나이징 중...")
val_dataset = val_dataset.map(tokenize_function, batched=True)
print("테스트 데이터 토크나이징 중...")
test_dataset = test_dataset.map(tokenize_function, batched=True)
print("토크나이징 완료")

# 5. 레이블 매핑 생성
print("\n" + "=" * 60)
print("레이블 매핑 생성")
print("=" * 60)

# NaN 값을 제거하고 정렬
category_names = sorted([cat for cat in df['category_name'].unique() if pd.notna(cat)])
category_to_label = {cat_name: idx for idx, cat_name in enumerate(category_names)}
label_to_category = {idx: cat_name for cat_name, idx in category_to_label.items()}

print(f"매핑 생성 완료: {len(category_to_label)}개")
print("예시 매핑:")
for i, (cat_name, label) in enumerate(list(category_to_label.items())[:5]):
    print(f"  {cat_name[:60]} -> 레이블 {label}")
if len(category_to_label) > 5:
    print(f"  ... 외 {len(category_to_label) - 5}개")

def convert_labels(examples):
    examples['labels'] = [category_to_label[cat_name] for cat_name in examples['category_name']]
    return examples

train_dataset = train_dataset.map(convert_labels, batched=True)
val_dataset = val_dataset.map(convert_labels, batched=True)
test_dataset = test_dataset.map(convert_labels, batched=True)

# 6. 평가 메트릭 정의
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

# 7. 학습 설정
print("\n" + "=" * 60)
print("학습 설정")
print("=" * 60)

# CUDA 사용 가능 여부에 따라 fp16 자동 설정
use_fp16 = torch.cuda.is_available()
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"FP16 사용: {use_fp16}")

training_args = TrainingArguments(
    output_dir='./results',
    
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    
    # CUDA 사용 가능 여부에 따라 fp16 자동 설정
    fp16=use_fp16,
    
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    
    logging_dir='./logs',
    logging_steps=10,
    logging_first_step=True,
    report_to='none',
    
    save_total_limit=2,
    dataloader_num_workers=0,  # Windows 안정성을 위해 0으로 설정
    warmup_steps=50,
    disable_tqdm=False,
)

print(f"효과적 배치 사이즈: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"모드: {'GPU (FP16)' if use_fp16 else 'CPU/GPU (FP32)'}")

# 8. Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 9. 학습 시작
print("\n" + "=" * 60)
print("Fine-tuning 시작")
print("=" * 60)
print(f"총 Epoch: {training_args.num_train_epochs}")
print()

try:
    # 학습 실행
    trainer.train()
    
    # 10. 최종 평가 (Test set)
    print("\n" + "=" * 60)
    print("최종 평가 (Test Set)")
    print("=" * 60)
    
    test_results = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_results.predictions, axis=1)
    test_labels = test_results.label_ids
    
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # 11. 모델 저장
    print("\n" + "=" * 60)
    print("모델 저장")
    print("=" * 60)
    
    save_dir = './results/finetuned_e5_large'
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 메타데이터 저장
    metadata = {
        'category_to_label': category_to_label,
        'label_to_category': label_to_category,
        'num_labels': num_labels,
        'training_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'model_name': model_name,
        'max_length': 128,
        'epochs': training_args.num_train_epochs,
    }
    
    with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장 완료: {save_dir}")
    print("\n" + "=" * 60)
    print("Fine-tuning 완료")
    print("=" * 60)
    print("다음 단계: python 03_use_finetuned_model.py 실행")
    
except KeyboardInterrupt:
    print("\n학습이 중단되었습니다.")