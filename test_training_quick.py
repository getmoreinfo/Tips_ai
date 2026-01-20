# test_training_quick.py
# 빠른 테스트: 학습이 정상 작동하는지 확인 (전체 학습은 하지 않음)

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

print("=" * 60)
print("빠른 학습 테스트 (1 Epoch만, 몇 스텝만)")
print("=" * 60)
print()

# GPU 확인
print("GPU 상태 확인:")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv('training_data_10000.csv')
df = df.dropna(subset=['category_name'])
df = df[df['category_name'].notna()]

print(f"총 데이터: {len(df):,}개")
print(f"카테고리 수: {df['category_name'].nunique()}개")
print()

# 데이터 분할 (소량만 사용하여 빠른 테스트)
print("데이터 분할 중...")
# 테스트를 위해 소량만 사용
df_small = df.head(1000)  # 1000개만 사용

train_df, temp_df = train_test_split(
    df_small, 
    test_size=0.2, 
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

print(f"학습 데이터: {len(train_df):,}개")
print(f"검증 데이터: {len(val_df):,}개")
print(f"테스트 데이터: {len(test_df):,}개")
print()

# Dataset 변환
train_dataset = Dataset.from_pandas(train_df[['text', 'category_name']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['text', 'category_name']].reset_index(drop=True))

# 모델 및 토크나이저 로드
print("모델 로드 중...")
model_name = "intfloat/multilingual-e5-large"
num_labels = df['category_name'].nunique()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="single_label_classification"
)

model.gradient_checkpointing_enable()
print(f"모델 로드 완료: {model_name}")
print(f"분류 라벨 수: {num_labels}개")
print()

# 토크나이징
print("토크나이징 중...")
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
print("토크나이징 완료")
print()

# 레이블 매핑
category_names = sorted([cat for cat in df['category_name'].unique() if pd.notna(cat)])
category_to_label = {cat_name: idx for idx, cat_name in enumerate(category_names)}
label_to_category = {idx: cat_name for cat_name, idx in category_to_label.items()}

def convert_labels(examples):
    examples['labels'] = [category_to_label[cat_name] for cat_name in examples['category_name']]
    return examples

train_dataset = train_dataset.map(convert_labels, batched=True)
val_dataset = val_dataset.map(convert_labels, batched=True)
print("레이블 매핑 완료")
print()

# 평가 메트릭
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

# 학습 설정 (빠른 테스트용)
use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir='./results_test',
    
    per_device_train_batch_size=4,  # 작은 배치
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    
    fp16=use_fp16,
    
    learning_rate=2e-5,
    num_train_epochs=1,  # 1 Epoch만
    max_steps=10,  # 최대 10 스텝만 (테스트용)
    
    eval_strategy='steps',
    eval_steps=5,
    
    logging_dir='./logs_test',
    logging_steps=1,
    report_to='none',
    
    save_total_limit=1,
    dataloader_num_workers=0,
    warmup_steps=2,
    disable_tqdm=False,
)

print("=" * 60)
print("학습 시작 (테스트: 10 스텝만)")
print("=" * 60)
print()

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 학습 시작
try:
    print("학습 시작 중...")
    print("GPU 사용률을 확인하려면 다른 터미널에서 'nvidia-smi' 실행")
    print()
    
    trainer.train()
    
    print()
    print("=" * 60)
    print("✅ 테스트 성공!")
    print("=" * 60)
    print("학습이 정상적으로 작동합니다!")
    print("이제 전체 학습을 실행할 수 있습니다: python 02_finetune_local.py")
    print()
    
except Exception as e:
    print()
    print("=" * 60)
    print("❌ 테스트 실패")
    print("=" * 60)
    print(f"오류 발생: {e}")
    print()
    raise
