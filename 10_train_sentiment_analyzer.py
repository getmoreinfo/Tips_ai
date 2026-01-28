# 10_train_sentiment_analyzer.py
# 역할: 리뷰 태그와 평점을 기반으로 제품 감성 분석 모델 학습

import torch
import pandas as pd
import numpy as np
import json
import ast
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# GPU 확인
print("=" * 60)
print("리뷰 감성 분석 모델 학습")
print("=" * 60)
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()

# 1. 데이터 로드 및 전처리
print("=" * 60)
print("데이터 로드 및 전처리")
print("=" * 60)

df = pd.read_csv('products_all_categorized.csv')

# review_tags가 있고 평점이 있는 데이터만 선택
df = df[df['review_tags'].notna() & (df['review_tags'] != '[]')]
df = df[df['average_rating'].notna() & (df['average_rating'] > 0)]

print(f"리뷰가 있는 데이터: {len(df):,}개")

# review_tags 파싱 함수
def parse_review_tags(tags_str):
    try:
        tags = ast.literal_eval(tags_str)
        if isinstance(tags, list):
            return ' '.join(tags)
        return ''
    except:
        return ''

# review_tags를 텍스트로 변환
df['review_text'] = df['review_tags'].apply(parse_review_tags)
df = df[df['review_text'].str.len() > 0]

# 감성 레이블 생성 (평점 기반) - 2클래스로 단순화
# 대부분 평점이 높으므로 4.7 기준으로 분류
def get_sentiment(rating):
    if rating >= 4.7:
        return '매우긍정'  # 평점 4.7 이상
    else:
        return '긍정'  # 평점 4.7 미만

df['sentiment'] = df['average_rating'].apply(get_sentiment)

print(f"전처리 후 데이터: {len(df):,}개")
print(f"\n감성 분포:")
print(df['sentiment'].value_counts())
print()

# 학습용 텍스트: 상품명 + 리뷰태그
df['text'] = df['name'] + ' | ' + df['review_text']

# 2. Train/Validation/Test 분할
print("=" * 60)
print("데이터 분할")
print("=" * 60)

# 클래스 불균형이 심하므로 stratify 사용
train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df['sentiment']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['sentiment']
)

print(f"학습 데이터: {len(train_df):,}개 (80%)")
print(f"검증 데이터: {len(val_df):,}개 (10%)")
print(f"테스트 데이터: {len(test_df):,}개 (10%)")

# Dataset 변환
train_dataset = Dataset.from_pandas(train_df[['text', 'sentiment']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['text', 'sentiment']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['text', 'sentiment']].reset_index(drop=True))

# 3. 모델 및 토크나이저 로드
print("\n" + "=" * 60)
print("모델 로드")
print("=" * 60)

model_name = "klue/roberta-base"  # 한국어 전용 모델
num_labels = 2  # 매우긍정, 긍정

print(f"모델: {model_name}")
print(f"분류 라벨 수: {num_labels}개")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="single_label_classification"
)

model.gradient_checkpointing_enable()
print("Gradient Checkpointing 활성화")

# 4. 토크나이징
print("\n" + "=" * 60)
print("토크나이징")
print("=" * 60)

MAX_LENGTH = 128

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=MAX_LENGTH
    )

print("토크나이징 중...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
print("토크나이징 완료")

# 5. 레이블 매핑
sentiment_to_label = {'매우긍정': 0, '긍정': 1}
label_to_sentiment = {0: '매우긍정', 1: '긍정'}

def convert_labels(examples):
    examples['labels'] = [sentiment_to_label[s] for s in examples['sentiment']]
    return examples

train_dataset = train_dataset.map(convert_labels, batched=True)
val_dataset = val_dataset.map(convert_labels, batched=True)
test_dataset = test_dataset.map(convert_labels, batched=True)

# 6. 클래스 가중치 계산 (불균형 해결)
class_counts = df['sentiment'].value_counts()
total = len(df)
class_weights = {
    sentiment_to_label['매우긍정']: total / (2 * class_counts.get('매우긍정', 1)),
    sentiment_to_label['긍정']: total / (2 * class_counts.get('긍정', 1)),
}
print(f"\n클래스 가중치: {class_weights}")

# 7. 평가 메트릭
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

# 8. 학습 설정
print("\n" + "=" * 60)
print("학습 설정")
print("=" * 60)

training_args = TrainingArguments(
    output_dir='./results_sentiment',
    
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  # 효과적 배치: 32
    
    fp16=False,
    bf16=False,
    
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    
    logging_dir='./logs_sentiment',
    logging_steps=50,
    report_to='none',
    
    save_total_limit=1,
    dataloader_num_workers=0,
    warmup_steps=100,
    save_safetensors=False,
)

print(f"효과적 배치 사이즈: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# 9. Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 10. 학습 시작
print("\n" + "=" * 60)
print("학습 시작")
print("=" * 60)

try:
    trainer.train()
    
    # 11. 최종 평가
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
    
    # 분류 리포트
    print("\n분류 리포트:")
    print(classification_report(
        test_labels, 
        test_predictions, 
        target_names=['매우긍정', '긍정']
    ))
    
    # 12. 모델 저장
    print("\n" + "=" * 60)
    print("모델 저장")
    print("=" * 60)
    
    save_dir = './results_sentiment/finetuned_sentiment_analyzer'
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 메타데이터 저장
    metadata = {
        'sentiment_to_label': sentiment_to_label,
        'label_to_sentiment': label_to_sentiment,
        'num_labels': num_labels,
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'model_name': model_name,
        'max_length': MAX_LENGTH,
    }
    
    with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장 완료: {save_dir}")
    print("\n다음 단계: python 11_use_sentiment_analyzer.py 실행")
    
except KeyboardInterrupt:
    print("\n학습이 중단되었습니다.")
except Exception as e:
    print(f"\n[ERROR] 학습 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
