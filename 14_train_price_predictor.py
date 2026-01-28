# 14_train_price_predictor.py
# 역할: 상품 특성으로 가격대 예측 모델 학습 (강화 버전)

import torch
import pandas as pd
import numpy as np
import json
import os
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("가격대 예측 모델 학습")
print("=" * 60)
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
print()

# 1. 데이터 로드 및 전처리
print("=" * 60)
print("데이터 로드 및 전처리")
print("=" * 60)

df = pd.read_csv('products_all_categorized.csv')

# 가격 정보가 있는 데이터만 선택
df = df[df['min_price'].notna() & (df['min_price'] > 0)]
df = df[df['name'].notna()]

print(f"가격 정보가 있는 데이터: {len(df):,}개")

# 가격 통계 확인
print(f"\n가격 통계:")
print(f"  최소: {df['min_price'].min():,.0f}원")
print(f"  최대: {df['min_price'].max():,.0f}원")
print(f"  평균: {df['min_price'].mean():,.0f}원")
print(f"  중앙값: {df['min_price'].median():,.0f}원")

# 가격대 분류 (3단계로 단순화)
def get_price_range(price):
    if price < 30000:
        return '저가'      # 3만원 미만
    elif price < 100000:
        return '중가'      # 3만원 ~ 10만원
    else:
        return '고가'      # 10만원 이상

df['price_range'] = df['min_price'].apply(get_price_range)

print(f"\n가격대 분포:")
print(df['price_range'].value_counts().sort_index())
print(f"\n가격대 기준:")
print("  저가: 3만원 미만")
print("  중가: 3만원 ~ 10만원")
print("  고가: 10만원 이상")

# 학습용 텍스트: 상품명 + 제조사 + 카테고리 (더 풍부한 정보)
df['manufacturer'] = df['manufacturer'].fillna('')
df['category_level2'] = df['category_level2'].fillna('')
df['category_level3'] = df['category_level3'].fillna('')

# 텍스트에 더 많은 정보 포함
df['text'] = (
    df['name'] + ' [제조사] ' + df['manufacturer'] + 
    ' [카테고리] ' + df['category_level2'] + ' > ' + df['category_level3']
)

print(f"\n전처리 완료: {len(df):,}개")

# 데이터 증강 함수
def augment_text(text):
    """텍스트 증강: 단어 순서 변경, 일부 제거 등"""
    augmented = []
    
    # 원본
    augmented.append(text)
    
    # 제조사 정보 제거 버전
    if '[제조사]' in text:
        parts = text.split('[제조사]')
        if len(parts) > 1:
            name = parts[0].strip()
            rest = parts[1].split('[카테고리]')
            if len(rest) > 1:
                category = rest[1].strip()
                augmented.append(f"{name} [카테고리] {category}")
    
    return augmented

print("데이터 증강 준비 완료")

# 2. Train/Validation/Test 분할
print("\n" + "=" * 60)
print("데이터 분할")
print("=" * 60)

train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df['price_range']
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['price_range']
)

print(f"학습 데이터: {len(train_df):,}개 (80%)")
print(f"검증 데이터: {len(val_df):,}개 (10%)")
print(f"테스트 데이터: {len(test_df):,}개 (10%)")

# Dataset 변환
train_dataset = Dataset.from_pandas(train_df[['text', 'price_range']].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[['text', 'price_range']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['text', 'price_range']].reset_index(drop=True))

# 3. 모델 및 토크나이저 로드
print("\n" + "=" * 60)
print("모델 로드")
print("=" * 60)

model_name = "klue/roberta-base"
num_labels = 3  # 3개 가격대 (저가/중가/고가)

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
price_to_label = {
    '저가': 0,
    '중가': 1,
    '고가': 2
}
label_to_price = {v: k for k, v in price_to_label.items()}

def convert_labels(examples):
    examples['labels'] = [price_to_label[p] for p in examples['price_range']]
    return examples

train_dataset = train_dataset.map(convert_labels, batched=True)
val_dataset = val_dataset.map(convert_labels, batched=True)
test_dataset = test_dataset.map(convert_labels, batched=True)

print(f"레이블 매핑: {price_to_label}")

# 6. 평가 메트릭
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

# 7. 클래스 가중치 계산 (불균형 데이터 처리)
print("\n" + "=" * 60)
print("클래스 가중치 계산")
print("=" * 60)

train_labels = [price_to_label[p] for p in train_df['price_range']]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"클래스 가중치: 저가={class_weights[0]:.3f}, 중가={class_weights[1]:.3f}, 고가={class_weights[2]:.3f}")

# 커스텀 Trainer (클래스 가중치 적용)
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device if hasattr(self.args, 'device') else 'cuda')
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 가중치 적용 CrossEntropy
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# 8. 학습 설정 (강화 버전)
print("\n" + "=" * 60)
print("학습 설정 (강화 버전)")
print("=" * 60)

training_args = TrainingArguments(
    output_dir='./results_price',
    
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    
    fp16=False,
    bf16=False,
    
    learning_rate=3e-5,          # 학습률 약간 증가
    num_train_epochs=15,         # 에폭 5 -> 15 (더 오래 학습)
    weight_decay=0.02,           # 정규화 강화
    
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    
    logging_dir='./logs_price',
    logging_steps=50,
    report_to='none',
    
    save_total_limit=2,
    dataloader_num_workers=0,
    warmup_ratio=0.1,            # 전체의 10% warmup
    lr_scheduler_type='cosine',  # 코사인 스케줄러 (더 부드러운 학습)
    save_safetensors=False,
)

print(f"효과적 배치 사이즈: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"학습 에폭: {training_args.num_train_epochs}")
print(f"학습률: {training_args.learning_rate}")
print(f"스케줄러: {training_args.lr_scheduler_type}")

# 9. Trainer 생성 (가중치 적용)
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  # patience 증가
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
        target_names=['저가', '중가', '고가']
    ))
    
    # 12. 모델 저장
    print("\n" + "=" * 60)
    print("모델 저장")
    print("=" * 60)
    
    save_dir = './results_price/finetuned_price_predictor'
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 메타데이터 저장
    metadata = {
        'price_to_label': price_to_label,
        'label_to_price': label_to_price,
        'num_labels': num_labels,
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'model_name': model_name,
        'max_length': MAX_LENGTH,
        'price_ranges': {
            '저가': '3만원 미만',
            '중가': '3만원 ~ 10만원',
            '고가': '10만원 이상'
        }
    }
    
    with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장 완료: {save_dir}")
    print("\n다음 단계: python 15_use_price_predictor.py 실행")
    
except KeyboardInterrupt:
    print("\n학습이 중단되었습니다.")
except Exception as e:
    print(f"\n[ERROR] 학습 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
