# 05_save_trained_model.py
# 역할: 학습된 체크포인트에서 최종 모델 저장 및 테스트

import torch
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

print("=" * 60)
print("학습된 모델 저장 및 테스트")
print("=" * 60)

# 1. 데이터 로드 (레이블 매핑 생성용)
print("\n데이터 로드 중...")
df = pd.read_csv('products_all.csv')
df = df[['id', 'category_id', 'category', 'name', 'manufacturer']].copy()
df = df.dropna(subset=['category', 'name'])
df['manufacturer'] = df['manufacturer'].fillna('')
df['text'] = df['name'] + ' | ' + df['manufacturer']

# 레이블 매핑 생성
category_names = sorted([cat for cat in df['category'].unique() if pd.notna(cat)])
category_to_label = {cat_name: idx for idx, cat_name in enumerate(category_names)}
label_to_category = {idx: cat_name for cat_name, idx in category_to_label.items()}
num_labels = len(category_names)

print(f"카테고리 수: {num_labels}개")

# 2. 테스트 데이터 분할 (동일한 random_state로)
_, temp_df = train_test_split(df, test_size=0.2, random_state=42)
_, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f"테스트 데이터: {len(test_df):,}개")

# 3. 체크포인트에서 모델 로드
checkpoint_path = './results_category/checkpoint-1988'  # 최신 체크포인트
print(f"\n체크포인트 로드: {checkpoint_path}")

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"Device: {device}")

# 4. 테스트 데이터 평가
print("\n" + "=" * 60)
print("테스트 평가")
print("=" * 60)

correct = 0
total = len(test_df)
predictions_list = []

# 배치 처리
batch_size = 32
test_texts = test_df['text'].tolist()
test_categories = test_df['category'].tolist()

for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i+batch_size]
    batch_categories = test_categories[i:i+batch_size]
    
    inputs = tokenizer(
        batch_texts,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    for pred, actual_cat in zip(preds, batch_categories):
        pred_cat = label_to_category[pred]
        predictions_list.append(pred_cat)
        if pred_cat == actual_cat:
            correct += 1

test_accuracy = correct / total
test_f1 = f1_score(
    [category_to_label[c] for c in test_categories],
    [category_to_label[c] for c in predictions_list],
    average='weighted'
)

print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test F1 Score: {test_f1:.4f}")

# 5. 최종 모델 저장
print("\n" + "=" * 60)
print("모델 저장")
print("=" * 60)

save_dir = './results_category/finetuned_category_classifier'
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 메타데이터 저장
metadata = {
    'category_to_label': category_to_label,
    'label_to_category': label_to_category,
    'num_labels': num_labels,
    'training_samples': len(df) - len(test_df) - len(test_df),  # 대략적인 값
    'test_samples': len(test_df),
    'test_accuracy': float(test_accuracy),
    'test_f1': float(test_f1),
    'model_name': 'intfloat/multilingual-e5-base',
    'max_length': 128,
    'epochs': 3,
    'total_categories': len(category_names),
}

with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"모델 저장 완료: {save_dir}")

print("\n" + "=" * 60)
print("완료")
print("=" * 60)
print("다음 단계: python 06_use_category_classifier.py 실행")
