# 08_train_hierarchical_classifier.py
# 역할: 계층적 카테고리 분류 모델 학습 (대분류 → 중분류 → 소분류)

import torch
import pandas as pd
import numpy as np
import json
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
from sklearn.metrics import accuracy_score, f1_score

# GPU 확인
print("=" * 60)
print("GPU 상태 확인")
print("=" * 60)
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()

# 설정
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LENGTH = 128
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 3e-5
EPOCHS = 7

def train_classifier(df, target_column, level_name, output_dir):
    """특정 레벨의 분류기 학습"""
    
    print("\n" + "=" * 60)
    print(f"{level_name} 분류기 학습")
    print("=" * 60)
    
    # 결측값 제거
    df_clean = df.dropna(subset=[target_column, 'name']).copy()
    df_clean['manufacturer'] = df_clean['manufacturer'].fillna('')
    df_clean['text'] = df_clean['name'] + ' | ' + df_clean['manufacturer']
    
    print(f"학습 데이터: {len(df_clean):,}개")
    print(f"클래스 수: {df_clean[target_column].nunique()}개")
    
    # 클래스별 샘플 수
    class_counts = df_clean[target_column].value_counts()
    print(f"최소 샘플: {class_counts.min()}개, 최대 샘플: {class_counts.max()}개")
    
    # Train/Val/Test 분할
    train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # Dataset 변환
    train_dataset = Dataset.from_pandas(train_df[['text', target_column]].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', target_column]].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['text', target_column]].reset_index(drop=True))
    
    # 토크나이저 및 모델 로드
    num_labels = df_clean[target_column].nunique()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    model.gradient_checkpointing_enable()
    
    # 토크나이징
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LENGTH
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # 레이블 매핑
    class_names = sorted([c for c in df_clean[target_column].unique() if pd.notna(c)])
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    label_to_class = {idx: name for name, idx in class_to_label.items()}
    
    def convert_labels(examples):
        examples['labels'] = [class_to_label[c] for c in examples[target_column]]
        return examples
    
    train_dataset = train_dataset.map(convert_labels, batched=True)
    val_dataset = val_dataset.map(convert_labels, batched=True)
    test_dataset = test_dataset.map(convert_labels, batched=True)
    
    # 평가 메트릭
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=f'./results_hierarchical/{level_name}',
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        fp16=False,  # CUDA 오류 방지
        bf16=False,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_dir=f'./logs_hierarchical/{level_name}',
        logging_steps=50,
        report_to='none',
        save_total_limit=1,
        dataloader_num_workers=0,
        warmup_steps=100,
        save_safetensors=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 학습
    print(f"\n{level_name} 학습 시작...")
    trainer.train()
    
    # 테스트 평가
    test_results = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_results.predictions, axis=1)
    test_accuracy = accuracy_score(test_results.label_ids, test_predictions)
    test_f1 = f1_score(test_results.label_ids, test_predictions, average='weighted')
    
    print(f"\n{level_name} 테스트 결과:")
    print(f"  Accuracy: {test_accuracy*100:.2f}%")
    print(f"  F1 Score: {test_f1:.4f}")
    
    # 모델 저장
    save_dir = f'{output_dir}/{level_name}'
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 메타데이터 저장
    metadata = {
        'level_name': level_name,
        'target_column': target_column,
        'class_to_label': class_to_label,
        'label_to_class': label_to_class,
        'num_classes': num_labels,
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
    }
    
    with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장: {save_dir}")
    
    # 메모리 정리
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return test_accuracy, test_f1, class_to_label, label_to_class


def main():
    # 데이터 로드
    print("=" * 60)
    print("계층적 카테고리 분류 모델 학습")
    print("=" * 60)
    
    # 분리된 카테고리 CSV 확인
    if not os.path.exists('products_all_categorized.csv'):
        print("products_all_categorized.csv 파일이 없습니다.")
        print("먼저 python 07_split_category_levels.py 를 실행하세요.")
        return
    
    df = pd.read_csv('products_all_categorized.csv')
    print(f"데이터 로드 완료: {len(df):,}개")
    
    output_dir = './results_hierarchical/models'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 대분류(Level 1)는 클래스가 1개뿐이므로 스킵
    print("\n[참고] 대분류(Level 1)는 '식품/유아/완구' 1개뿐이므로 학습 스킵")
    results['level1'] = {'accuracy': 1.0, 'f1': 1.0, 'skipped': True}
    
    # 1. 중분류 (Level 2) 학습 - 5개 클래스
    acc2, f1_2, c2l_2, l2c_2 = train_classifier(
        df, 'category_level2', 'level2_classifier', output_dir
    )
    results['level2'] = {'accuracy': acc2, 'f1': f1_2}
    
    # 2. 소분류 (Level 3) 학습 - 35개 클래스
    acc3, f1_3, c2l_3, l2c_3 = train_classifier(
        df, 'category_level3', 'level3_classifier', output_dir
    )
    results['level3'] = {'accuracy': acc3, 'f1': f1_3}
    
    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("최종 학습 결과 요약")
    print("=" * 60)
    print(f"\n대분류 (Level 1): 스킵 (클래스 1개)")
    print(f"중분류 (Level 2): Accuracy {results['level2']['accuracy']*100:.2f}%, F1 {results['level2']['f1']:.4f}")
    print(f"소분류 (Level 3): Accuracy {results['level3']['accuracy']*100:.2f}%, F1 {results['level3']['f1']:.4f}")
    
    # 전체 결과 저장
    with open(f'{output_dir}/training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n학습 결과 저장: {output_dir}/training_summary.json")
    print("\n다음 단계: python 09_use_hierarchical_classifier.py 실행")


if __name__ == '__main__':
    main()
