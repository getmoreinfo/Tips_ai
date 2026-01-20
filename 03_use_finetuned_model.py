# 03_use_finetuned_model.py
# 역할: Fine-tuning된 모델로 샘플 데이터 테스트

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd

class FinetunedCategoryMapper:
    """Fine-tuning된 모델을 사용한 카테고리 매퍼"""
    
    def __init__(self, model_path='./results/finetuned_e5_large'):
        print("=" * 60)
        print("모델 로딩")
        print("=" * 60)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        with open(f'{model_path}/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            self.label_to_category = {int(k): v for k, v in self.metadata['label_to_category'].items()}
            self.max_length = self.metadata.get('max_length', 128)
        
        print(f"모델 로드 완료")
        print(f"Device: {self.device}")
        print(f"Test Accuracy: {self.metadata['test_accuracy']*100:.2f}%")
        print(f"카테고리 수: {self.metadata['num_labels']}개")
        print("=" * 60)
    
    def predict(self, product_text, top_k=3):
        """단일 상품 예측"""
        inputs = self.tokenizer(
            product_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            category_id = self.label_to_category[idx.item()]
            confidence = prob.item()
            results.append((category_id, confidence))
        
        return results
    
    def predict_batch(self, product_texts, batch_size=32):
        """배치 예측"""
        results = []
        
        for i in range(0, len(product_texts), batch_size):
            batch = product_texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            top_probs, top_indices = torch.max(probs, dim=1)
            
            for prob, idx in zip(top_probs, top_indices):
                category_id = self.label_to_category[idx.item()]
                confidence = prob.item()
                results.append((category_id, confidence))
        
        return results


def test_with_training_data(sample_size=100):
    """
    training_data.csv에서 샘플 추출하여 테스트
    """
    print("\n" + "=" * 60)
    print("학습 데이터 기반 테스트")
    print("=" * 60)
    
    # CSV 로드 (학습에 사용한 동일한 데이터셋 사용)
    df = pd.read_csv('training_data_10000.csv')
    print(f"전체 데이터: {len(df):,}개")
    
    # 샘플 추출 (각 카테고리에서 골고루)
    test_df = df.groupby('category_name').apply(
        lambda x: x.sample(n=min(sample_size, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    print(f"테스트 데이터: {len(test_df):,}개")
    print()
    
    # 모델 로드
    mapper = FinetunedCategoryMapper()
    
    # 배치 예측
    print("\n" + "=" * 60)
    print("예측 중...")
    print("=" * 60)
    
    predictions = mapper.predict_batch(test_df['text'].tolist(), batch_size=32)
    
    # 정확도 계산
    correct = 0
    total = len(test_df)
    
    for i, (pred_cat, confidence) in enumerate(predictions):
        actual_cat = test_df.iloc[i]['category_name']
        if pred_cat == actual_cat:
            correct += 1
    
    accuracy = correct / total * 100
    
    print(f"\n정확도: {correct} / {total} = {accuracy:.2f}%")
    
    # 카테고리별 정확도
    print("\n" + "=" * 60)
    print("카테고리별 정확도")
    print("=" * 60)
    
    category_stats = {}
    for i, (pred_cat, confidence) in enumerate(predictions):
        actual_cat = test_df.iloc[i]['category_name']
        
        if actual_cat not in category_stats:
            category_stats[actual_cat] = {
                'total': 0,
                'correct': 0
            }
        
        category_stats[actual_cat]['total'] += 1
        if pred_cat == actual_cat:
            category_stats[actual_cat]['correct'] += 1
    
    for cat_name in sorted(category_stats.keys()):
        stats = category_stats[cat_name]
        cat_accuracy = stats['correct'] / stats['total'] * 100
        print(f"{cat_name[:50]}: {stats['correct']}/{stats['total']} ({cat_accuracy:.1f}%)")
    
    # 오분류 사례 출력 (상위 10개)
    print("\n" + "=" * 60)
    print("오분류 사례 (상위 10개)")
    print("=" * 60)
    
    errors = []
    for i, (pred_cat, confidence) in enumerate(predictions):
        actual_cat = test_df.iloc[i]['category_name']
        if pred_cat != actual_cat:
            errors.append({
                'text': test_df.iloc[i]['text'][:60],
                'actual_name': actual_cat,
                'predicted': pred_cat,
                'confidence': confidence
            })
    
    for i, error in enumerate(errors[:10], 1):
        print(f"\n{i}. {error['text']}...")
        print(f"   정답: {error['actual_name'][:50]}")
        print(f"   예측: {error['predicted'][:50]} (신뢰도: {error['confidence']*100:.1f}%)")
    
    if len(errors) > 10:
        print(f"\n... 외 {len(errors)-10}개 오분류")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    # training_data.csv 기반 테스트 (각 카테고리에서 100개씩)
    test_with_training_data(sample_size=100)