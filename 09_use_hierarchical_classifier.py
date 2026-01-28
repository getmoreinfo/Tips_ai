# 09_use_hierarchical_classifier.py
# 역할: 학습된 계층적 분류 모델을 사용하여 상품 카테고리 예측

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HierarchicalCategoryClassifier:
    """계층적 카테고리 분류기"""
    
    def __init__(self, models_dir='./results_hierarchical/models'):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"디바이스: {self.device}")
        print("모델 로딩 중...")
        
        # 각 레벨별 모델 로드
        self.classifiers = {}
        
        for level in ['level1', 'level2', 'level3']:
            model_path = f'{models_dir}/{level}_classifier'
            if os.path.exists(model_path):
                self.classifiers[level] = self._load_model(model_path, level)
                print(f"  {level} 모델 로드 완료")
            else:
                print(f"  경고: {level} 모델을 찾을 수 없습니다: {model_path}")
        
        print("모델 로딩 완료!\n")
    
    def _load_model(self, model_path, level_name):
        """개별 모델 로드"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        # 메타데이터 로드
        with open(f'{model_path}/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return {
            'tokenizer': tokenizer,
            'model': model,
            'metadata': metadata,
            'label_to_class': {int(k): v for k, v in metadata['label_to_class'].items()}
        }
    
    def predict(self, product_name, manufacturer='', top_k=3):
        """
        상품명과 제조사로 카테고리 예측
        
        Args:
            product_name: 상품명
            manufacturer: 제조사 (선택)
            top_k: 각 레벨별 상위 k개 예측 반환
            
        Returns:
            dict: 각 레벨별 예측 결과
        """
        text = f"{product_name} | {manufacturer}"
        results = {}
        
        for level in ['level1', 'level2', 'level3']:
            if level not in self.classifiers:
                continue
                
            classifier = self.classifiers[level]
            tokenizer = classifier['tokenizer']
            model = classifier['model']
            label_to_class = classifier['label_to_class']
            
            # 토크나이징
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            # Top-k 결과
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                class_name = label_to_class.get(int(idx), f"Unknown_{idx}")
                predictions.append({
                    'category': class_name,
                    'confidence': float(prob)
                })
            
            results[level] = predictions
        
        return results
    
    def predict_full_category(self, product_name, manufacturer=''):
        """
        전체 카테고리 경로 예측 (대분류 > 중분류 > 소분류)
        
        Returns:
            dict: 전체 카테고리 경로와 신뢰도
        """
        predictions = self.predict(product_name, manufacturer, top_k=1)
        
        result = {
            'product_name': product_name,
            'manufacturer': manufacturer,
            'level1': None,
            'level2': None,
            'level3': None,
            'full_category': None,
            'confidences': {}
        }
        
        parts = []
        for level in ['level1', 'level2', 'level3']:
            if level in predictions and predictions[level]:
                pred = predictions[level][0]
                result[level] = pred['category']
                result['confidences'][level] = pred['confidence']
                parts.append(pred['category'])
        
        if parts:
            result['full_category'] = ' > '.join(parts)
        
        return result


def main():
    print("=" * 60)
    print("계층적 카테고리 분류기 테스트")
    print("=" * 60)
    
    # 분류기 초기화
    classifier = HierarchicalCategoryClassifier()
    
    # 테스트 상품 목록
    test_products = [
        ("에르고베이비 옴니 브리즈 아기띠", "에르고베이비"),
        ("뽀로로 장난감 자동차", "뽀로로"),
        ("하기스 기저귀 점보팩", "하기스"),
        ("그린키즈 이솝우화 동화책 세트", "그린키즈"),
        ("베이비뵨 베이비 캐리어 하모니", "베이비뵨"),
        ("레고 시티 경찰서", "레고"),
        ("피셔프라이스 아기 체육관", "피셔프라이스"),
        ("유아용 멜라민 식기 세트", ""),
        ("신생아 손싸개 발싸개 세트", ""),
        ("보행기 점퍼루 쏘서", ""),
    ]
    
    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    
    for product_name, manufacturer in test_products:
        print(f"\n{'─' * 50}")
        print(f"상품명: {product_name}")
        if manufacturer:
            print(f"제조사: {manufacturer}")
        
        # 전체 카테고리 예측
        result = classifier.predict_full_category(product_name, manufacturer)
        
        print(f"\n예측 카테고리:")
        print(f"  대분류: {result['level1']} ({result['confidences'].get('level1', 0)*100:.1f}%)")
        print(f"  중분류: {result['level2']} ({result['confidences'].get('level2', 0)*100:.1f}%)")
        print(f"  소분류: {result['level3']} ({result['confidences'].get('level3', 0)*100:.1f}%)")
        print(f"\n전체 경로: {result['full_category']}")
        
        # Top-3 예측 (상세)
        detailed = classifier.predict(product_name, manufacturer, top_k=3)
        print(f"\nTop-3 예측:")
        for level in ['level1', 'level2', 'level3']:
            if level in detailed:
                level_name = {'level1': '대분류', 'level2': '중분류', 'level3': '소분류'}[level]
                print(f"  [{level_name}]")
                for i, pred in enumerate(detailed[level], 1):
                    print(f"    {i}. {pred['category']} ({pred['confidence']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
