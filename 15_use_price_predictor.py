# 15_use_price_predictor.py
# 역할: 학습된 가격대 예측 모델로 상품 가격 예측

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PricePredictor:
    """상품 가격대 예측기"""
    
    def __init__(self, model_dir='./results_price/finetuned_price_predictor'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"디바이스: {self.device}")
        print("모델 로딩 중...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # 메타데이터 로드
        with open(f'{model_dir}/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.label_to_price = {int(k): v for k, v in self.metadata['label_to_price'].items()}
        self.price_ranges = self.metadata['price_ranges']
        
        print("모델 로드 완료!")
    
    def predict(self, product_name, manufacturer='', category='', top_k=3):
        """
        상품 가격대 예측
        
        Args:
            product_name: 상품명
            manufacturer: 제조사 (선택)
            category: 카테고리 (선택)
            top_k: 상위 k개 예측 반환
        """
        # 텍스트 구성
        text = f"{product_name} | {manufacturer} | {category}"
        
        # 토크나이징
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Top-k 결과
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            price_range = self.label_to_price.get(int(idx), f"Unknown_{idx}")
            results.append({
                'price_range': price_range,
                'price_detail': self.price_ranges.get(price_range, ''),
                'confidence': float(prob)
            })
        
        return {
            'product_name': product_name,
            'manufacturer': manufacturer,
            'category': category,
            'predictions': results,
            'top_prediction': results[0]['price_range'] if results else None,
            'top_confidence': results[0]['confidence'] if results else 0
        }
    
    def analyze_price_position(self, product_name, actual_price, manufacturer='', category=''):
        """
        실제 가격과 예측 가격 비교 분석
        
        Args:
            product_name: 상품명
            actual_price: 실제 가격
            manufacturer: 제조사
            category: 카테고리
        """
        prediction = self.predict(product_name, manufacturer, category)
        
        # 실제 가격대 계산 (3단계)
        if actual_price < 30000:
            actual_range = '저가'
        elif actual_price < 100000:
            actual_range = '중가'
        else:
            actual_range = '고가'
        
        predicted_range = prediction['top_prediction']
        
        # 가격 포지션 분석
        range_order = ['저가', '중가', '고가']
        actual_idx = range_order.index(actual_range)
        predicted_idx = range_order.index(predicted_range)
        
        if actual_idx == predicted_idx:
            position = '적정가'
            analysis = '시장 평균 가격대에 위치합니다.'
        elif actual_idx < predicted_idx:
            position = '저가'
            analysis = f'예상보다 저렴합니다. (예상: {predicted_range}, 실제: {actual_range})'
        else:
            position = '고가'
            analysis = f'예상보다 비쌉니다. (예상: {predicted_range}, 실제: {actual_range})'
        
        return {
            'product_name': product_name,
            'actual_price': actual_price,
            'actual_range': actual_range,
            'predicted_range': predicted_range,
            'confidence': prediction['top_confidence'],
            'position': position,
            'analysis': analysis,
            'all_predictions': prediction['predictions']
        }


def main():
    print("=" * 60)
    print("가격대 예측기 테스트")
    print("=" * 60)
    
    # 예측기 초기화
    predictor = PricePredictor()
    
    # 테스트 상품 목록
    test_products = [
        ("에르고베이비 옴니 브리즈 아기띠", "에르고베이비", "아기띠/외출용품", 259000),
        ("그린키즈 이솝우화 동화책 세트", "그린키즈", "그림/동화/놀이책", 45900),
        ("레고 시티 경찰서", "레고", "레고/블럭", 89000),
        ("하기스 기저귀 점보팩", "하기스", "기저귀", 25000),
        ("베이비뵨 베이비 캐리어", "베이비뵨", "아기띠/외출용품", 306000),
        ("뽀로로 장난감 자동차", "뽀로로", "인기 캐릭터완구", 15000),
        ("유아용 멜라민 식기 세트", "", "이유식용품", 8000),
    ]
    
    print("\n" + "=" * 60)
    print("가격대 예측 결과")
    print("=" * 60)
    
    for product_name, manufacturer, category, actual_price in test_products:
        print(f"\n{'─' * 50}")
        print(f"상품명: {product_name}")
        print(f"제조사: {manufacturer}")
        print(f"카테고리: {category}")
        print(f"실제 가격: {actual_price:,}원")
        
        # 가격 포지션 분석
        analysis = predictor.analyze_price_position(
            product_name, actual_price, manufacturer, category
        )
        
        print(f"\n예측 가격대: {analysis['predicted_range']} ({analysis['confidence']*100:.1f}%)")
        print(f"실제 가격대: {analysis['actual_range']}")
        print(f"가격 포지션: {analysis['position']}")
        print(f"분석: {analysis['analysis']}")
        
        print(f"\nTop-3 예측:")
        for i, pred in enumerate(analysis['all_predictions'], 1):
            print(f"  {i}. {pred['price_range']} ({pred['confidence']*100:.1f}%) - {pred['price_detail']}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
