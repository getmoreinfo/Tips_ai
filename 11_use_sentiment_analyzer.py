# 11_use_sentiment_analyzer.py
# 역할: 학습된 감성 분석 모델로 제품 리뷰 감성 예측

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    """제품 리뷰 감성 분석기"""
    
    def __init__(self, model_dir='./results_sentiment/finetuned_sentiment_analyzer'):
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
        
        self.label_to_sentiment = {int(k): v for k, v in self.metadata['label_to_sentiment'].items()}
        
        print("모델 로드 완료!")
    
    def analyze(self, product_name, review_tags=None, top_k=3):
        """
        제품명과 리뷰 태그로 감성 분석
        
        Args:
            product_name: 상품명
            review_tags: 리뷰 태그 리스트 (예: ['만족', '추천', '좋아요'])
            top_k: 상위 k개 예측 반환
        """
        # 텍스트 구성
        if review_tags:
            if isinstance(review_tags, list):
                review_text = ' '.join(review_tags)
            else:
                review_text = review_tags
            text = f"{product_name} | {review_text}"
        else:
            text = product_name
        
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
            sentiment = self.label_to_sentiment.get(int(idx), f"Unknown_{idx}")
            results.append({
                'sentiment': sentiment,
                'confidence': float(prob)
            })
        
        return {
            'product_name': product_name,
            'review_tags': review_tags,
            'predictions': results,
            'top_sentiment': results[0]['sentiment'] if results else None,
            'top_confidence': results[0]['confidence'] if results else 0
        }
    
    def analyze_batch(self, products):
        """
        여러 제품 일괄 분석
        
        Args:
            products: [(product_name, review_tags), ...] 리스트
        """
        results = []
        for product_name, review_tags in products:
            result = self.analyze(product_name, review_tags)
            results.append(result)
        return results


def main():
    print("=" * 60)
    print("리뷰 감성 분석기 테스트")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = SentimentAnalyzer()
    
    # 테스트 데이터
    test_products = [
        ("에르고베이비 옴니 브리즈 아기띠", ['만족', '추천', '편안', '좋아요', '최고']),
        ("그린키즈 이솝우화 동화책 세트", ['아이', '재미', '그림', '만족', '추천']),
        ("폴레드 3D 유아 카시트 보호매트", ['카시트', '설치', '만족', '보호', '깔끔']),
        ("하기스 기저귀 점보팩", ['아기', '가격', '저렴', '만족', '추천']),
        ("레고 시티 경찰서", ['아이', '재미', '조립', '선물', '만족']),
        ("불량 제품 테스트", ['불편', '불만', '교환', '환불', '최악']),
        ("보통 제품 테스트", ['그냥', '보통', '무난', '평범']),
    ]
    
    print("\n" + "=" * 60)
    print("감성 분석 결과")
    print("=" * 60)
    
    for product_name, review_tags in test_products:
        result = analyzer.analyze(product_name, review_tags)
        
        print(f"\n{'─' * 50}")
        print(f"상품명: {result['product_name']}")
        print(f"리뷰 태그: {result['review_tags']}")
        print(f"\n감성 분석 결과:")
        
        for pred in result['predictions']:
            emoji = {'긍정': '😊', '보통': '😐', '부정': '😞'}.get(pred['sentiment'], '❓')
            print(f"  {emoji} {pred['sentiment']}: {pred['confidence']*100:.1f}%")
        
        print(f"\n최종 판정: {result['top_sentiment']} ({result['top_confidence']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
