# 17_use_keyword_extractor.py
# 역할: 학습된 키워드 추출기 사용

import torch
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("키워드 추출기 사용")
print("=" * 60)

class KeywordExtractor:
    def __init__(self, model_dir='./results_keyword'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        # 메타데이터 로드
        with open(f'{model_dir}/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 도메인 키워드 로드
        with open(f'{model_dir}/domain_keywords.json', 'r', encoding='utf-8') as f:
            self.domain_keywords = json.load(f)
        
        # 키워드 임베딩 로드
        self.keyword_embeddings = np.load(f'{model_dir}/keyword_embeddings.npy')
        
        # 모델 로드
        print(f"모델 로드 중: {self.metadata['model_name']}")
        self.model = SentenceTransformer(self.metadata['model_name'], device=self.device)
        
        print(f"도메인 키워드: {len(self.domain_keywords)}개")
        print(f"모델 성능 - F1: {self.metadata['avg_f1']*100:.2f}%")
    
    def extract_keywords(self, text, top_n=5, diversity=0.5):
        """텍스트에서 키워드 추출"""
        if not text:
            return []
        
        # 텍스트 임베딩
        text_embedding = self.model.encode([text])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(text_embedding, self.keyword_embeddings)[0]
        
        # MMR로 다양성 확보
        keywords_idx = []
        candidates = list(range(len(self.domain_keywords)))
        
        for _ in range(min(top_n, len(candidates))):
            if not candidates:
                break
            
            if not keywords_idx:
                best_idx = max(candidates, key=lambda x: similarities[x])
            else:
                mmr_scores = []
                for idx in candidates:
                    relevance = similarities[idx]
                    selected_embeddings = self.keyword_embeddings[keywords_idx]
                    candidate_embedding = self.keyword_embeddings[idx:idx+1]
                    redundancy = max(cosine_similarity(candidate_embedding, selected_embeddings)[0])
                    mmr = diversity * relevance - (1 - diversity) * redundancy
                    mmr_scores.append((idx, mmr))
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            
            keywords_idx.append(best_idx)
            candidates.remove(best_idx)
        
        results = [(self.domain_keywords[idx], float(similarities[idx])) 
                  for idx in keywords_idx]
        return results
    
    def extract_from_product(self, name, manufacturer='', category='', top_n=5):
        """상품 정보에서 키워드 추출"""
        text = f"{name} {manufacturer} {category}"
        return self.extract_keywords(text, top_n=top_n)
    
    def get_keyword_summary(self, name, manufacturer='', category='', top_n=5):
        """키워드 요약 문자열 반환"""
        keywords = self.extract_from_product(name, manufacturer, category, top_n)
        return ', '.join([kw for kw, score in keywords])

# 테스트
if __name__ == "__main__":
    extractor = KeywordExtractor()
    
    print("\n" + "=" * 60)
    print("키워드 추출 테스트")
    print("=" * 60)
    
    # 테스트 상품들
    test_products = [
        {
            'name': '그린키즈 요술지팡이 이솝우화 세트 20권',
            'manufacturer': '그린키즈',
            'category': '그림/동화/놀이책'
        },
        {
            'name': '남양유업 아이엠마더 분유 1단계 800g',
            'manufacturer': '남양유업',
            'category': '분유'
        },
        {
            'name': '하기스 매직팬티 기저귀 대형 4단계',
            'manufacturer': '하기스',
            'category': '기저귀'
        },
        {
            'name': '레고 듀플로 대형 놀이공원',
            'manufacturer': '레고',
            'category': '블록/레고'
        },
        {
            'name': '보솜이 천연코튼 물티슈 캡형 80매',
            'manufacturer': '보솜이',
            'category': '물티슈'
        }
    ]
    
    for product in test_products:
        print(f"\n{'='*50}")
        print(f"상품명: {product['name']}")
        print(f"제조사: {product['manufacturer']}")
        print(f"카테고리: {product['category']}")
        
        keywords = extractor.extract_from_product(
            product['name'],
            product['manufacturer'],
            product['category'],
            top_n=7
        )
        
        print(f"\n추출된 키워드:")
        for kw, score in keywords:
            print(f"  - {kw} (유사도: {score:.3f})")
        
        print(f"\n요약: {extractor.get_keyword_summary(product['name'], product['manufacturer'], product['category'])}")
    
    # 사용자 입력 테스트
    print("\n" + "=" * 60)
    print("직접 입력 테스트")
    print("=" * 60)
    
    while True:
        print("\n상품 정보를 입력하세요 (종료: q)")
        name = input("상품명: ").strip()
        if name.lower() == 'q':
            break
        
        manufacturer = input("제조사 (선택): ").strip()
        category = input("카테고리 (선택): ").strip()
        
        keywords = extractor.extract_from_product(name, manufacturer, category, top_n=7)
        
        print(f"\n추출된 키워드:")
        for kw, score in keywords:
            print(f"  - {kw} (유사도: {score:.3f})")
