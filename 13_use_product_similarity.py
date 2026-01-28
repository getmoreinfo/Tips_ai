# 13_use_product_similarity.py
# 역할: 유사 제품 검색 및 경쟁 제품 분석

import torch
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ProductSimilaritySearch:
    """유사 제품 검색 엔진"""
    
    def __init__(self, data_dir='./results_similarity'):
        print("유사 제품 검색 엔진 초기화...")
        
        # 메타데이터 로드
        with open(f'{data_dir}/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 임베딩 로드
        self.embeddings = np.load(f'{data_dir}/product_embeddings.npy')
        
        # 상품 정보 로드
        with open(f'{data_dir}/product_info.pkl', 'rb') as f:
            self.product_info = pickle.load(f)
        
        # 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(self.metadata['model_name'], device=self.device)
        
        print(f"디바이스: {self.device}")
        print(f"상품 수: {len(self.product_info):,}개")
        print(f"임베딩 차원: {self.embeddings.shape[1]}")
        print("초기화 완료!\n")
    
    def search_by_text(self, query, top_k=10, category_filter=None):
        """
        텍스트로 유사 상품 검색
        
        Args:
            query: 검색어 (상품명, 키워드 등)
            top_k: 반환할 상품 수
            category_filter: 특정 카테고리만 검색 (선택)
        """
        # 쿼리 임베딩
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 카테고리 필터 적용
        if category_filter:
            for i, p in enumerate(self.product_info):
                if p['category_level3'] != category_filter and p['category_level2'] != category_filter:
                    similarities[i] = -1
        
        # 상위 k개
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in similar_indices:
            if similarities[idx] < 0:
                continue
            results.append({
                'id': self.product_info[idx]['id'],
                'name': self.product_info[idx]['name'],
                'manufacturer': self.product_info[idx]['manufacturer'],
                'category': self.product_info[idx]['category_level3'],
                'price': self.product_info[idx]['min_price'],
                'rating': self.product_info[idx]['average_rating'],
                'reviews': self.product_info[idx]['review_count'],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def find_competitors(self, product_name, manufacturer='', top_k=10):
        """
        경쟁 제품 찾기 (같은 카테고리 내 유사 제품)
        
        Args:
            product_name: 기준 상품명
            manufacturer: 제조사 (선택)
            top_k: 반환할 경쟁 제품 수
        """
        query = f"{product_name} | {manufacturer}" if manufacturer else product_name
        
        # 먼저 기준 상품 찾기
        results = self.search_by_text(query, top_k=1)
        if not results:
            return {'error': '상품을 찾을 수 없습니다.'}
        
        base_product = results[0]
        base_category = base_product['category']
        
        # 같은 카테고리에서 유사 제품 검색
        competitors = self.search_by_text(
            query, 
            top_k=top_k + 1,  # 자기 자신 포함
            category_filter=base_category
        )
        
        # 자기 자신 제외
        competitors = [c for c in competitors if c['name'] != base_product['name']][:top_k]
        
        return {
            'base_product': base_product,
            'category': base_category,
            'competitors': competitors
        }
    
    def analyze_brand(self, brand_name, top_k=20):
        """
        브랜드 분석 - 해당 브랜드의 모든 제품과 경쟁 현황
        
        Args:
            brand_name: 브랜드/제조사명
            top_k: 분석할 제품 수
        """
        # 해당 브랜드 제품 찾기
        brand_products = [
            p for p in self.product_info 
            if brand_name.lower() in p['manufacturer'].lower()
        ]
        
        if not brand_products:
            return {'error': f'"{brand_name}" 브랜드를 찾을 수 없습니다.'}
        
        # 카테고리별 분류
        categories = {}
        for p in brand_products:
            cat = p['category_level3']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(p)
        
        # 가격 및 평점 통계
        prices = [p['min_price'] for p in brand_products if p['min_price'] and p['min_price'] > 0]
        ratings = [p['average_rating'] for p in brand_products if p['average_rating'] and p['average_rating'] > 0]
        
        return {
            'brand': brand_name,
            'total_products': len(brand_products),
            'categories': {k: len(v) for k, v in categories.items()},
            'price_stats': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'avg': sum(prices) / len(prices) if prices else 0
            },
            'rating_stats': {
                'avg': sum(ratings) / len(ratings) if ratings else 0,
                'count': len(ratings)
            },
            'top_products': sorted(brand_products, key=lambda x: x['review_count'] or 0, reverse=True)[:5]
        }


def main():
    print("=" * 60)
    print("유사 제품 검색 및 경쟁 분석")
    print("=" * 60)
    
    # 검색 엔진 초기화
    search = ProductSimilaritySearch()
    
    # 테스트 1: 텍스트 검색
    print("=" * 60)
    print("[테스트 1] '에르고베이비 아기띠' 검색")
    print("=" * 60)
    
    results = search.search_by_text("에르고베이비 아기띠", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['name'][:50]}")
        print(f"   제조사: {r['manufacturer']}")
        print(f"   카테고리: {r['category']}")
        print(f"   가격: {r['price']:,.0f}원" if r['price'] else "   가격: 정보없음")
        print(f"   평점: {r['rating']:.1f} ({r['reviews']:.0f}개 리뷰)" if r['rating'] else "   평점: 정보없음")
        print(f"   유사도: {r['similarity']*100:.1f}%")
    
    # 테스트 2: 경쟁 제품 분석
    print("\n" + "=" * 60)
    print("[테스트 2] 경쟁 제품 분석")
    print("=" * 60)
    
    analysis = search.find_competitors("에르고베이비 옴니 브리즈 아기띠", "에르고베이비", top_k=5)
    
    if 'error' not in analysis:
        base = analysis['base_product']
        print(f"\n기준 상품: {base['name'][:50]}")
        print(f"카테고리: {analysis['category']}")
        print(f"가격: {base['price']:,.0f}원" if base['price'] else "가격: 정보없음")
        
        print(f"\n경쟁 제품 ({len(analysis['competitors'])}개):")
        for i, c in enumerate(analysis['competitors'], 1):
            print(f"\n  {i}. {c['name'][:45]}")
            print(f"     제조사: {c['manufacturer']}")
            print(f"     가격: {c['price']:,.0f}원" if c['price'] else "     가격: 정보없음")
            print(f"     유사도: {c['similarity']*100:.1f}%")
    
    # 테스트 3: 브랜드 분석
    print("\n" + "=" * 60)
    print("[테스트 3] 브랜드 분석 - '에르고베이비'")
    print("=" * 60)
    
    brand_analysis = search.analyze_brand("에르고베이비")
    
    if 'error' not in brand_analysis:
        print(f"\n브랜드: {brand_analysis['brand']}")
        print(f"총 제품 수: {brand_analysis['total_products']}개")
        
        print(f"\n카테고리별 제품 수:")
        for cat, count in brand_analysis['categories'].items():
            print(f"  - {cat}: {count}개")
        
        stats = brand_analysis['price_stats']
        print(f"\n가격 범위: {stats['min']:,.0f}원 ~ {stats['max']:,.0f}원")
        print(f"평균 가격: {stats['avg']:,.0f}원")
        
        rating = brand_analysis['rating_stats']
        print(f"평균 평점: {rating['avg']:.2f} ({rating['count']}개 제품)")
        
        print(f"\n인기 제품 Top 5:")
        for i, p in enumerate(brand_analysis['top_products'], 1):
            print(f"  {i}. {p['name'][:40]}... (리뷰 {p['review_count']:.0f}개)")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
