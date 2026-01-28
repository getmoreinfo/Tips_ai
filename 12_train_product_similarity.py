# 12_train_product_similarity.py
# 역할: 상품 임베딩 모델 학습 - 유사 제품 매칭용

import torch
import pandas as pd
import numpy as np
import json
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

print("=" * 60)
print("유사 제품 매칭 모델 생성")
print("=" * 60)
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
print()

# 1. 데이터 로드
print("=" * 60)
print("데이터 로드")
print("=" * 60)

df = pd.read_csv('products_all_categorized.csv')

# 필요한 컬럼 선택 및 결측값 처리
df = df[['id', 'name', 'manufacturer', 'category', 'category_level1', 
         'category_level2', 'category_level3', 'min_price', 'max_price', 
         'average_rating', 'review_count']].copy()

df['manufacturer'] = df['manufacturer'].fillna('')
df['average_rating'] = df['average_rating'].fillna(0)
df['review_count'] = df['review_count'].fillna(0)

# 검색용 텍스트 생성: 상품명 + 제조사 + 카테고리
df['search_text'] = df['name'] + ' | ' + df['manufacturer'] + ' | ' + df['category_level3'].fillna('')

print(f"총 상품 수: {len(df):,}개")
print(f"카테고리 수: {df['category_level3'].nunique()}개")
print()

# 2. 임베딩 모델 로드
print("=" * 60)
print("임베딩 모델 로드")
print("=" * 60)

# 한국어에 최적화된 문장 임베딩 모델
model_name = "jhgan/ko-sroberta-multitask"  # 한국어 Sentence-BERT
print(f"모델: {model_name}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(model_name, device=device)
print(f"디바이스: {device}")
print("모델 로드 완료!")
print()

# 3. 상품 임베딩 생성
print("=" * 60)
print("상품 임베딩 생성")
print("=" * 60)

texts = df['search_text'].tolist()
print(f"임베딩 생성 중... ({len(texts):,}개 상품)")

# 배치로 임베딩 생성 (메모리 효율)
batch_size = 64
embeddings = model.encode(
    texts, 
    batch_size=batch_size, 
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"임베딩 shape: {embeddings.shape}")
print(f"임베딩 차원: {embeddings.shape[1]}")
print()

# 4. 임베딩 저장
print("=" * 60)
print("임베딩 저장")
print("=" * 60)

save_dir = './results_similarity'
os.makedirs(save_dir, exist_ok=True)

# 임베딩 저장
np.save(f'{save_dir}/product_embeddings.npy', embeddings)

# 상품 정보 저장 (검색용)
product_info = df[['id', 'name', 'manufacturer', 'category', 'category_level2', 
                   'category_level3', 'min_price', 'average_rating', 'review_count']].to_dict('records')

with open(f'{save_dir}/product_info.pkl', 'wb') as f:
    pickle.dump(product_info, f)

# 메타데이터 저장
metadata = {
    'model_name': model_name,
    'embedding_dim': int(embeddings.shape[1]),
    'num_products': len(df),
    'categories': df['category_level3'].unique().tolist(),
}

with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"임베딩 저장: {save_dir}/product_embeddings.npy")
print(f"상품 정보 저장: {save_dir}/product_info.pkl")
print()

# 5. 유사도 테스트
print("=" * 60)
print("유사 제품 검색 테스트")
print("=" * 60)

def find_similar_products(query_idx, embeddings, product_info, top_k=5):
    """특정 상품과 유사한 상품 찾기"""
    query_embedding = embeddings[query_idx].reshape(1, -1)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # 자기 자신 제외하고 상위 k개
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = []
    for idx in similar_indices:
        results.append({
            'product': product_info[idx],
            'similarity': float(similarities[idx])
        })
    
    return results

def search_similar_by_text(query_text, model, embeddings, product_info, top_k=5):
    """텍스트로 유사한 상품 검색"""
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # 상위 k개
    similar_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in similar_indices:
        results.append({
            'product': product_info[idx],
            'similarity': float(similarities[idx])
        })
    
    return results

# 테스트 1: 특정 상품과 유사한 상품 찾기
print("\n[테스트 1] 첫 번째 상품과 유사한 상품")
print(f"기준 상품: {product_info[0]['name'][:50]}...")
print(f"제조사: {product_info[0]['manufacturer']}")
print(f"카테고리: {product_info[0]['category_level3']}")
print("\n유사 상품:")

similar = find_similar_products(0, embeddings, product_info, top_k=5)
for i, item in enumerate(similar, 1):
    p = item['product']
    print(f"  {i}. {p['name'][:40]}... ({item['similarity']*100:.1f}%)")
    print(f"     제조사: {p['manufacturer']}, 카테고리: {p['category_level3']}")

# 테스트 2: 텍스트로 검색
print("\n" + "-" * 50)
print("[테스트 2] '아기띠' 검색")
results = search_similar_by_text("아기띠", model, embeddings, product_info, top_k=5)
for i, item in enumerate(results, 1):
    p = item['product']
    print(f"  {i}. {p['name'][:40]}... ({item['similarity']*100:.1f}%)")
    print(f"     제조사: {p['manufacturer']}, 가격: {p['min_price']:,.0f}원")

# 테스트 3: 다른 검색어
print("\n" + "-" * 50)
print("[테스트 3] '레고 블럭' 검색")
results = search_similar_by_text("레고 블럭", model, embeddings, product_info, top_k=5)
for i, item in enumerate(results, 1):
    p = item['product']
    print(f"  {i}. {p['name'][:40]}... ({item['similarity']*100:.1f}%)")
    print(f"     제조사: {p['manufacturer']}, 가격: {p['min_price']:,.0f}원")

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print(f"저장 위치: {save_dir}")
print("\n다음 단계: python 13_use_product_similarity.py 실행")
