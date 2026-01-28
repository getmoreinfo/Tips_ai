# 16_train_keyword_extractor.py
# 역할: 상품명/설명에서 핵심 키워드 추출 모델 학습

import torch
import pandas as pd
import numpy as np
import json
import os
import re
import ast
from collections import Counter
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

print("=" * 60)
print("키워드 추출 모델 학습")
print("=" * 60)
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
print()

# 1. 데이터 로드
print("=" * 60)
print("데이터 로드 및 전처리")
print("=" * 60)

df = pd.read_csv('products_all_categorized.csv')
print(f"전체 데이터: {len(df):,}개")

# specifications에서 키워드 추출
def parse_specifications(spec_str):
    """specifications 컬럼에서 키워드 추출"""
    if pd.isna(spec_str):
        return []
    
    try:
        spec_dict = ast.literal_eval(spec_str)
        keywords = []
        
        for key, value in spec_dict.items():
            # 키 자체가 키워드
            if key not in ['KC인증', 'manufacturer', '등록년월', '제조회사']:
                keywords.append(key)
            
            # 값이 딕셔너리인 경우
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_value == True:
                        keywords.append(sub_key)
                    elif isinstance(sub_value, str) and len(sub_value) < 30:
                        keywords.append(sub_value)
            elif isinstance(value, str) and len(value) < 30:
                keywords.append(value)
        
        return keywords
    except:
        return []

# review_tags에서 키워드 추출
def parse_review_tags(tags_str):
    """review_tags 컬럼에서 키워드 추출"""
    if pd.isna(tags_str) or tags_str == '[]':
        return []
    
    try:
        tags = ast.literal_eval(tags_str)
        return tags if isinstance(tags, list) else []
    except:
        return []

# 키워드 추출
df['spec_keywords'] = df['specifications'].apply(parse_specifications)
df['review_keywords'] = df['review_tags'].apply(parse_review_tags)

# 모든 키워드 통합
df['all_keywords'] = df['spec_keywords'] + df['review_keywords']

# 키워드가 있는 데이터만 선택
df_with_keywords = df[df['all_keywords'].apply(len) > 0].copy()
print(f"키워드가 있는 데이터: {len(df_with_keywords):,}개")

# 키워드 빈도 분석
all_keywords_flat = [kw for kws in df_with_keywords['all_keywords'] for kw in kws]
keyword_counts = Counter(all_keywords_flat)
print(f"\n총 고유 키워드 (정제 전): {len(keyword_counts):,}개")

# ========== 키워드 정제 ==========
print("\n" + "=" * 60)
print("키워드 정제")
print("=" * 60)

def is_valid_keyword(kw):
    """유효한 키워드인지 판단"""
    # 제외할 패턴들
    exclude_patterns = [
        '상세설명', '판매 사이트', '문의', '인증번호', '확인',
        '년 ', '월', '등록', 'KC인증', '제조회사', 'manufacturer',
        '/', '>', '<'  # 카테고리 구분자
    ]
    
    # 제외 패턴 포함 여부
    for pattern in exclude_patterns:
        if pattern in kw:
            return False
    
    # 너무 짧거나 긴 키워드 제외
    if len(kw) < 2 or len(kw) > 15:
        return False
    
    # 숫자만 있는 키워드 제외
    if kw.replace('.', '').replace(',', '').isdigit():
        return False
    
    # 단위만 있는 키워드 제외 (예: 100g, 20ml)
    import re
    if re.match(r'^\d+[gmlkgL매개권조각세트팩]+$', kw):
        return False
    
    # 날짜 형식 제외 (예: 2019년 03월)
    if re.match(r'^\d{4}년', kw):
        return False
    
    # 범위 형식 제외 (예: 61~90, 7~10kg)
    if re.match(r'^\d+[~\-]\d+', kw):
        return False
    
    return True

# 정제된 키워드만 필터링
cleaned_keyword_counts = {kw: count for kw, count in keyword_counts.items() 
                          if is_valid_keyword(kw)}

print(f"정제 후 키워드: {len(cleaned_keyword_counts):,}개")

# 추가 정제: 리뷰 감성 키워드만 선별
review_sentiment_keywords = [
    '만족', '추천', '좋아요', '최고', '가성비', '저렴', '품질', '튼튼',
    '편리', '안전', '부드러움', '흡수력', '디자인', '귀여움', '실용적',
    '선물', '재구매', '아이', '아기', '애기', '유아', '신생아',
    '피부', '순한', '자극', '냄새', '향', '촉감', '사이즈', '크기',
    '배송', '포장', '가격', '할인', '이벤트'
]

# 상품 속성 키워드
product_attr_keywords = [
    '유아용', '신생아용', '팬티형', '밴드형', '캡형', '리필형',
    '무향', '유기농', '천연', '저자극', '무첨가', '국산', '수입',
    '대용량', '소용량', '세트', '단품', '프리미엄', '일반',
    '1단계', '2단계', '3단계', '4단계', '5단계', '6단계',
    '봉제인형', '블록', '퍼즐', '동화책', '그림책', '교구',
    '분유', '이유식', '간식', '음료', '기저귀', '물티슈'
]

# 브랜드/제조사 키워드 (상위 빈도)
brand_keywords = [
    '레고', '하기스', '보솜이', '팸퍼스', '남양유업', '매일유업',
    '일동후디스', '앱솔루트', '파스퇴르', '산양분유'
]

# 핵심 키워드 세트 구성
core_keywords = set(review_sentiment_keywords + product_attr_keywords + brand_keywords)

# 빈도 기반 + 핵심 키워드 결합
final_keywords = {}
for kw, count in cleaned_keyword_counts.items():
    # 핵심 키워드이거나 빈도가 높은 키워드
    if kw in core_keywords or count >= 50:
        final_keywords[kw] = count

print(f"최종 키워드: {len(final_keywords):,}개")
print("\n상위 30개 정제된 키워드:")
for kw, count in sorted(final_keywords.items(), key=lambda x: -x[1])[:30]:
    print(f"  {kw}: {count:,}회")

# keyword_counts를 정제된 버전으로 교체
keyword_counts = Counter(final_keywords)

# 2. KeyBERT 방식으로 키워드 추출 모델 구현
print("\n" + "=" * 60)
print("KeyBERT 기반 키워드 추출기 구현")
print("=" * 60)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KeywordExtractor:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask"):
        print(f"모델 로드 중: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"모델 로드 완료 (Device: {self.device})")
        
        # 도메인 특화 키워드 사전 구축
        self.domain_keywords = self._build_domain_keywords(keyword_counts)
    
    def _build_domain_keywords(self, keyword_counts, min_count=10, max_keywords=300):
        """빈도 기반 도메인 키워드 사전 구축 (정제된 버전)"""
        keywords = [kw for kw, count in keyword_counts.most_common(max_keywords) 
                   if count >= min_count and len(kw) >= 2]
        print(f"도메인 키워드 사전: {len(keywords)}개")
        
        # 키워드 임베딩 미리 계산
        if keywords:
            self.keyword_embeddings = self.model.encode(keywords, show_progress_bar=True)
        else:
            self.keyword_embeddings = None
        
        return keywords
    
    def extract_keywords(self, text, top_n=5, diversity=0.5):
        """텍스트에서 키워드 추출"""
        if not text or not self.domain_keywords:
            return []
        
        # 텍스트 임베딩
        text_embedding = self.model.encode([text])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(text_embedding, self.keyword_embeddings)[0]
        
        # MMR (Maximal Marginal Relevance)로 다양성 확보
        keywords_idx = []
        candidates = list(range(len(self.domain_keywords)))
        
        for _ in range(min(top_n, len(candidates))):
            if not candidates:
                break
            
            if not keywords_idx:
                # 첫 번째는 가장 유사한 키워드
                best_idx = max(candidates, key=lambda x: similarities[x])
            else:
                # MMR 점수 계산
                mmr_scores = []
                for idx in candidates:
                    relevance = similarities[idx]
                    
                    # 이미 선택된 키워드들과의 유사도
                    if keywords_idx:
                        selected_embeddings = self.keyword_embeddings[keywords_idx]
                        candidate_embedding = self.keyword_embeddings[idx:idx+1]
                        redundancy = max(cosine_similarity(candidate_embedding, selected_embeddings)[0])
                    else:
                        redundancy = 0
                    
                    mmr = diversity * relevance - (1 - diversity) * redundancy
                    mmr_scores.append((idx, mmr))
                
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            
            keywords_idx.append(best_idx)
            candidates.remove(best_idx)
        
        # 결과 반환
        results = [(self.domain_keywords[idx], float(similarities[idx])) 
                  for idx in keywords_idx]
        return results
    
    def extract_from_product(self, name, manufacturer='', category='', top_n=5):
        """상품 정보에서 키워드 추출"""
        text = f"{name} {manufacturer} {category}"
        return self.extract_keywords(text, top_n=top_n)

# 3. 모델 학습 및 평가
print("\n" + "=" * 60)
print("키워드 추출기 학습")
print("=" * 60)

extractor = KeywordExtractor()

# 4. 테스트
print("\n" + "=" * 60)
print("키워드 추출 테스트")
print("=" * 60)

# 샘플 테스트
test_samples = df_with_keywords.sample(min(5, len(df_with_keywords)), random_state=42)

for idx, row in test_samples.iterrows():
    print(f"\n{'='*50}")
    print(f"상품명: {row['name']}")
    print(f"제조사: {row['manufacturer']}")
    print(f"카테고리: {row['category_level3']}")
    
    # 실제 키워드
    actual_keywords = row['all_keywords'][:10]
    print(f"\n실제 키워드: {actual_keywords}")
    
    # 추출된 키워드
    extracted = extractor.extract_from_product(
        row['name'], 
        row['manufacturer'] if pd.notna(row['manufacturer']) else '',
        row['category_level3'] if pd.notna(row['category_level3']) else ''
    )
    print(f"추출 키워드: {[kw for kw, score in extracted]}")
    print(f"유사도 점수: {[f'{score:.3f}' for kw, score in extracted]}")

# 5. 정량적 평가 (배치 처리로 최적화)
print("\n" + "=" * 60)
print("정량적 평가 (간소화)")
print("=" * 60)

# 평가용 샘플 (50개로 축소)
eval_df = df_with_keywords.sample(min(50, len(df_with_keywords)), random_state=42)

total_precision = 0
total_recall = 0
total_f1 = 0
count = 0

print("평가 중...")
for i, (idx, row) in enumerate(eval_df.iterrows()):
    actual = set(row['all_keywords'])
    
    # 간단한 유사도 기반 추출 (MMR 생략)
    text = f"{row['name']} {row['manufacturer'] if pd.notna(row['manufacturer']) else ''} {row['category_level3'] if pd.notna(row['category_level3']) else ''}"
    text_embedding = extractor.model.encode([text])
    similarities = cosine_similarity(text_embedding, extractor.keyword_embeddings)[0]
    
    # 상위 10개 선택
    top_indices = np.argsort(similarities)[-10:][::-1]
    predicted = set([extractor.domain_keywords[i] for i in top_indices])
    
    if predicted and actual:
        precision = len(predicted & actual) / len(predicted)
        recall = len(predicted & actual) / len(actual)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1
    
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(eval_df)} 완료")

avg_precision = total_precision / count if count > 0 else 0
avg_recall = total_recall / count if count > 0 else 0
avg_f1 = total_f1 / count if count > 0 else 0

print(f"\n평가 샘플: {count}개")
print(f"평균 Precision: {avg_precision*100:.2f}%")
print(f"평균 Recall: {avg_recall*100:.2f}%")
print(f"평균 F1 Score: {avg_f1*100:.2f}%")

# 6. 모델 저장
print("\n" + "=" * 60)
print("모델 저장")
print("=" * 60)

save_dir = './results_keyword'
os.makedirs(save_dir, exist_ok=True)

# 도메인 키워드 저장
with open(f'{save_dir}/domain_keywords.json', 'w', encoding='utf-8') as f:
    json.dump(extractor.domain_keywords, f, ensure_ascii=False, indent=2)

# 키워드 임베딩 저장
np.save(f'{save_dir}/keyword_embeddings.npy', extractor.keyword_embeddings)

# 메타데이터 저장
metadata = {
    'model_name': 'jhgan/ko-sroberta-multitask',
    'num_keywords': len(extractor.domain_keywords),
    'avg_precision': float(avg_precision),
    'avg_recall': float(avg_recall),
    'avg_f1': float(avg_f1)
}

with open(f'{save_dir}/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {save_dir}")
print("\n다음 단계: python 17_use_keyword_extractor.py 실행")
