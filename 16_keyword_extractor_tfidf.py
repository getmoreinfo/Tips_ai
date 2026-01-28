# 16_keyword_extractor_tfidf.py
# 역할: 규칙 기반 키워드 추출 (konlpy 없이)

import pandas as pd
import numpy as np
import json
import os
import re
import ast
from collections import Counter

print("=" * 60)
print("규칙 기반 키워드 추출기")
print("=" * 60)

# 1. 데이터 로드
print("\n데이터 로드 중...")
df = pd.read_csv('products_all_categorized.csv')
print(f"전체 데이터: {len(df):,}개")

# 2. 키워드 추출 클래스
class RuleBasedKeywordExtractor:
    def __init__(self):
        # 불용어 (제외할 단어)
        self.stopwords = {
            '있는', '하는', '되는', '위한', '통한', '대한', '관한',
            '그', '이', '저', '것', '수', '등', '및', '또는', '의',
            '을', '를', '에', '에서', '로', '으로', '와', '과',
            '상품', '제품', '구매', '판매', '일반', '기타', '용',
            'nan', 'None', '', '일반구매'
        }
        
        # 브랜드/제조사 사전
        self.brands = self._load_brands(df)
        
        # 카테고리 키워드
        self.category_keywords = self._load_category_keywords(df)
    
    def _load_brands(self, df):
        """제조사 목록 로드"""
        brands = set()
        for m in df['manufacturer'].dropna().unique():
            if len(str(m)) >= 2:
                brands.add(str(m))
        print(f"브랜드 사전: {len(brands)}개")
        return brands
    
    def _load_category_keywords(self, df):
        """카테고리에서 키워드 추출"""
        keywords = set()
        for cat in df['category_level3'].dropna().unique():
            keywords.add(str(cat))
        print(f"카테고리 키워드: {len(keywords)}개")
        return keywords
    
    def extract_words(self, text):
        """공백/특수문자 기준 단어 추출 (형태소 분석 없이)"""
        if not text or pd.isna(text):
            return []
        
        text = str(text)
        # 괄호 내용 분리
        text = re.sub(r'\(([^)]+)\)', r' \1 ', text)
        # 특수문자를 공백으로
        text = re.sub(r'[^\w가-힣]', ' ', text)
        # 공백 기준 분리
        words = text.split()
        # 불용어 제거 및 길이 필터
        words = [w for w in words if w not in self.stopwords and len(w) >= 2]
        return words
    
    def extract_patterns(self, text):
        """패턴 기반 키워드 추출"""
        if not text:
            return []
        
        text = str(text)
        keywords = []
        
        # 단계 패턴 (1단계, 2단계 등)
        stages = re.findall(r'\d단계', text)
        keywords.extend(stages)
        
        # 용량 패턴 (100g, 500ml 등)
        volumes = re.findall(r'\d+(?:g|ml|L|kg|매|개|권|팩|cm)', text, re.IGNORECASE)
        keywords.extend(volumes)
        
        # 연령 패턴 (0~6개월, 신생아용 등)
        ages = re.findall(r'(?:\d+[~\-]\d+개월|\d+개월|신생아|유아|영아|아기)', text)
        keywords.extend(ages)
        
        # 형태 패턴 (팬티형, 밴드형, 캡형 등)
        types = re.findall(r'[가-힣]+형', text)
        keywords.extend(types)
        
        # 세트/권 패턴
        sets = re.findall(r'세트\s*\d+권|\d+권\s*세트|\d+권', text)
        keywords.extend(sets)
        
        return keywords
    
    def extract_keywords(self, name, manufacturer='', category='', top_n=7):
        """상품 정보에서 키워드 추출"""
        keywords = []
        scores = {}
        
        # 1. 브랜드 매칭
        if manufacturer and str(manufacturer) != 'nan' and str(manufacturer) in self.brands:
            keywords.append(str(manufacturer))
            scores[str(manufacturer)] = 1.0
        
        # 2. 카테고리 매칭
        if category and str(category) != 'nan' and str(category) in self.category_keywords:
            keywords.append(str(category))
            scores[str(category)] = 0.9
        
        # 3. 패턴 기반 추출
        patterns = self.extract_patterns(name)
        for p in patterns:
            if p not in keywords:
                keywords.append(p)
                scores[p] = 0.85
        
        # 4. 단어 추출 (형태소 분석 없이)
        words = self.extract_words(name)
        for word in words:
            if word not in keywords and word not in self.stopwords:
                # 브랜드명이면 높은 점수
                if word in self.brands:
                    keywords.append(word)
                    scores[word] = 0.95
                else:
                    keywords.append(word)
                    scores[word] = 0.7
        
        # 5. 상위 N개 반환
        result = [(kw, scores.get(kw, 0.5)) for kw in keywords[:top_n]]
        return result
    
    def get_keyword_summary(self, name, manufacturer='', category='', top_n=5):
        """키워드 요약 문자열"""
        keywords = self.extract_keywords(name, manufacturer, category, top_n)
        return ', '.join([kw for kw, score in keywords])

# 3. 모델 생성 및 테스트
print("\n" + "=" * 60)
print("키워드 추출기 생성")
print("=" * 60)

extractor = RuleBasedKeywordExtractor()

# 4. 테스트
print("\n" + "=" * 60)
print("키워드 추출 테스트")
print("=" * 60)

test_products = [
    {'name': '그린키즈 요술지팡이 이솝우화 세트 20권', 'manufacturer': '그린키즈', 'category': '그림/동화/놀이책'},
    {'name': '남양유업 아이엠마더 분유 1단계 800g', 'manufacturer': '남양유업', 'category': '분유'},
    {'name': '하기스 매직팬티 기저귀 대형 4단계', 'manufacturer': '하기스', 'category': '기저귀'},
    {'name': '레고 듀플로 대형 놀이공원', 'manufacturer': '레고', 'category': '블록/레고'},
    {'name': '보솜이 천연코튼 물티슈 캡형 80매', 'manufacturer': '보솜이', 'category': '물티슈'},
]

for product in test_products:
    print(f"\n{'='*50}")
    print(f"상품명: {product['name']}")
    print(f"제조사: {product['manufacturer']}")
    print(f"카테고리: {product['category']}")
    
    keywords = extractor.extract_keywords(
        product['name'],
        product['manufacturer'],
        product['category']
    )
    
    print(f"\n추출된 키워드:")
    for kw, score in keywords:
        print(f"  - {kw} (점수: {score:.2f})")

# 5. 정량적 평가
print("\n" + "=" * 60)
print("정량적 평가")
print("=" * 60)

# review_tags가 있는 데이터로 평가
def parse_review_tags(tags_str):
    if pd.isna(tags_str) or tags_str == '[]':
        return []
    try:
        tags = ast.literal_eval(tags_str)
        return tags if isinstance(tags, list) else []
    except:
        return []

df['review_keywords'] = df['review_tags'].apply(parse_review_tags)
eval_df = df[df['review_keywords'].apply(len) > 0].sample(min(100, len(df)), random_state=42)

total_precision = 0
total_recall = 0
count = 0

for idx, row in eval_df.iterrows():
    actual = set(row['review_keywords'])
    
    extracted = extractor.extract_keywords(
        row['name'],
        row['manufacturer'] if pd.notna(row['manufacturer']) else '',
        row['category_level3'] if pd.notna(row['category_level3']) else '',
        top_n=10
    )
    predicted = set([kw for kw, score in extracted])
    
    if predicted and actual:
        precision = len(predicted & actual) / len(predicted) if predicted else 0
        recall = len(predicted & actual) / len(actual) if actual else 0
        
        total_precision += precision
        total_recall += recall
        count += 1

avg_precision = total_precision / count if count > 0 else 0
avg_recall = total_recall / count if count > 0 else 0
avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

print(f"평가 샘플: {count}개")
print(f"평균 Precision: {avg_precision*100:.2f}%")
print(f"평균 Recall: {avg_recall*100:.2f}%")
print(f"평균 F1 Score: {avg_f1*100:.2f}%")

# 6. 저장
print("\n" + "=" * 60)
print("모델 저장")
print("=" * 60)

save_dir = './results_keyword'
os.makedirs(save_dir, exist_ok=True)

metadata = {
    'method': 'Rule-based (no konlpy)',
    'num_brands': len(extractor.brands),
    'num_categories': len(extractor.category_keywords),
    'avg_precision': float(avg_precision),
    'avg_recall': float(avg_recall),
    'avg_f1': float(avg_f1)
}

with open(f'{save_dir}/metadata_tfidf.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# 브랜드 목록 저장
with open(f'{save_dir}/brands.json', 'w', encoding='utf-8') as f:
    json.dump(list(extractor.brands), f, ensure_ascii=False, indent=2)

# 카테고리 키워드 저장
with open(f'{save_dir}/categories.json', 'w', encoding='utf-8') as f:
    json.dump(list(extractor.category_keywords), f, ensure_ascii=False, indent=2)

print(f"저장 완료: {save_dir}")
print("\n완료!")
