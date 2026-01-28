# 18_report_generator.py
# 역할: 학습된 모델들을 조합하여 상품/브랜드 리포트 자동 생성 (개선 버전)

import torch
import pandas as pd
import numpy as np
import json
import os
import re
import ast
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("통합 리포트 생성기 v2.0")
print("=" * 60)

# ============================================================
# 1. 유틸리티 함수
# ============================================================

def safe_value(value, default="정보없음"):
    """nan 값 안전하게 처리"""
    if pd.isna(value) or value is None or str(value) == 'nan':
        return default
    return value

def safe_number(value, default=0):
    """숫자 nan 처리"""
    if pd.isna(value) or value is None:
        return default
    return value

def truncate_text(text, max_len=50):
    """텍스트 적절한 길이로 자르기"""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."

def format_price(price):
    """가격 포맷팅"""
    if pd.isna(price) or price is None or price == 0:
        return "정보없음"
    return f"{price:,.0f}원"

def format_rating(rating):
    """평점 포맷팅"""
    if pd.isna(rating) or rating is None:
        return "정보없음"
    return f"{rating:.1f}점"

# ============================================================
# 2. 모델 로드
# ============================================================
print("\n모델 로드 중...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 카테고리 분류 모델
print("  - 카테고리 분류 모델 로드...")
category_dir = './results_category/finetuned_category_classifier'
category_tokenizer = AutoTokenizer.from_pretrained(category_dir)
category_model = AutoModelForSequenceClassification.from_pretrained(category_dir)
category_model.to(device)
category_model.eval()

with open(f'{category_dir}/metadata.json', 'r', encoding='utf-8') as f:
    category_metadata = json.load(f)

# 가격대 예측 모델
print("  - 가격대 예측 모델 로드...")
price_dir = './results_price/finetuned_price_predictor'
price_tokenizer = AutoTokenizer.from_pretrained(price_dir)
price_model = AutoModelForSequenceClassification.from_pretrained(price_dir)
price_model.to(device)
price_model.eval()

with open(f'{price_dir}/metadata.json', 'r', encoding='utf-8') as f:
    price_metadata = json.load(f)

# 상품 유사도 모델
print("  - 상품 유사도 모델 로드...")
similarity_dir = './results_similarity'
similarity_model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)
product_embeddings = np.load(f'{similarity_dir}/product_embeddings.npy')

import pickle
with open(f'{similarity_dir}/product_info.pkl', 'rb') as f:
    product_info = pickle.load(f)

print("모델 로드 완료!")

# ============================================================
# 3. 데이터 로드
# ============================================================
print("\n데이터 로드 중...")
df = pd.read_csv('products_all_categorized.csv')
print(f"전체 상품: {len(df):,}개")

# ============================================================
# 4. 키워드 추출기
# ============================================================
class KeywordExtractor:
    def __init__(self):
        self.stopwords = {'상품', '제품', '구매', '판매', '일반', '기타', '용', 'nan', 'None', '', '일반구매'}
    
    def extract(self, name, manufacturer='', category=''):
        keywords = []
        
        if manufacturer and str(manufacturer) != 'nan':
            keywords.append(str(manufacturer))
        
        if category and str(category) != 'nan':
            keywords.append(str(category))
        
        text = str(name)
        patterns = re.findall(r'\d단계|\d+(?:g|ml|L|kg|매|개|권|팩|cm)|[가-힣]+형', text, re.IGNORECASE)
        keywords.extend(patterns)
        
        text = re.sub(r'[^\w가-힣]', ' ', text)
        words = [w for w in text.split() if len(w) >= 2 and w not in self.stopwords]
        keywords.extend(words[:5])
        
        return list(dict.fromkeys(keywords))[:7]

keyword_extractor = KeywordExtractor()

# ============================================================
# 5. 분석 함수들
# ============================================================

def predict_category(text):
    """카테고리 예측"""
    inputs = category_tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = category_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    label_to_category = category_metadata['label_to_category']
    return label_to_category[str(pred_idx)], confidence

def predict_price_range(text):
    """가격대 예측"""
    inputs = price_tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = price_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    label_to_price = price_metadata['label_to_price']
    return label_to_price[str(pred_idx)], confidence

def find_similar_products(text, top_n=5):
    """유사 상품 찾기"""
    query_embedding = similarity_model.encode([text])
    similarities = cosine_similarity(query_embedding, product_embeddings)[0]
    
    top_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
    
    results = []
    for idx in top_indices:
        item = product_info[idx]
        results.append({
            'name': safe_value(item.get('name', ''), ''),
            'manufacturer': safe_value(item.get('manufacturer', ''), ''),
            'category': safe_value(item.get('category_level3', ''), ''),
            'price': safe_number(item.get('min_price', 0)),
            'similarity': float(similarities[idx])
        })
    
    return results

def get_brand_stats(manufacturer):
    """브랜드 통계"""
    brand_df = df[df['manufacturer'] == manufacturer]
    
    if len(brand_df) == 0:
        return None
    
    return {
        'product_count': len(brand_df),
        'categories': brand_df['category_level3'].dropna().unique().tolist()[:5],
        'avg_price': safe_number(brand_df['min_price'].mean()),
        'avg_rating': safe_number(brand_df['average_rating'].mean()),
        'total_reviews': safe_number(brand_df['review_count'].sum())
    }

def get_category_stats(category):
    """카테고리 통계"""
    # category에서 마지막 부분만 추출 (예: "홈 > 식품/유아/완구 > 분유/기저귀/물티슈 > 기저귀" -> "기저귀")
    if '>' in str(category):
        category_name = category.split('>')[-1].strip()
    else:
        category_name = category
    
    cat_df = df[df['category_level3'] == category_name]
    
    if len(cat_df) == 0:
        # 전체 카테고리로 다시 시도
        cat_df = df[df['category'].str.contains(category_name, na=False)]
    
    if len(cat_df) == 0:
        return None
    
    return {
        'category_name': category_name,
        'product_count': len(cat_df),
        'top_brands': cat_df['manufacturer'].value_counts().head(5).to_dict(),
        'price_range': {
            'min': safe_number(cat_df['min_price'].min()),
            'max': safe_number(cat_df['min_price'].max()),
            'avg': safe_number(cat_df['min_price'].mean())
        },
        'avg_rating': safe_number(cat_df['average_rating'].mean())
    }

def analyze_price_position(actual_price, category_stats):
    """가격 포지션 분석 (카테고리 평균 대비)"""
    if not category_stats or actual_price == 0:
        return None
    
    avg_price = category_stats['price_range']['avg']
    if avg_price == 0:
        return None
    
    ratio = actual_price / avg_price
    
    if ratio < 0.7:
        return {'position': '저가', 'ratio': ratio, 'description': f'카테고리 평균 대비 {(1-ratio)*100:.0f}% 저렴'}
    elif ratio > 1.3:
        return {'position': '고가', 'ratio': ratio, 'description': f'카테고리 평균 대비 {(ratio-1)*100:.0f}% 비쌈'}
    else:
        return {'position': '적정가', 'ratio': ratio, 'description': '카테고리 평균 수준'}

def extract_review_keywords(product_df):
    """리뷰 태그에서 키워드 추출"""
    all_tags = []
    
    for tags_str in product_df['review_tags'].dropna():
        try:
            tags = ast.literal_eval(tags_str)
            if isinstance(tags, list):
                all_tags.extend(tags)
        except:
            pass
    
    if not all_tags:
        return []
    
    from collections import Counter
    tag_counts = Counter(all_tags)
    return tag_counts.most_common(10)

def analyze_price_trend(product_row):
    """가격 트렌드 분석"""
    try:
        trend_str = product_row.get('price_trend', '')
        if pd.isna(trend_str) or not trend_str:
            return None
        
        trend_data = ast.literal_eval(trend_str)
        if not trend_data or len(trend_data) < 2:
            return None
        
        # 최근 가격과 과거 가격 비교
        current = trend_data[0].get('price', 0)
        oldest = trend_data[-1].get('price', 0)
        
        if current == 0 or oldest == 0:
            return None
        
        change_rate = (current - oldest) / oldest * 100
        
        return {
            'current': current,
            'oldest': oldest,
            'change_rate': change_rate,
            'trend': '상승' if change_rate > 5 else ('하락' if change_rate < -5 else '유지')
        }
    except:
        return None

# ============================================================
# 6. 리포트 생성 (콘솔 + 파일)
# ============================================================

def generate_product_report(product_name, manufacturer='', category='', save_file=True):
    """상품 리포트 생성"""
    report_lines = []
    
    def add_line(text=""):
        print(text)
        report_lines.append(text)
    
    add_line("\n" + "=" * 60)
    add_line("📦 상품 분석 리포트")
    add_line("=" * 60)
    
    # 기본 정보
    add_line(f"\n## 📌 기본 정보")
    add_line(f"- **상품명**: {product_name}")
    if manufacturer:
        add_line(f"- **제조사**: {manufacturer}")
    if category:
        add_line(f"- **카테고리**: {category}")
    
    # 키워드 추출
    keywords = keyword_extractor.extract(product_name, manufacturer, category)
    add_line(f"\n## 🏷️ 핵심 키워드")
    add_line(f"`{', '.join(keywords)}`")
    
    # 카테고리 예측
    text = f"{product_name} {manufacturer} {category}"
    pred_category, cat_conf = predict_category(text)
    add_line(f"\n## 📂 카테고리 분석")
    add_line(f"- **예측 카테고리**: {pred_category}")
    add_line(f"- **신뢰도**: {cat_conf*100:.1f}%")
    
    # 가격대 예측
    pred_price, price_conf = predict_price_range(text)
    add_line(f"\n## 💰 가격대 분석")
    add_line(f"- **예측 가격대**: {pred_price}")
    add_line(f"- **신뢰도**: {price_conf*100:.1f}%")
    add_line(f"- **가격 기준**: {price_metadata['price_ranges'].get(pred_price, '')}")
    
    # 카테고리 시장 분석 + 가격 포지션
    cat_stats = get_category_stats(pred_category)
    if cat_stats:
        add_line(f"\n## 📊 카테고리 시장 분석 ({cat_stats['category_name']})")
        add_line(f"- **전체 상품 수**: {cat_stats['product_count']:,}개")
        add_line(f"- **가격 범위**: {format_price(cat_stats['price_range']['min'])} ~ {format_price(cat_stats['price_range']['max'])}")
        add_line(f"- **평균 가격**: {format_price(cat_stats['price_range']['avg'])}")
        add_line(f"- **평균 평점**: {format_rating(cat_stats['avg_rating'])}")
        
        add_line(f"\n### 주요 브랜드")
        for i, (brand, count) in enumerate(list(cat_stats['top_brands'].items())[:5], 1):
            marker = "👑" if brand == manufacturer else f"{i}."
            share = count / cat_stats['product_count'] * 100
            add_line(f"  {marker} {brand}: {count}개 ({share:.1f}%)")
    
    # 유사 상품
    similar = find_similar_products(text, top_n=5)
    add_line(f"\n## 🔍 유사 상품 (경쟁 제품)")
    add_line("| 순위 | 상품명 | 제조사 | 가격 | 유사도 |")
    add_line("|------|--------|--------|------|--------|")
    for i, item in enumerate(similar, 1):
        add_line(f"| {i} | {truncate_text(item['name'], 30)} | {safe_value(item['manufacturer'], '-')} | {format_price(item['price'])} | {item['similarity']*100:.1f}% |")
    
    # 브랜드 분석
    if manufacturer:
        brand_stats = get_brand_stats(manufacturer)
        if brand_stats:
            add_line(f"\n## 🏢 브랜드 분석 ({manufacturer})")
            add_line(f"- **등록 상품 수**: {brand_stats['product_count']:,}개")
            add_line(f"- **주요 카테고리**: {', '.join(brand_stats['categories'][:3])}")
            add_line(f"- **평균 가격**: {format_price(brand_stats['avg_price'])}")
            add_line(f"- **평균 평점**: {format_rating(brand_stats['avg_rating'])}")
            add_line(f"- **총 리뷰 수**: {brand_stats['total_reviews']:,.0f}개")
            
            # 브랜드 리뷰 키워드
            brand_df = df[df['manufacturer'] == manufacturer]
            review_keywords = extract_review_keywords(brand_df)
            if review_keywords:
                add_line(f"\n### 브랜드 리뷰 키워드")
                keywords_str = ', '.join([f"{kw}({cnt})" for kw, cnt in review_keywords[:7]])
                add_line(f"`{keywords_str}`")
    
    add_line("\n" + "-" * 60)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    add_line(f"리포트 생성: {timestamp}")
    add_line("=" * 60)
    
    # 파일 저장
    if save_file:
        save_report(report_lines, f"product_{manufacturer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    return report_lines


def generate_brand_report(manufacturer, save_file=True):
    """브랜드 리포트 생성"""
    report_lines = []
    
    def add_line(text=""):
        print(text)
        report_lines.append(text)
    
    brand_df = df[df['manufacturer'] == manufacturer]
    
    if len(brand_df) == 0:
        add_line(f"\n'{manufacturer}' 브랜드를 찾을 수 없습니다.")
        return report_lines
    
    add_line("\n" + "=" * 60)
    add_line("🏢 브랜드 분석 리포트")
    add_line("=" * 60)
    
    add_line(f"\n## 📌 브랜드 개요: {manufacturer}")
    add_line(f"- **등록 상품 수**: {len(brand_df):,}개")
    
    # 카테고리 분포
    add_line(f"\n## 📂 카테고리 분포")
    cat_dist = brand_df['category_level3'].value_counts().head(10)
    add_line("| 카테고리 | 상품 수 | 비중 |")
    add_line("|----------|---------|------|")
    for cat, count in cat_dist.items():
        add_line(f"| {cat} | {count}개 | {count/len(brand_df)*100:.1f}% |")
    
    # 가격 분석
    if brand_df['min_price'].notna().any():
        add_line(f"\n## 💰 가격 분석")
        add_line(f"- **최저가**: {format_price(brand_df['min_price'].min())}")
        add_line(f"- **최고가**: {format_price(brand_df['min_price'].max())}")
        add_line(f"- **평균가**: {format_price(brand_df['min_price'].mean())}")
        
        # 가격대 분포
        def get_price_range(price):
            if pd.isna(price): return '정보없음'
            if price < 30000: return '저가(3만원미만)'
            elif price < 100000: return '중가(3-10만원)'
            else: return '고가(10만원이상)'
        
        price_dist = brand_df['min_price'].apply(get_price_range).value_counts()
        add_line(f"\n### 가격대 분포")
        for pr, cnt in price_dist.items():
            add_line(f"  - {pr}: {cnt}개 ({cnt/len(brand_df)*100:.1f}%)")
    
    # 평점 분석
    if brand_df['average_rating'].notna().any():
        add_line(f"\n## ⭐ 평점 분석")
        add_line(f"- **평균 평점**: {format_rating(brand_df['average_rating'].mean())}")
        add_line(f"- **총 리뷰 수**: {safe_number(brand_df['review_count'].sum()):,.0f}개")
        
        # 평점 분포
        def get_rating_range(rating):
            if pd.isna(rating): return '정보없음'
            if rating >= 4.5: return '⭐⭐⭐⭐⭐ (4.5+)'
            elif rating >= 4.0: return '⭐⭐⭐⭐ (4.0-4.5)'
            elif rating >= 3.5: return '⭐⭐⭐ (3.5-4.0)'
            else: return '⭐⭐ (3.5미만)'
        
        rating_dist = brand_df['average_rating'].apply(get_rating_range).value_counts()
        add_line(f"\n### 평점 분포")
        for rt, cnt in rating_dist.items():
            if rt != '정보없음':
                add_line(f"  - {rt}: {cnt}개")
    
    # 리뷰 키워드 분석
    review_keywords = extract_review_keywords(brand_df)
    if review_keywords:
        add_line(f"\n## 💬 리뷰 키워드 분석")
        add_line("| 키워드 | 언급 횟수 |")
        add_line("|--------|----------|")
        for kw, cnt in review_keywords:
            add_line(f"| {kw} | {cnt}회 |")
    
    # 인기 상품
    add_line(f"\n## 🔥 인기 상품 (리뷰 수 기준)")
    top_products = brand_df.nlargest(5, 'review_count')[['name', 'min_price', 'review_count', 'average_rating']]
    add_line("| 순위 | 상품명 | 가격 | 리뷰 | 평점 |")
    add_line("|------|--------|------|------|------|")
    for i, (_, row) in enumerate(top_products.iterrows(), 1):
        add_line(f"| {i} | {truncate_text(row['name'], 35)} | {format_price(row['min_price'])} | {safe_number(row['review_count']):,.0f}개 | {format_rating(row['average_rating'])} |")
    
    # 경쟁 브랜드 분석
    add_line(f"\n## 🏆 경쟁 브랜드 분석")
    main_category = brand_df['category_level3'].mode().iloc[0] if len(brand_df) > 0 else None
    if main_category:
        cat_df = df[df['category_level3'] == main_category]
        competitors = cat_df['manufacturer'].value_counts().head(10)
        add_line(f"- **주요 카테고리**: {main_category}")
        add_line(f"- **카테고리 전체 상품**: {len(cat_df):,}개")
        
        # 시장 점유율
        brand_share = len(brand_df[brand_df['category_level3'] == main_category]) / len(cat_df) * 100
        add_line(f"- **{manufacturer} 점유율**: {brand_share:.1f}%")
        
        add_line(f"\n### 카테고리 내 브랜드 순위")
        add_line("| 순위 | 브랜드 | 상품 수 | 점유율 |")
        add_line("|------|--------|---------|--------|")
        for i, (brand, count) in enumerate(competitors.items(), 1):
            marker = "👑" if brand == manufacturer else ""
            share = count / len(cat_df) * 100
            add_line(f"| {marker}{i} | {brand} | {count}개 | {share:.1f}% |")
    
    add_line("\n" + "-" * 60)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    add_line(f"리포트 생성: {timestamp}")
    add_line("=" * 60)
    
    # 파일 저장
    if save_file:
        save_report(report_lines, f"brand_{manufacturer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    return report_lines


def save_report(report_lines, filename):
    """리포트를 마크다운 파일로 저장"""
    report_dir = './reports'
    os.makedirs(report_dir, exist_ok=True)
    
    filepath = f"{report_dir}/{filename}.md"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📁 리포트 저장됨: {filepath}")


# ============================================================
# 7. 메인 실행
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("리포트 생성 테스트")
    print("=" * 60)
    
    # 테스트 1: 상품 리포트
    generate_product_report(
        product_name="하기스 매직팬티 기저귀 대형 4단계",
        manufacturer="하기스",
        category="기저귀"
    )
    
    print("\n\n")
    
    # 테스트 2: 브랜드 리포트
    generate_brand_report("남양유업")
    
    # 사용자 입력 모드
    print("\n\n" + "=" * 60)
    print("직접 리포트 생성")
    print("=" * 60)
    
    while True:
        print("\n리포트 유형을 선택하세요:")
        print("  1. 상품 리포트")
        print("  2. 브랜드 리포트")
        print("  q. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == 'q':
            print("종료합니다.")
            break
        elif choice == '1':
            name = input("상품명: ").strip()
            mfr = input("제조사 (선택): ").strip()
            cat = input("카테고리 (선택): ").strip()
            generate_product_report(name, mfr, cat)
        elif choice == '2':
            brand = input("브랜드명: ").strip()
            generate_brand_report(brand)
        else:
            print("잘못된 선택입니다.")
