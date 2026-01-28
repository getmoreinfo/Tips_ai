# 07_split_category_levels.py
# 역할: products_all.csv의 카테고리를 대분류/중분류/소분류로 분리하여 새 CSV 생성

import pandas as pd

print("=" * 60)
print("카테고리 계층 분리")
print("=" * 60)

# 1. 데이터 로드
df = pd.read_csv('products_all.csv')
print(f"원본 데이터: {len(df):,}개")

# 2. 카테고리 분리 함수
def split_category(category_str):
    """
    '홈 > 식품/유아/완구 > 유아의류/신발/도서 > 그림/동화/놀이책'
    → level1: 식품/유아/완구
    → level2: 유아의류/신발/도서
    → level3: 그림/동화/놀이책
    """
    if pd.isna(category_str):
        return None, None, None
    
    parts = [p.strip() for p in category_str.split('>')]
    
    # '홈' 제거 (첫 번째 요소)
    if parts and parts[0] == '홈':
        parts = parts[1:]
    
    level1 = parts[0] if len(parts) > 0 else None
    level2 = parts[1] if len(parts) > 1 else None
    level3 = parts[2] if len(parts) > 2 else None
    
    return level1, level2, level3

# 3. 카테고리 분리 적용
print("\n카테고리 분리 중...")
category_splits = df['category'].apply(split_category)

df['category_level1'] = category_splits.apply(lambda x: x[0])
df['category_level2'] = category_splits.apply(lambda x: x[1])
df['category_level3'] = category_splits.apply(lambda x: x[2])

# 4. 결과 확인
print("\n" + "=" * 60)
print("분리 결과")
print("=" * 60)

print(f"\n대분류 (level1) 개수: {df['category_level1'].nunique()}개")
print("대분류 목록:")
for cat in sorted(df['category_level1'].dropna().unique()):
    count = len(df[df['category_level1'] == cat])
    print(f"  - {cat}: {count:,}개")

print(f"\n중분류 (level2) 개수: {df['category_level2'].nunique()}개")
print("중분류 샘플:")
for cat in list(df['category_level2'].dropna().unique())[:10]:
    count = len(df[df['category_level2'] == cat])
    print(f"  - {cat}: {count:,}개")
if df['category_level2'].nunique() > 10:
    print(f"  ... 외 {df['category_level2'].nunique() - 10}개")

print(f"\n소분류 (level3) 개수: {df['category_level3'].nunique()}개")
print("소분류 샘플:")
for cat in list(df['category_level3'].dropna().unique())[:10]:
    count = len(df[df['category_level3'] == cat])
    print(f"  - {cat}: {count:,}개")
if df['category_level3'].nunique() > 10:
    print(f"  ... 외 {df['category_level3'].nunique() - 10}개")

# 5. 예시 출력
print("\n" + "=" * 60)
print("분리 예시 (처음 5개)")
print("=" * 60)
sample_df = df[['name', 'category', 'category_level1', 'category_level2', 'category_level3']].head(5)
for idx, row in sample_df.iterrows():
    print(f"\n상품명: {row['name'][:50]}...")
    print(f"  원본: {row['category']}")
    print(f"  대분류: {row['category_level1']}")
    print(f"  중분류: {row['category_level2']}")
    print(f"  소분류: {row['category_level3']}")

# 6. 새 CSV 저장
output_file = 'products_all_categorized.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("저장 완료")
print("=" * 60)
print(f"파일명: {output_file}")
print(f"총 레코드: {len(df):,}개")
print(f"컬럼 수: {len(df.columns)}개 (기존 + 3개 추가)")

# 7. 추가된 컬럼 확인
print("\n추가된 컬럼:")
print("  - category_level1 (대분류)")
print("  - category_level2 (중분류)")
print("  - category_level3 (소분류)")

print("\n다음 단계: python 08_train_hierarchical_classifier.py 실행")
