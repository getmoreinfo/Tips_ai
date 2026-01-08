# 01_prepare_training_data.py
# 역할: 크롤링 DB에서 다나와 카테고리별로 샘플링 (category 컬럼 사용)

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def prepare_training_data(samples_per_category=3):
    """
    크롤링 DB에서 다나와 카테고리별로 N개씩만 샘플링
    
    Args:
        samples_per_category: 카테고리당 샘플 수 (기본값: 3)
    """
    
    print("=" * 60)
    print("크롤링 DB 연결 중...")
    print("=" * 60)
    
    # 크롤링 DB 연결
    conn = psycopg2.connect(
        host=os.getenv('CRAWLING_DB_HOST'),
        port=os.getenv('CRAWLING_DB_PORT'),
        database=os.getenv('CRAWLING_DB_NAME'),
        user=os.getenv('CRAWLING_DB_USER'),
        password=os.getenv('CRAWLING_DB_PASSWORD')
    )
    
    print("크롤링 DB 연결 완료")
    print()
    
    # 크롤링 데이터 조회 (category 컬럼 사용)
    print("=" * 60)
    print("데이터 조회 중...")
    print("=" * 60)
    
    query = """
    SELECT 
        p.id,
        p.name,
        p.manufacturer,
        p.specifications::text as specifications,
        p.category as category_name
    FROM public1.products p
    WHERE p.category IS NOT NULL
        AND p.category != 'N/A'
    ORDER BY p.id
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"전체 데이터: {len(df):,}개")
    print(f"다나와 카테고리 수: {df['category_name'].nunique()}개")
    print()
    
    # 다나와 카테고리 분포 확인
    print("다나와 카테고리별 분포:")
    category_dist = df['category_name'].value_counts()
    for i, (cat_name, count) in enumerate(category_dist.items(), 1):
        if i <= 20:  # 상위 20개만 출력
            print(f"  [{i}] {cat_name}: {count:,}개")
    if len(category_dist) > 20:
        print(f"  ... 외 {len(category_dist) - 20}개 카테고리")
    print()
    
    # 카테고리별 샘플링
    print("=" * 60)
    print(f"각 다나와 카테고리당 {samples_per_category}개씩 샘플링")
    print("=" * 60)
    
    sampled_dfs = []
    
    for category_name in sorted(df['category_name'].unique()):
        category_data = df[df['category_name'] == category_name]
        
        # 해당 카테고리에서 N개 샘플링
        if len(category_data) > samples_per_category:
            sampled = category_data.sample(n=samples_per_category, random_state=42)
            print(f"{category_name}: {len(category_data):,}개 중 {samples_per_category}개 샘플링")
        else:
            sampled = category_data
            print(f"{category_name}: {len(category_data):,}개 전체 사용")
        
        sampled_dfs.append(sampled)
    
    # 합치기
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    print()
    print("=" * 60)
    print("샘플링 완료")
    print("=" * 60)
    print(f"최종 데이터: {len(df_sampled):,}개")
    print(f"다나와 카테고리 수: {df_sampled['category_name'].nunique()}개")
    print()
    
    # 텍스트 생성 (상품명 + 제조사 + 스펙)
    print("텍스트 생성 중...")
    df_sampled['text'] = (
        df_sampled['name'] + ' ' + 
        df_sampled['manufacturer'].fillna('') + ' ' + 
        df_sampled['specifications'].fillna('')
    ).str.strip()
    
    # CSV 저장 (category_name만 사용)
    output_df = df_sampled[['text', 'category_name']]
    output_df.to_csv('training_data.csv', index=False, encoding='utf-8-sig')
    
    print()
    print("=" * 60)
    print("CSV 저장 완료: training_data.csv")
    print("=" * 60)
    print(f"파일 위치: {os.path.abspath('training_data.csv')}")
    print()
    print("참고: AI는 다나와 카테고리(문자열)를 학습합니다.")
    print("      플랫폼 카테고리 매핑은 04번에서 진행됩니다.")
    print()
    print("다음 단계: python 02_finetune_local.py 실행")
    
    return df_sampled

if __name__ == "__main__":
    # 크롤링 DB에서 각 다나와 카테고리당 3개씩 샘플링
    df = prepare_training_data(samples_per_category=3)