# 01_export_sample_10000.py
# 역할: PostgreSQL의 public1.products 테이블에서 카테고리별 균형 샘플링으로 10,000개 추출

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import math

# .env 파일 로드
load_dotenv()

def export_sample_10000(target_samples=10000):
    """
    PostgreSQL의 products 테이블에서 카테고리별 균형 샘플링으로 10,000개 추출
    
    Args:
        target_samples: 목표 샘플 수 (기본값: 10000)
    """
    
    print("=" * 60)
    print("PostgreSQL 연결 중...")
    print("=" * 60)
    
    # PostgreSQL 연결
    conn = psycopg2.connect(
        host=os.getenv('PGHOST'),
        port=os.getenv('PGPORT'),
        database=os.getenv('PGDATABASE'),
        user=os.getenv('PGUSER'),
        password=os.getenv('PGPASSWORD')
    )
    
    schema = os.getenv('PGSCHEMA', 'public1')
    table = os.getenv('PGTABLE', 'products')
    
    print(f"[OK] 데이터베이스 연결 완료: {schema}.{table}")
    print()
    
    # 전체 카테고리별 개수 조회
    print("=" * 60)
    print("카테고리별 데이터 개수 조회 중...")
    print("=" * 60)
    
    count_query = f"""
    SELECT category, COUNT(*) as cnt
    FROM {schema}.{table}
    WHERE category IS NOT NULL
        AND category != ''
    GROUP BY category
    ORDER BY cnt DESC
    """
    
    category_counts = pd.read_sql(count_query, conn)
    
    print(f"전체 카테고리 수: {len(category_counts)}개")
    print(f"전체 데이터 개수: {category_counts['cnt'].sum():,}개")
    print()
    
    # 상위 10개 카테고리 표시
    print("상위 10개 카테고리:")
    for idx, row in category_counts.head(10).iterrows():
        print(f"  [{idx+1}] {row['category']}: {row['cnt']:,}개")
    print()
    
    # 카테고리별 샘플링 개수 계산
    # 목표 샘플 수에 맞춰 각 카테고리에서 균등하게 샘플링
    num_categories = len(category_counts)
    samples_per_category = max(1, math.floor(target_samples / num_categories))
    
    # 실제 추출할 샘플 수 조정 (각 카테고리에서 최대한 균등하게)
    total_samples = 0
    sampled_dfs = []
    
    print("=" * 60)
    print(f"카테고리별 샘플링 시작 (목표: {target_samples:,}개)")
    print("=" * 60)
    
    for idx, row in category_counts.iterrows():
        category = row['category']
        available_count = row['cnt']
        
        # 각 카테고리에서 샘플링 개수 결정
        # 목표 샘플 수를 넘지 않도록 조정
        if total_samples >= target_samples:
            break
            
        remaining_samples = target_samples - total_samples
        remaining_categories = num_categories - idx
        
        # 남은 카테고리 수를 고려하여 샘플링 개수 결정
        if remaining_categories > 0:
            current_samples = min(
                available_count,
                max(1, math.ceil(remaining_samples / remaining_categories))
            )
        else:
            current_samples = min(available_count, remaining_samples)
        
        # 실제 샘플링
        sample_query = f"""
        SELECT 
            id,
            category as category_name,
            CONCAT_WS(' | ', name, manufacturer, category) as text
        FROM {schema}.{table}
        WHERE category = %s
        ORDER BY RANDOM()
        LIMIT %s
        """
        
        sampled_df = pd.read_sql(sample_query, conn, params=(category, current_samples))
        
        if len(sampled_df) > 0:
            sampled_dfs.append(sampled_df)
            total_samples += len(sampled_df)
            print(f"{category}: {available_count:,}개 중 {len(sampled_df):,}개 샘플링")
    
    conn.close()
    
    # 모든 샘플 합치기
    if len(sampled_dfs) > 0:
        df_final = pd.concat(sampled_dfs, ignore_index=True)
    else:
        print("경고: 샘플링된 데이터가 없습니다.")
        return None
    
    print()
    print("=" * 60)
    print("샘플링 완료")
    print("=" * 60)
    print(f"최종 데이터: {len(df_final):,}개")
    print(f"카테고리 수: {df_final['category_name'].nunique()}개")
    print()
    
    # 카테고리별 샘플 수 확인
    category_dist = df_final['category_name'].value_counts()
    print("카테고리별 샘플 수:")
    print(f"  최소: {category_dist.min()}개")
    print(f"  최대: {category_dist.max()}개")
    print(f"  평균: {category_dist.mean():.1f}개")
    print()
    
    # CSV 저장
    output_file = 'training_data_10000.csv'
    print("=" * 60)
    print(f"CSV 저장 중: {output_file}")
    print("=" * 60)
    
    # 컬럼 순서: id, category_name, text
    output_df = df_final[['id', 'category_name', 'text']]
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"[OK] 저장 완료: {os.path.abspath(output_file)}")
    print()
    print("=" * 60)
    print("추출 완료")
    print("=" * 60)
    print("다음 단계: python 02_finetune_local.py 실행")
    print()
    
    return df_final

if __name__ == "__main__":
    df = export_sample_10000(target_samples=10000)
