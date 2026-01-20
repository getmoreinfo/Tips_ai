# export_all_products.py
# 역할: danawa_crawlingdb의 public1.products 테이블 전체를 CSV로 저장

import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

print("=" * 60)
print("Products 테이블 전체 CSV 내보내기")
print("=" * 60)

try:
    # 환경 변수에서 DB 연결 정보 읽기
    db_config = {
        'host': os.getenv('PGHOST'),
        'port': os.getenv('PGPORT'),
        'database': 'danawa_crawlingdb',  # 명시적으로 지정
        'user': os.getenv('PGUSER'),
        'password': os.getenv('PGPASSWORD')
    }
    
    print(f"데이터베이스: {db_config['database']}")
    print(f"호스트: {db_config['host']}:{db_config['port']}")
    print()
    
    # PostgreSQL 연결
    print("데이터베이스 연결 중...")
    conn = psycopg2.connect(**db_config)
    print("[OK] 연결 성공")
    print()
    
    # 테이블 정보 확인
    schema = 'public1'
    table = 'products'
    
    print("=" * 60)
    print(f"테이블 조회: {schema}.{table}")
    print("=" * 60)
    
    # 전체 레코드 수 확인
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    total_count = cur.fetchone()[0]
    print(f"전체 레코드 수: {total_count:,}개")
    print()
    
    # 컬럼 정보 확인
    cur.execute(f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
    """)
    columns = cur.fetchall()
    print("컬럼 정보:")
    for col_name, col_type in columns:
        print(f"  - {col_name}: {col_type}")
    print()
    
    # 전체 데이터 조회 및 DataFrame으로 변환
    print("데이터 로딩 중... (시간이 걸릴 수 있습니다)")
    query = f"SELECT * FROM {schema}.{table}"
    df = pd.read_sql(query, conn)
    print(f"[OK] 데이터 로드 완료: {len(df):,}개 행, {len(df.columns)}개 컬럼")
    print()
    
    # CSV 파일로 저장
    output_file = 'products_all.csv'
    print(f"CSV 저장 중: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')  # utf-8-sig: Excel에서 한글 깨짐 방지
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_file)
    if file_size > 1024 * 1024 * 1024:  # 1GB 이상
        size_str = f"{file_size / (1024**3):.2f} GB"
    elif file_size > 1024 * 1024:  # 1MB 이상
        size_str = f"{file_size / (1024**2):.2f} MB"
    else:
        size_str = f"{file_size / 1024:.2f} KB"
    
    print(f"[OK] 저장 완료: {output_file} ({size_str})")
    print()
    
    # 샘플 데이터 미리보기
    print("=" * 60)
    print("데이터 미리보기 (상위 5개)")
    print("=" * 60)
    print(df.head())
    
    # 연결 종료
    cur.close()
    conn.close()
    
    print()
    print("=" * 60)
    print("내보내기 완료")
    print("=" * 60)
    print(f"파일: {output_file}")
    print(f"레코드: {len(df):,}개")
    print(f"파일 크기: {size_str}")

except psycopg2.Error as e:
    print(f"[ERROR] 데이터베이스 오류: {e}")
    exit(1)
except Exception as e:
    print(f"[ERROR] 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
