# 00_db_smoke_test.py
# 역할: .env를 로드해서 PostgreSQL 접속 테스트

import psycopg2
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

print("=" * 60)
print("PostgreSQL 접속 테스트")
print("=" * 60)

try:
    # 환경 변수에서 DB 연결 정보 읽기
    db_config = {
        'host': os.getenv('PGHOST'),
        'port': os.getenv('PGPORT'),
        'database': os.getenv('PGDATABASE'),
        'user': os.getenv('PGUSER'),
        'password': os.getenv('PGPASSWORD')
    }
    
    print("환경 변수 확인:")
    print(f"  PGHOST: {db_config['host']}")
    print(f"  PGPORT: {db_config['port']}")
    print(f"  PGDATABASE: {db_config['database']}")
    print(f"  PGUSER: {db_config['user']}")
    print(f"  PGPASSWORD: {'*' * len(db_config['password']) if db_config['password'] else 'None'}")
    print()
    
    # PostgreSQL 연결
    print("데이터베이스 연결 중...")
    conn = psycopg2.connect(**db_config)
    print("[OK] 데이터베이스 연결 성공")
    print()
    
    # 커서 생성
    cur = conn.cursor()
    
    # SELECT 1 테스트
    print("=" * 60)
    print("기본 쿼리 테스트: SELECT 1")
    print("=" * 60)
    cur.execute("SELECT 1")
    result = cur.fetchone()
    print(f"결과: {result[0]}")
    print("[OK] 기본 쿼리 성공")
    print()
    
    # products 테이블 샘플 조회
    schema = os.getenv('PGSCHEMA', 'public1')
    table = os.getenv('PGTABLE', 'products')
    
    print("=" * 60)
    print(f"테이블 샘플 조회: {schema}.{table}")
    print("=" * 60)
    
    query = f"""
    SELECT id, category, name
    FROM {schema}.{table}
    LIMIT 3
    """
    
    cur.execute(query)
    results = cur.fetchall()
    
    print(f"조회된 레코드 수: {len(results)}개")
    print()
    print("결과:")
    for idx, row in enumerate(results, 1):
        print(f"  [{idx}] id={row[0]}, category={row[1]}, name={row[2][:50] if row[2] else 'None'}...")
    print()
    print("[OK] 테이블 조회 성공")
    print()
    
    # 커서 및 연결 종료
    cur.close()
    conn.close()
    
    print("=" * 60)
    print("모든 테스트 완료")
    print("=" * 60)
    print("다음 단계: python 01_export_sample_10000.py 실행")
    
except psycopg2.Error as e:
    print(f"[ERROR] 데이터베이스 오류: {e}")
    exit(1)
except Exception as e:
    print(f"[ERROR] 오류 발생: {e}")
    exit(1)
