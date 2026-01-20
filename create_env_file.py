# create_env_file.py
# 역할: .env 파일 템플릿 생성

import os

def create_env_file():
    """
    .env 파일이 없으면 템플릿을 생성합니다.
    이미 존재하면 덮어쓰지 않습니다.
    """
    
    env_file = '.env'
    
    # .env 파일이 이미 존재하는지 확인
    if os.path.exists(env_file):
        print(f"[WARNING] {env_file} 파일이 이미 존재합니다.")
        response = input("덮어쓰시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            return
    
    # .env 템플릿 내용
    env_template = """# PostgreSQL 데이터베이스 연결 설정
PGHOST=localhost
PGPORT=5432
PGDATABASE=your_database_name
PGUSER=your_username
PGPASSWORD=your_password

# 테이블 설정
PGSCHEMA=public1
PGTABLE=products
"""
    
    # .env 파일 작성
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_template)
    
    print(f"[OK] {env_file} 파일이 생성되었습니다.")
    print()
    print("다음 단계:")
    print("1. .env 파일을 열어서 실제 값으로 수정하세요.")
    print("2. python 00_db_smoke_test.py 실행하여 연결을 테스트하세요.")
    print()
    print(f"파일 위치: {os.path.abspath(env_file)}")

if __name__ == "__main__":
    create_env_file()
