# shopee_translate_translategemma.py
# 역할: Shopee 크롤링 데이터의 베트남어를 TranslateGemma로 한국어로 번역

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import time

load_dotenv()

class ShopeeTranslator:
    def __init__(self, model_size='4b', device='cuda'):
        """
        TranslateGemma 모델 초기화
        
        Args:
            model_size: '4b', '12b', '27b' 중 선택
            device: 'cuda' 또는 'cpu'
        """
        print("=" * 60)
        print("TranslateGemma 모델 로딩 중...")
        print("=" * 60)
        
        # 모델 이름 매핑 (실제 Hugging Face 모델명)
        model_map = {
            '4b': 'google/translategemma-4b-it',
            '12b': 'google/translategemma-12b-it',
            '27b': 'google/translategemma-27b-it'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['4b'])
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"모델: {model_name}")
        print(f"디바이스: {self.device}")
        
        # Hugging Face 토큰 확인 (환경 변수)
        hf_token = os.getenv('HF_TOKEN')
        
        # 모델 및 토크나이저 로드
        if hf_token:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None
            )
        
        if self.device.type == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print("[OK] 모델 로드 완료")
        print()
    
    def translate(self, vietnamese_text, source_lang='vi', target_lang='ko', max_length=512):
        """
        베트남어 텍스트를 한국어로 번역
        
        Args:
            vietnamese_text: 번역할 베트남어 텍스트
            source_lang: 원본 언어 코드 (vi = 베트남어)
            target_lang: 목표 언어 코드 (ko = 한국어)
            max_length: 최대 생성 길이
        
        Returns:
            번역된 한국어 텍스트
        """
        if pd.isna(vietnamese_text) or not str(vietnamese_text).strip():
            return ""
        
        # TranslateGemma 프롬프트 형식
        prompt = f"Translate from {source_lang} to {target_lang}: {vietnamese_text}"
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # 번역 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                temperature=0.7
            )
        
        # 디코딩
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거하고 번역 결과만 추출
        if f"Translate from {source_lang} to {target_lang}:" in translated:
            translated = translated.split(f"Translate from {source_lang} to {target_lang}:")[-1].strip()
        
        return translated
    
    def translate_batch(self, texts, batch_size=8, source_lang='vi', target_lang='ko'):
        """
        여러 텍스트를 배치로 번역
        
        Args:
            texts: 번역할 텍스트 리스트
            batch_size: 배치 크기
            source_lang: 원본 언어
            target_lang: 목표 언어
        
        Returns:
            번역된 텍스트 리스트
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="번역 진행 중"):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                try:
                    translated = self.translate(text, source_lang, target_lang)
                    batch_results.append(translated)
                except Exception as e:
                    print(f"번역 오류: {e}")
                    batch_results.append("")  # 오류 시 빈 문자열
            
            results.extend(batch_results)
            
            # GPU 메모리 정리
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results


def translate_from_csv(input_csv, output_csv, columns_to_translate, model_size='4b'):
    """
    CSV 파일에서 특정 컬럼들을 번역
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
        columns_to_translate: 번역할 컬럼명 리스트 (예: ['name', 'description'])
        model_size: 모델 크기 ('4b', '12b', '27b')
    """
    print("=" * 60)
    print("CSV 파일 번역 시작")
    print("=" * 60)
    
    # CSV 로드
    print(f"CSV 파일 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"데이터 개수: {len(df):,}개")
    print(f"컬럼: {list(df.columns)}")
    print()
    
    # 번역기 초기화
    translator = ShopeeTranslator(model_size=model_size)
    
    # 각 컬럼 번역
    for col in columns_to_translate:
        if col not in df.columns:
            print(f"경고: 컬럼 '{col}'이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        print(f"컬럼 '{col}' 번역 중...")
        translated_col = f"{col}_ko"  # 번역된 컬럼명
        
        # 번역 실행
        df[translated_col] = translator.translate_batch(
            df[col].fillna("").astype(str).tolist(),
            batch_size=8
        )
        
        print(f"[OK] '{col}' → '{translated_col}' 번역 완료")
        print()
    
    # 결과 저장
    print(f"결과 저장: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("[OK] 저장 완료")
    print()


def translate_from_db(table_name, columns_to_translate, output_csv=None, model_size='4b', 
                      schema='public1', limit=None):
    """
    데이터베이스에서 데이터를 읽어서 번역
    
    Args:
        table_name: 테이블명
        columns_to_translate: 번역할 컬럼명 리스트
        output_csv: 출력 CSV 파일 경로 (None이면 DB에 업데이트)
        model_size: 모델 크기
        schema: 스키마명
        limit: 처리할 최대 레코드 수 (None이면 전체)
    """
    print("=" * 60)
    print("데이터베이스 번역 시작")
    print("=" * 60)
    
    # DB 연결
    print("데이터베이스 연결 중...")
    conn = psycopg2.connect(
        host=os.getenv('CRAWLING_DB_HOST'),
        port=os.getenv('CRAWLING_DB_PORT'),
        database=os.getenv('CRAWLING_DB_NAME'),
        user=os.getenv('CRAWLING_DB_USER'),
        password=os.getenv('CRAWLING_DB_PASSWORD')
    )
    print("[OK] 연결 완료")
    print()
    
    # 데이터 조회
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT *
    FROM {schema}.{table_name}
    {limit_clause}
    """
    
    print(f"데이터 조회 중: {schema}.{table_name}")
    df = pd.read_sql(query, conn)
    print(f"데이터 개수: {len(df):,}개")
    print()
    
    # 번역기 초기화
    translator = ShopeeTranslator(model_size=model_size)
    
    # 각 컬럼 번역
    for col in columns_to_translate:
        if col not in df.columns:
            print(f"경고: 컬럼 '{col}'이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        print(f"컬럼 '{col}' 번역 중...")
        translated_col = f"{col}_ko"
        
        # 번역 실행
        df[translated_col] = translator.translate_batch(
            df[col].fillna("").astype(str).tolist(),
            batch_size=8
        )
        
        print(f"[OK] '{col}' → '{translated_col}' 번역 완료")
        print()
    
    # 결과 처리
    if output_csv:
        # CSV로 저장
        print(f"결과 저장: {output_csv}")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print("[OK] 저장 완료")
    else:
        # DB에 업데이트 (번역된 컬럼 추가)
        print("데이터베이스에 번역 결과 저장 중...")
        cursor = conn.cursor()
        
        for col in columns_to_translate:
            translated_col = f"{col}_ko"
            if translated_col in df.columns:
                # 컬럼이 없으면 추가
                cursor.execute(f"""
                    ALTER TABLE {schema}.{table_name}
                    ADD COLUMN IF NOT EXISTS {translated_col} TEXT
                """)
                
                # 데이터 업데이트
                for idx, row in df.iterrows():
                    cursor.execute(f"""
                        UPDATE {schema}.{table_name}
                        SET {translated_col} = %s
                        WHERE id = %s
                    """, (row[translated_col], row['id']))
        
        conn.commit()
        cursor.close()
        print("[OK] DB 업데이트 완료")
    
    conn.close()
    print()


if __name__ == "__main__":
    # 사용 예시 1: CSV 파일 번역
    # translate_from_csv(
    #     input_csv='shopee_products.csv',
    #     output_csv='shopee_products_translated.csv',
    #     columns_to_translate=['name', 'description', 'category'],
    #     model_size='4b'  # 또는 '12b', '27b'
    # )
    
    # 사용 예시 2: 데이터베이스에서 번역
    # translate_from_db(
    #     table_name='shopee_products',
    #     columns_to_translate=['name', 'description'],
    #     output_csv='shopee_products_translated.csv',
    #     model_size='4b',
    #     limit=1000  # 테스트용으로 1000개만
    # )
    
    print("사용 예시:")
    print("1. CSV 파일 번역: translate_from_csv() 함수 사용")
    print("2. DB 번역: translate_from_db() 함수 사용")
    print()
    print("주의:")
    print("1. pip install transformers torch accelerate")
    print("2. Hugging Face 로그인: huggingface-cli login")
    print("3. Gemma 라이선스 동의 필요:")
    print("   - https://huggingface.co/google/translategemma-4b-it 방문")
    print("   - 'Agree and access repository' 클릭하여 라이선스 동의")
    print("4. 모델명: google/translategemma-4b-it (또는 12b-it, 27b-it)")
