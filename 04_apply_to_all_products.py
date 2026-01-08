# 04_apply_to_all_products.py
# 역할: 크롤링 DB → AI 예측 → 플랫폼 카테고리 매핑 → 플랫폼 DB 저장

import psycopg2
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

load_dotenv()

# 다나와 카테고리 ID → 플랫폼 카테고리 ID 매핑
DANAWA_TO_PLATFORM_MAPPING = {
    2: 14,   # 기저귀 → 기저귀
    3: 16,   # 물티슈 → 물티슈
    4: 11,   # 분유 → 분유
    5: 13,   # 유아간식/영양제 → 유아간식
    6: 12,   # 이유식/유아식 → 이유식/유아식
    7: 15,   # 천기저귀/용품 → 천기저귀/용품
    8: 26,   # 놀이방매트/안전용품 → 놀이방매트/베이비룸
    9: 22,   # 레고/블럭 → 레고/블럭
    10: 23,  # 로봇/배틀카드 → 로봇/배틀카드
    11: 21,  # 물놀이완구 → 인기 캐릭터완구 (가장 가까운 카테고리)
    12: 45,  # 신생아/영유아완구 → 신생아/영유아완구
    13: 26,  # 실내대형완구 → 놀이방매트/베이비룸 (가장 가까운 카테고리)
    14: 24,  # 역할놀이/소꿉놀이 → 역할놀이/소꿉놀이
    15: 21,  # 음악/미술놀이 → 인기 캐릭터완구 (가장 가까운 카테고리)
    16: 21,  # 인기 캐릭터완구 → 인기 캐릭터완구
    17: 25,  # 인형/피규어 → 인형
    18: 21,  # 자연/과학완구 → 인기 캐릭터완구 (가장 가까운 카테고리)
    19: 21,  # 킥보드/승용완구 → 인기 캐릭터완구 (가장 가까운 카테고리)
    20: 35,  # 아기띠/외출용품 → 아기띠/외출용품
    21: 31,  # 유모차 → 유모차
    22: 32,  # 유모차용품 → 유모차용품
    23: 33,  # 카시트 → 카시트
    24: 34,  # 카시트용품 → 카시트 용품
    # 25-31: 의류/도서 관련 (플랫폼 카테고리 없음 - 제외)
    32: 45,  # 신생아/영유아완구 (중복) → 신생아/영유아완구
    33: 43,  # 위생/목욕용품 → 위생/목욕용품
    34: 46,  # 유아가구 → 유아가구
    35: 42,  # 이유식용품 → 이유식용품
    36: 41,  # 젖병/수유용품 → 젖병/수유용품
    37: 44,  # 출산/신생아용품 → 출산/신생아용품
}

class ProductProcessor:
    def __init__(self, model_path='./finetuned_e5_large'):
        """AI 모델 및 DB 연결 초기화"""
        
        print("=" * 60)
        print("초기화")
        print("=" * 60)
        
        # AI 모델 로드
        print("AI 모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 메타데이터 로드
        with open(f'{model_path}/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.label_to_category = {int(k): v for k, v in metadata['label_to_category'].items()}
            self.max_length = metadata.get('max_length', 128)
        
        print(f"AI 모델 로드 완료 (Device: {self.device})")
        
        # 크롤링 DB 연결
        print("크롤링 DB 연결 중...")
        self.crawling_conn = psycopg2.connect(
            host=os.getenv('CRAWLING_DB_HOST'),
            port=os.getenv('CRAWLING_DB_PORT'),
            database=os.getenv('CRAWLING_DB_NAME'),
            user=os.getenv('CRAWLING_DB_USER'),
            password=os.getenv('CRAWLING_DB_PASSWORD')
        )
        print("크롤링 DB 연결 완료")
        
        # 플랫폼 DB 연결
        print("플랫폼 DB 연결 중...")
        self.platform_conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        print("플랫폼 DB 연결 완료")
        
        print("=" * 60)
    
    def predict_danawa_category(self, product_text):
        """AI로 다나와 카테고리 예측"""
        inputs = self.tokenizer(
            product_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        top_prob, top_idx = torch.max(probs, dim=0)
        danawa_category_id = self.label_to_category[top_idx.item()]
        confidence = top_prob.item()
        
        return danawa_category_id, confidence
    
    def process_all_products(self, batch_size=100):
        """
        크롤링 DB 모든 상품을 AI로 처리해서 플랫폼 DB에 저장
        
        흐름:
        1. AI로 다나와 카테고리 예측
        2. 다나와 카테고리 → 플랫폼 카테고리 매핑
        3. 사수 피드백 적용 (region, sales_count, revenue, representative_price)
        4. 플랫폼 DB에 저장
        """
        
        crawling_cursor = self.crawling_conn.cursor()
        platform_cursor = self.platform_conn.cursor()
        
        # 크롤링 DB에서 전체 상품 조회
        print("\n" + "=" * 60)
        print("크롤링 DB에서 상품 조회")
        print("=" * 60)
        
        crawling_cursor.execute("""
            SELECT 
                id,
                name,
                manufacturer,
                specifications::text,
                min_price,
                max_price,
                average_rating,
                review_count,
                url,
                product_code
            FROM products
            WHERE name IS NOT NULL
            ORDER BY id
        """)
        
        products = crawling_cursor.fetchall()
        total_count = len(products)
        
        print(f"총 {total_count:,}개 상품 조회 완료")
        
        # 배치 처리
        print("\n" + "=" * 60)
        print("AI 예측 및 플랫폼 DB 저장")
        print("=" * 60)
        
        success_count = 0
        error_count = 0
        mapping_fail_count = 0
        
        for i in range(0, total_count, batch_size):
            batch = products[i:i+batch_size]
            
            for product in batch:
                try:
                    (crawling_id, name, manufacturer, specifications, 
                     min_price, max_price, average_rating, review_count, 
                     url, product_code) = product
                    
                    # AI로 다나와 카테고리 예측
                    product_text = f"{name} {manufacturer or ''} {specifications or ''}".strip()
                    danawa_category_id, confidence = self.predict_danawa_category(product_text)
                    
                    # 다나와 카테고리 → 플랫폼 카테고리 매핑
                    platform_category_id = DANAWA_TO_PLATFORM_MAPPING.get(danawa_category_id)
                    
                    if platform_category_id is None:
                        # 매핑 안 됨 (의류/도서 등) - 건너뛰기
                        mapping_fail_count += 1
                        continue
                    
                    # 사수 피드백 반영
                    sales_count = review_count if review_count else 0
                    
                    if sales_count > 0 and min_price and min_price > 0:
                        revenue = sales_count * min_price
                    else:
                        revenue = 0
                    
                    # 플랫폼 DB에 INSERT
                    platform_cursor.execute("""
                        INSERT INTO products (
                            name,
                            manufacturer,
                            specifications,
                            category_id,          -- 플랫폼 카테고리 ID
                            min_price,
                            max_price,
                            average_rating,
                            review_count,
                            url,
                            product_code,
                            region,               -- NULL (사수 피드백)
                            sales_count,          -- review_count (사수 피드백)
                            revenue,              -- 계산값 (사수 피드백)
                            representative_price, -- NULL (사수 피드백)
                            ai_confidence,
                            ai_predicted_danawa_category,
                            crawling_source_id
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            NULL, %s, %s, NULL, %s, %s, %s
                        )
                        ON CONFLICT (product_code) 
                        DO UPDATE SET
                            category_id = EXCLUDED.category_id,
                            sales_count = EXCLUDED.sales_count,
                            revenue = EXCLUDED.revenue,
                            ai_confidence = EXCLUDED.ai_confidence,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        name,
                        manufacturer,
                        specifications,
                        platform_category_id,  # 매핑된 플랫폼 카테고리
                        min_price,
                        max_price,
                        average_rating,
                        review_count,
                        url,
                        product_code,
                        sales_count,           # 사수 피드백
                        revenue,               # 사수 피드백
                        confidence,
                        danawa_category_id,
                        crawling_id
                    ))
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"오류 (상품 ID {crawling_id}): {str(e)[:100]}")
                    error_count += 1
            
            # 배치마다 커밋
            self.platform_conn.commit()
            
            # 진행 상황 출력
            current = min(i + batch_size, total_count)
            progress = current / total_count * 100
            print(f"진행: {current:,} / {total_count:,} ({progress:.1f}%) | 성공: {success_count:,} | 실패: {error_count:,} | 매핑실패: {mapping_fail_count:,}")
        
        print("\n" + "=" * 60)
        print("처리 완료")
        print("=" * 60)
        print(f"총 처리: {total_count:,}개")
        print(f"성공: {success_count:,}개")
        print(f"실패: {error_count:,}개")
        print(f"매핑 실패: {mapping_fail_count:,}개 (의류/도서 카테고리)")
        
        # 연결 종료
        crawling_cursor.close()
        platform_cursor.close()
        self.crawling_conn.close()
        self.platform_conn.close()

if __name__ == "__main__":
    processor = ProductProcessor()
    processor.process_all_products(batch_size=100)