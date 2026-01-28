# translation_postprocessor.py
# 역할: TranslateGemma 번역 결과의 문제를 자동으로 수정하는 후처리 시스템

import pandas as pd
import re
import json
from typing import Dict, List, Tuple

class TranslationPostProcessor:
    """
    번역 결과의 문제를 자동으로 수정하는 후처리 클래스
    
    주요 기능:
    1. 브랜드명/고유명사 복원
    2. 키워드 오번역 수정
    3. 숫자/정보 누락 복원
    """
    
    def __init__(self, brand_dict_path=None):
        """
        초기화
        
        Args:
            brand_dict_path: 브랜드명 사전 JSON 파일 경로 (선택사항)
        """
        # 브랜드명 사전 (한국어 → 베트남어 표준명 또는 원본 보존)
        self.brand_dict = {
            '그린키즈': 'Green Kids',
            '반달': 'Bandal',
            '어스본': 'Usborne',
            '지학사': 'Jihak Publishing',
            '아가짱': 'Agachan',
            '폴레드': 'Poled',
            '에르고베이비': 'Ergobaby',
            '새들베이비': 'SaddleBaby',
            '헬로미니미': 'Hello Mini'
        }
        
        # 브랜드명 사전 파일이 있으면 로드
        if brand_dict_path and os.path.exists(brand_dict_path):
            with open(brand_dict_path, 'r', encoding='utf-8') as f:
                custom_dict = json.load(f)
                self.brand_dict.update(custom_dict)
        
        # 키워드 오번역 수정 사전
        self.keyword_fixes = {
            'khoai tây': 'khoai lang',  # 감자 → 고구마
            'quần áo trẻ em tiện lợi': None,  # 기저귀 체크 → 특별 처리 필요
        }
        
        # 숫자 패턴 (1년, 2년 등)
        self.number_pattern = re.compile(r'(\d+)\s*년')
        
    def fix_brand_names(self, original_text: str, translated_text: str, original_brand: str = None) -> str:
        """
        브랜드명 복원
        
        Args:
            original_text: 원본 한국어 텍스트
            translated_text: 번역된 베트남어 텍스트
            original_brand: 원본 브랜드명 (manufacturer 컬럼 값)
        
        Returns:
            수정된 번역 텍스트
        """
        result = translated_text
        
        # 원본 브랜드명이 있으면 우선 사용
        if original_brand and original_brand in self.brand_dict:
            brand_standard = self.brand_dict[original_brand]
            # 번역된 텍스트에서 브랜드명이 잘못 번역된 부분 찾기
            # 간단한 휴리스틱: 원본 텍스트에 브랜드명이 포함되어 있으면 복원
            if original_brand in original_text:
                # 번역된 텍스트에서 브랜드명 관련 부분 찾아서 교체
                # 여러 패턴 시도
                patterns_to_replace = [
                    # 잘못 번역된 패턴들
                    r'Hình mặt trăng',  # 반달
                    r'Chào bạn nhỏ',  # 헬로미니미
                    r'Bé yêu',  # 아가짱
                    r'Sẻ bé',  # 새들베이비
                    r'Green Kids',  # 그린키즈 (의도적일 수 있으나 일관성 유지)
                    r'Earthson',  # 어스본
                    r'Nha xuất bản Địa Học',  # 지학사
                    r'Poléd',  # 폴레드
                ]
                
                for pattern in patterns_to_replace:
                    if re.search(pattern, result, re.IGNORECASE):
                        # 원본 브랜드명이 사전에 있으면 표준명으로 교체
                        result = re.sub(pattern, brand_standard, result, flags=re.IGNORECASE)
                        break
        
        # 브랜드명 사전 기반 자동 교체
        for korean_brand, standard_name in self.brand_dict.items():
            if korean_brand in original_text:
                # 잘못 번역된 패턴 찾아서 교체
                # 이 부분은 더 정교한 패턴 매칭이 필요할 수 있음
                pass  # 위에서 이미 처리했으므로 생략
        
        return result
    
    def fix_keyword_errors(self, original_text: str, translated_text: str) -> str:
        """
        키워드 오번역 수정
        
        Args:
            original_text: 원본 한국어 텍스트
            translated_text: 번역된 베트남어 텍스트
        
        Returns:
            수정된 번역 텍스트
        """
        result = translated_text
        
        # 고구마 → 감자 오번역 수정
        if '고구마' in original_text and 'khoai tây' in result:
            result = result.replace('khoai tây', 'khoai lang')
        
        # 기저귀 키워드 누락 복원
        if '기저귀' in original_text:
            # 번역에 기저귀 관련 단어가 없으면 추가
            if 'tã' not in result.lower() and 'bỉm' not in result.lower():
                # 원본 텍스트의 맥락에 따라 적절히 추가
                if '체크' in original_text:
                    # "기저귀 체크" → "Kiểm tra tã" 또는 유사한 표현
                    # "Kiểm tra quần áo trẻ em tiện lợi" → "Kiểm tra tã"로 교체
                    if 'Kiểm tra' in result and 'quần áo trẻ em tiện lợi' in result:
                        # "Kiểm tra"가 이미 있으면 "quần áo trẻ em tiện lợi"만 "tã"로 교체
                        result = result.replace('quần áo trẻ em tiện lợi', 'tã')
                    else:
                        # "quần áo trẻ em tiện lợi"를 "tã"로 교체
                        result = result.replace('quần áo trẻ em tiện lợi', 'tã')
                else:
                    # 기저귀 키워드 추가
                    result = f"{result} (tã)"
        
        return result
    
    def fix_missing_numbers(self, original_text: str, translated_text: str) -> str:
        """
        숫자/정보 누락 복원
        
        Args:
            original_text: 원본 한국어 텍스트
            translated_text: 번역된 베트남어 텍스트
        
        Returns:
            수정된 번역 텍스트
        """
        result = translated_text
        
        # "1년", "2년" 등 숫자+년 패턴 찾기
        year_matches = self.number_pattern.findall(original_text)
        if year_matches:
            for year_num in year_matches:
                # 번역에 해당 숫자가 없으면 추가
                if year_num not in result and 'năm' in result.lower():
                    # "hàng năm" (매년) → "1 năm" (1년) 등으로 수정
                    result = result.replace('hàng năm', f'{year_num} năm')
                elif year_num not in result:
                    # 숫자가 완전히 누락된 경우 추가
                    result = f"{result} ({year_num} năm)"
        
        return result
    
    def process_translation(self, original_text: str, translated_text: str, 
                          original_brand: str = None) -> str:
        """
        번역 결과 전체 후처리
        
        Args:
            original_text: 원본 한국어 텍스트
            translated_text: 번역된 베트남어 텍스트
            original_brand: 원본 브랜드명
        
        Returns:
            수정된 번역 텍스트
        """
        if pd.isna(translated_text) or not str(translated_text).strip():
            return translated_text
        
        result = str(translated_text).strip()
        
        # 1. 브랜드명 복원
        result = self.fix_brand_names(original_text, result, original_brand)
        
        # 2. 키워드 오번역 수정
        result = self.fix_keyword_errors(original_text, result)
        
        # 3. 숫자/정보 누락 복원
        result = self.fix_missing_numbers(original_text, result)
        
        return result
    
    def process_csv(self, input_csv: str, output_csv: str, 
                   original_columns: List[str], translated_columns: List[str],
                   brand_column: str = 'manufacturer'):
        """
        CSV 파일의 번역 결과를 일괄 후처리
        
        Args:
            input_csv: 입력 CSV 파일 경로
            output_csv: 출력 CSV 파일 경로
            original_columns: 원본 컬럼명 리스트 (예: ['name', 'category_level1'])
            translated_columns: 번역된 컬럼명 리스트 (예: ['name_vi', 'category_level1_vi'])
            brand_column: 브랜드명 컬럼명 (기본값: 'manufacturer')
        """
        print("=" * 60)
        print("번역 결과 후처리 시작")
        print("=" * 60)
        print()
        
        # CSV 로드
        print(f"CSV 파일 로딩: {input_csv}")
        df = pd.read_csv(input_csv, encoding='utf-8')
        print(f"총 데이터: {len(df):,}개")
        print()
        
        # 각 컬럼 쌍 처리
        for orig_col, trans_col in zip(original_columns, translated_columns):
            if orig_col not in df.columns or trans_col not in df.columns:
                print(f"경고: 컬럼 '{orig_col}' 또는 '{trans_col}'이 존재하지 않습니다. 건너뜁니다.")
                continue
            
            print(f"처리 중: {orig_col} → {trans_col}")
            
            # 브랜드명 컬럼 확인
            brand_col = brand_column if brand_column in df.columns else None
            
            # 각 행 처리
            fixed_count = 0
            for idx in df.index:
                original = df.loc[idx, orig_col]
                translated = df.loc[idx, trans_col]
                brand = df.loc[idx, brand_col] if brand_col else None
                
                if pd.notna(original) and pd.notna(translated):
                    fixed = self.process_translation(
                        str(original),
                        str(translated),
                        str(brand) if pd.notna(brand) else None
                    )
                    
                    if fixed != translated:
                        df.loc[idx, trans_col] = fixed
                        fixed_count += 1
            
            print(f"  수정된 항목: {fixed_count}개")
            print()
        
        # 결과 저장
        print(f"결과 저장: {output_csv}")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print("[OK] 저장 완료")
        print()


if __name__ == "__main__":
    import os
    
    # 사용 예시
    processor = TranslationPostProcessor()
    
    # CSV 파일 후처리
    processor.process_csv(
        input_csv='products_all_categorized_translated_vi.csv',
        output_csv='products_all_categorized_translated_vi_fixed.csv',
        original_columns=['name', 'manufacturer', 'category_level1', 'category_level2', 'category_level3'],
        translated_columns=['name_vi', 'manufacturer_vi', 'category_level1_vi', 'category_level2_vi', 'category_level3_vi'],
        brand_column='manufacturer'
    )
    
    print("=" * 60)
    print("후처리 완료!")
    print("=" * 60)
