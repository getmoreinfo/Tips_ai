# translation_preprocessor.py
# 역할: TranslateGemma 번역 전에 고유명사를 보존하기 위한 전처리 시스템

import pandas as pd
import re
import json
from typing import Dict, List, Tuple

class TranslationPreProcessor:
    """
    번역 전에 고유명사를 보존하기 위한 전처리 클래스
    
    주요 기능:
    1. 브랜드명/고유명사를 플레이스홀더로 치환
    2. 번역 후 플레이스홀더를 원본으로 복원
    """
    
    def __init__(self, brand_dict_path=None):
        """
        초기화
        
        Args:
            brand_dict_path: 브랜드명 사전 JSON 파일 경로 (선택사항)
        """
        # 브랜드명 사전 (한국어 → 플레이스홀더)
        self.brand_dict = {
            '그린키즈': 'BRAND_GREENKIDS',
            '반달': 'BRAND_BANDAL',
            '어스본': 'BRAND_USBORNE',
            '지학사': 'BRAND_JIHAK',
            '아가짱': 'BRAND_AGACHAN',
            '폴레드': 'BRAND_POLED',
            '에르고베이비': 'BRAND_ERGOBABY',
            '새들베이비': 'BRAND_SADDLEBABY',
            '헬로미니미': 'BRAND_HELLOMINI'
        }
        
        # 브랜드명 사전 파일이 있으면 로드
        if brand_dict_path:
            import os
            if os.path.exists(brand_dict_path):
                with open(brand_dict_path, 'r', encoding='utf-8') as f:
                    custom_dict = json.load(f)
                    self.brand_dict.update(custom_dict)
        
        # 플레이스홀더 → 원본 매핑 (복원용)
        self.placeholder_to_original = {v: k for k, v in self.brand_dict.items()}
        
        # 플레이스홀더 → 표준명 매핑 (복원 시 사용)
        self.placeholder_to_standard = {
            'BRAND_GREENKIDS': 'Green Kids',
            'BRAND_BANDAL': 'Bandal',
            'BRAND_USBORNE': 'Usborne',
            'BRAND_JIHAK': 'Jihak Publishing',
            'BRAND_AGACHAN': 'Agachan',
            'BRAND_POLED': 'Poled',
            'BRAND_ERGOBABY': 'Ergobaby',
            'BRAND_SADDLEBABY': 'SaddleBaby',
            'BRAND_HELLOMINI': 'Hello Mini'
        }
        
        # 특수 키워드 보존 (고구마 등)
        self.special_keywords = {
            '고구마': 'KEYWORD_SWEETPOTATO',
            '기저귀': 'KEYWORD_DIAPER'
        }
        
        self.keyword_to_translation = {
            'KEYWORD_SWEETPOTATO': 'khoai lang',  # 고구마
            'KEYWORD_DIAPER': 'tã'  # 기저귀
        }
    
    def protect_brand_names(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        텍스트에서 브랜드명을 플레이스홀더로 치환
        
        Args:
            text: 원본 텍스트
        
        Returns:
            (치환된 텍스트, 치환 정보 딕셔너리)
        """
        if pd.isna(text) or not str(text).strip():
            return text, {}
        
        result = str(text)
        replacements = {}
        
        # 브랜드명 치환 (긴 것부터 먼저 처리)
        sorted_brands = sorted(self.brand_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for brand, placeholder in sorted_brands:
            if brand in result:
                result = result.replace(brand, placeholder)
                replacements[placeholder] = brand
        
        return result, replacements
    
    def protect_keywords(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        특수 키워드를 플레이스홀더로 치환
        
        Args:
            text: 원본 텍스트
        
        Returns:
            (치환된 텍스트, 치환 정보 딕셔너리)
        """
        if pd.isna(text) or not str(text).strip():
            return text, {}
        
        result = str(text)
        replacements = {}
        
        for keyword, placeholder in self.special_keywords.items():
            if keyword in result:
                result = result.replace(keyword, placeholder)
                replacements[placeholder] = keyword
        
        return result, replacements
    
    def preprocess(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        번역 전 전체 전처리
        
        Args:
            text: 원본 텍스트
        
        Returns:
            (전처리된 텍스트, 치환 정보 딕셔너리)
        """
        if pd.isna(text) or not str(text).strip():
            return text, {}
        
        result = str(text)
        all_replacements = {}
        
        # 1. 브랜드명 보호
        result, brand_replacements = self.protect_brand_names(result)
        all_replacements.update(brand_replacements)
        
        # 2. 특수 키워드 보호
        result, keyword_replacements = self.protect_keywords(result)
        all_replacements.update(keyword_replacements)
        
        return result, all_replacements
    
    def restore(self, translated_text: str, replacements: Dict[str, str], 
                use_standard_names: bool = True) -> str:
        """
        번역 후 플레이스홀더를 원본 또는 표준명으로 복원
        
        Args:
            translated_text: 번역된 텍스트
            replacements: 치환 정보 딕셔너리
            use_standard_names: True면 표준명 사용, False면 원본 보존
        
        Returns:
            복원된 텍스트
        """
        if pd.isna(translated_text) or not str(translated_text).strip():
            return translated_text
        
        result = str(translated_text)
        
        # 플레이스홀더 복원
        for placeholder, original in replacements.items():
            if placeholder in result:
                if placeholder.startswith('BRAND_'):
                    # 브랜드명: 표준명 또는 원본 사용
                    if use_standard_names and placeholder in self.placeholder_to_standard:
                        replacement = self.placeholder_to_standard[placeholder]
                    else:
                        replacement = original
                elif placeholder.startswith('KEYWORD_'):
                    # 키워드: 번역된 단어 사용
                    if placeholder in self.keyword_to_translation:
                        replacement = self.keyword_to_translation[placeholder]
                    else:
                        replacement = original
                else:
                    replacement = original
                
                result = result.replace(placeholder, replacement)
        
        return result


def preprocess_for_translation(text: str, preprocessor: TranslationPreProcessor) -> Tuple[str, Dict]:
    """
    번역 전 전처리 헬퍼 함수
    
    Args:
        text: 원본 텍스트
        preprocessor: TranslationPreProcessor 인스턴스
    
    Returns:
        (전처리된 텍스트, 치환 정보)
    """
    return preprocessor.preprocess(text)


def restore_after_translation(translated_text: str, replacements: Dict, 
                             preprocessor: TranslationPreProcessor,
                             use_standard_names: bool = True) -> str:
    """
    번역 후 복원 헬퍼 함수
    
    Args:
        translated_text: 번역된 텍스트
        replacements: 치환 정보
        preprocessor: TranslationPreProcessor 인스턴스
        use_standard_names: 표준명 사용 여부
    
    Returns:
        복원된 텍스트
    """
    return preprocessor.restore(translated_text, replacements, use_standard_names)


if __name__ == "__main__":
    # 사용 예시
    preprocessor = TranslationPreProcessor()
    
    # 테스트
    test_text = "그린키즈 요술지팡이 이솝우화"
    print(f"원본: {test_text}")
    
    preprocessed, replacements = preprocessor.preprocess(test_text)
    print(f"전처리 후: {preprocessed}")
    print(f"치환 정보: {replacements}")
    print()
    
    # 번역 시뮬레이션 (실제로는 TranslateGemma 사용)
    simulated_translation = "Bộ truyện cổ tích BRAND_GREENKIDS với cây đũa phép của Aesop"
    print(f"번역 결과 (플레이스홀더 포함): {simulated_translation}")
    
    restored = preprocessor.restore(simulated_translation, replacements, use_standard_names=True)
    print(f"복원 후: {restored}")
