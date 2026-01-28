# fix_existing_translations.py
# 역할: 기존 번역 결과(products_all_categorized_translated_vi.csv)의 문제를 수정

import pandas as pd
from translation_postprocessor import TranslationPostProcessor
import os

def fix_existing_translations(input_csv='products_all_categorized_translated_vi.csv',
                              output_csv='products_all_categorized_translated_vi_fixed.csv'):
    """
    기존 번역 결과의 문제를 수정
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
    """
    print("=" * 60)
    print("기존 번역 결과 수정")
    print("=" * 60)
    print()
    
    # CSV 로드
    if not os.path.exists(input_csv):
        print(f"[오류] 파일을 찾을 수 없습니다: {input_csv}")
        return
    
    print(f"CSV 파일 로딩: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"총 데이터: {len(df):,}개")
    print()
    
    # 후처리기 초기화
    processor = TranslationPostProcessor()
    
    # 번역 컬럼과 원본 컬럼 매핑
    column_pairs = [
        ('name', 'name_vi'),
        ('manufacturer', 'manufacturer_vi'),
        ('category_level1', 'category_level1_vi'),
        ('category_level2', 'category_level2_vi'),
        ('category_level3', 'category_level3_vi')
    ]
    
    total_fixed = 0
    
    # 각 컬럼 쌍 처리
    for orig_col, trans_col in column_pairs:
        if orig_col not in df.columns or trans_col not in df.columns:
            print(f"경고: 컬럼 '{orig_col}' 또는 '{trans_col}'이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        print(f"처리 중: {orig_col} → {trans_col}")
        
        fixed_count = 0
        for idx in df.index:
            original = df.loc[idx, orig_col]
            translated = df.loc[idx, trans_col]
            
            # 브랜드명 가져오기 (manufacturer 컬럼이 있는 경우)
            brand = None
            if 'manufacturer' in df.columns:
                brand = df.loc[idx, 'manufacturer']
            
            if pd.notna(original) and pd.notna(translated):
                fixed = processor.process_translation(
                    str(original),
                    str(translated),
                    str(brand) if pd.notna(brand) else None
                )
                
                if fixed != str(translated):
                    df.loc[idx, trans_col] = fixed
                    fixed_count += 1
        
        print(f"  수정된 항목: {fixed_count}개")
        total_fixed += fixed_count
        print()
    
    # 결과 저장
    print(f"결과 저장: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"[OK] 저장 완료 (총 {total_fixed}개 항목 수정)")
    print()
    
    # 수정 사항 샘플 출력
    print("=" * 60)
    print("수정 사항 샘플 (처음 5개)")
    print("=" * 60)
    
    sample_count = 0
    for orig_col, trans_col in column_pairs:
        if orig_col not in df.columns or trans_col not in df.columns:
            continue
        
        for idx in df.index:
            if sample_count >= 5:
                break
            
            original = df.loc[idx, orig_col]
            translated = df.loc[idx, trans_col]
            
            if pd.notna(original) and pd.notna(translated):
                # 원본 CSV에서 원래 번역 결과 가져오기 (비교용)
                original_df = pd.read_csv(input_csv, encoding='utf-8')
                original_translated = original_df.loc[idx, trans_col] if trans_col in original_df.columns else None
                
                if original_translated and str(translated) != str(original_translated):
                    print(f"\n[{sample_count + 1}] 컬럼: {orig_col}")
                    print(f"  원본: {str(original)[:60]}...")
                    print(f"  수정 전: {str(original_translated)[:60]}...")
                    print(f"  수정 후: {str(translated)[:60]}...")
                    sample_count += 1
    
    print()


if __name__ == "__main__":
    fix_existing_translations(
        input_csv='products_all_categorized_translated_vi.csv',
        output_csv='products_all_categorized_translated_vi_fixed.csv'
    )
    
    print("=" * 60)
    print("수정 완료!")
    print("=" * 60)
    print()
    print("다음 단계:")
    print("1. 수정된 파일 확인: products_all_categorized_translated_vi_fixed.csv")
    print("2. 번역 품질 검증")
    print("3. 필요시 추가 수정")
