# test_translategemma_ko_to_vi_improved.py
# 역할: TranslateGemma 번역 시스템 (전처리/후처리 통합 버전)

import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from tqdm import tqdm
import os
import sys
from translation_preprocessor import TranslationPreProcessor
from translation_postprocessor import TranslationPostProcessor

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class ImprovedTranslateGemmaTester:
    def __init__(self, model_size='4b', device='cuda'):
        """
        개선된 TranslateGemma 모델 초기화
        
        Args:
            model_size: '4b', '12b', '27b' 중 선택
            device: 'cuda' 또는 'cpu'
        """
        print("=" * 60)
        print("개선된 TranslateGemma 모델 로딩 중...")
        print("=" * 60)
        
        # 모델 이름 매핑
        model_map = {
            '4b': 'google/translategemma-4b-it',
            '12b': 'google/translategemma-12b-it',
            '27b': 'google/translategemma-27b-it'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['4b'])
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"모델: {model_name}")
        print(f"디바이스: {self.device}")
        print()
        
        # 전처리/후처리 초기화
        self.preprocessor = TranslationPreProcessor()
        self.postprocessor = TranslationPostProcessor()
        
        try:
            # Hugging Face 토큰 확인
            hf_token = os.getenv('HF_TOKEN')
            
            if not hf_token:
                print("[경고] HF_TOKEN 환경 변수가 설정되지 않았습니다.")
            else:
                print(f"[OK] Hugging Face 토큰 확인됨")
                try:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=False)
                    print("[OK] Hugging Face 로그인 완료")
                except Exception as login_error:
                    print(f"[경고] 로그인 실패: {login_error}")
            
            # 모델 및 프로세서 로드
            print("프로세서 로딩 중...")
            if hf_token:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    token=hf_token
                )
            else:
                self.processor = AutoProcessor.from_pretrained(model_name)
            
            print("모델 로딩 중... (시간이 걸릴 수 있습니다)")
            if hf_token:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    token=hf_token,
                    torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
                    device_map='auto' if self.device.type == 'cuda' else None
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
                    device_map='auto' if self.device.type == 'cuda' else None
                )
            
            if self.device.type == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            print("[OK] 모델 로드 완료")
            print()
        except Exception as e:
            print(f"[오류] 모델 로드 실패: {e}")
            raise
    
    def translate(self, korean_text, source_lang='ko', target_lang='vi', 
                 max_length=32, original_brand=None, use_preprocessing=True):
        """
        개선된 번역 함수 (전처리/후처리 포함)
        
        Args:
            korean_text: 번역할 한국어 텍스트
            source_lang: 원본 언어 코드
            target_lang: 목표 언어 코드
            max_length: 최대 생성 길이
            original_brand: 원본 브랜드명 (후처리용)
            use_preprocessing: 전처리 사용 여부
        
        Returns:
            번역된 베트남어 텍스트
        """
        if pd.isna(korean_text) or not str(korean_text).strip():
            return ""
        
        original_text = str(korean_text).strip()
        text_to_translate = original_text
        replacements = {}
        
        # 전처리: 고유명사 보호
        if use_preprocessing:
            text_to_translate, replacements = self.preprocessor.preprocess(original_text)
        
        try:
            # TranslateGemma chat template 형식
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": text_to_translate
                        }
                    ]
                }
            ]
            
            # Processor를 사용하여 chat template 적용 및 토크나이징
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # 디바이스로 이동
            dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
            inputs = {k: v.to(self.device).to(dtype) if v.dtype.is_floating_point else v.to(self.device) 
                     for k, v in inputs.items()}
            
            input_length = len(inputs['input_ids'][0])
            
            # 번역 생성
            with torch.no_grad():
                generation = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_length,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 디코딩
            generation = generation[0][input_length:]
            translated = self.processor.decode(generation, skip_special_tokens=True)
            translated = translated.strip()
            
            # 후처리 1: 플레이스홀더 복원
            if use_preprocessing and replacements:
                translated = self.preprocessor.restore(translated, replacements, use_standard_names=True)
            
            # 후처리 2: 추가 문제 수정
            translated = self.postprocessor.process_translation(
                original_text, translated, original_brand
            )
            
            return translated
        except Exception as e:
            print(f"번역 오류 (텍스트: {original_text[:50]}...): {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def translate_batch(self, texts, brands=None, batch_size=1, 
                       source_lang='ko', target_lang='vi', use_preprocessing=True):
        """
        여러 텍스트를 배치로 번역
        
        Args:
            texts: 번역할 텍스트 리스트
            brands: 브랜드명 리스트 (선택사항)
            batch_size: 배치 크기 (현재는 1)
            source_lang: 원본 언어
            target_lang: 목표 언어
            use_preprocessing: 전처리 사용 여부
        
        Returns:
            번역된 텍스트 리스트
        """
        results = []
        
        if brands is None:
            brands = [None] * len(texts)
        
        for i in tqdm(range(len(texts)), desc="번역 진행 중"):
            text = texts[i]
            brand = brands[i] if i < len(brands) else None
            translated = self.translate(
                text, source_lang, target_lang, 
                max_length=32, original_brand=brand, 
                use_preprocessing=use_preprocessing
            )
            results.append(translated)
            
            # GPU 메모리 정리
            if self.device.type == 'cuda' and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        return results


def test_translate_csv_improved(input_csv='products_all_categorized.csv', 
                                output_csv='products_all_categorized_translated_vi_improved.csv',
                                sample_size=10,
                                model_size='4b',
                                columns_to_translate=['name', 'manufacturer', 'category_level1', 'category_level2', 'category_level3'],
                                use_preprocessing=True,
                                save_after_each_column=True):
    """
    개선된 CSV 번역 함수
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
        sample_size: 테스트할 샘플 수
        model_size: 모델 크기
        columns_to_translate: 번역할 컬럼명 리스트
        use_preprocessing: 전처리 사용 여부
        save_after_each_column: 컬럼마다 번역 후 즉시 저장 (타임아웃/중단 시에도 결과 보존)
    """
    print("=" * 60)
    print("개선된 TranslateGemma 테스트: 한국어 → 베트남어")
    print("=" * 60)
    print()
    
    # CSV 로드
    print(f"CSV 파일 로딩: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"[오류] 파일을 찾을 수 없습니다: {input_csv}")
        return
    
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"전체 데이터 개수: {len(df):,}개")
    print()
    
    # 샘플링
    if sample_size and sample_size < len(df):
        print(f"샘플 {sample_size}개 추출 중...")
        df_sample = df.head(sample_size).copy()
    else:
        df_sample = df.copy()
    
    print(f"번역할 데이터: {len(df_sample):,}개")
    print()
    
    # 번역기 초기화
    translator = ImprovedTranslateGemmaTester(model_size=model_size)
    
    # 각 컬럼 번역
    for col in columns_to_translate:
        if col not in df_sample.columns:
            print(f"경고: 컬럼 '{col}'이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        print(f"컬럼 '{col}' 번역 중...")
        translated_col = f"{col}_vi"
        
        # 브랜드명 가져오기 (manufacturer 컬럼이 있는 경우)
        brands = None
        if 'manufacturer' in df_sample.columns:
            brands = df_sample['manufacturer'].fillna("").astype(str).tolist()
        
        # 번역 실행
        texts = df_sample[col].fillna("").astype(str).tolist()
        df_sample[translated_col] = translator.translate_batch(
            texts,
            brands=brands if col == 'name' else None,  # name 컬럼에만 브랜드명 전달
            batch_size=1,
            source_lang='ko',
            target_lang='vi',
            use_preprocessing=use_preprocessing
        )
        
        # 번역 결과 샘플 출력
        print(f"[OK] '{col}' → '{translated_col}' 번역 완료")
        print("번역 샘플 (처음 3개):")
        for idx in range(min(3, len(df_sample))):
            original = df_sample[col].iloc[idx]
            translated = df_sample[translated_col].iloc[idx]
            if original and translated:
                print(f"  [{idx+1}]")
                try:
                    print(f"    원본 (한국어): {original[:80]}...")
                    # 베트남어 출력 시 인코딩 오류 방지
                    translated_display = str(translated)[:80]
                    print(f"    번역 (베트남어): {translated_display}...")
                except UnicodeEncodeError:
                    # 인코딩 오류 시 간단히 표시
                    print(f"    원본 (한국어): [출력됨]")
                    print(f"    번역 (베트남어): [출력됨]")
        print()
        
        # 컬럼마다 즉시 저장 (타임아웃/중단 시에도 결과 보존)
        if save_after_each_column:
            df_sample.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"  [저장] {output_csv} (현재까지 {len([c for c in columns_to_translate if f'{c}_vi' in df_sample.columns])}개 컬럼)")
            print()
    
    # 최종 저장 (save_after_each_column=True면 이미 저장됐지만 한 번 더)
    print(f"결과 저장: {output_csv}")
    df_sample.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("[OK] 저장 완료")
    print()


if __name__ == "__main__":
    # 개선된 번역 테스트 (컬럼마다 즉시 저장 → 타임아웃/중단 시에도 결과 보존)
    test_translate_csv_improved(
        input_csv='products_all_categorized.csv',
        output_csv='products_all_categorized_translated_vi_improved.csv',
        sample_size=10,
        model_size='4b',
        columns_to_translate=['name', 'manufacturer', 'category_level1', 'category_level2', 'category_level3'],
        use_preprocessing=True,
        save_after_each_column=True
    )
    
    print("=" * 60)
    print("개선된 번역 테스트 완료!")
    print("=" * 60)
