# test_translategemma_ko_to_vi.py
# 역할: TranslateGemma를 테스트하기 위해 products_all_categorized.csv의 한국어 데이터를 베트남어로 번역

import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from tqdm import tqdm
import os

class TranslateGemmaTester:
    def __init__(self, model_size='4b', device='cuda'):
        """
        TranslateGemma 모델 초기화 (한국어 → 베트남어)
        
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
        print()
        
        try:
            # Hugging Face 토큰 확인 (환경 변수 또는 직접 설정)
            import os
            hf_token = os.getenv('HF_TOKEN')
            
            if not hf_token:
                print("[경고] HF_TOKEN 환경 변수가 설정되지 않았습니다.")
                print("PowerShell에서 다음 명령어 실행:")
                print('  $env:HF_TOKEN="여기에_토큰_붙여넣기"')
                print()
                print("또는 Hugging Face에 로그인하세요:")
                print("  huggingface-cli login")
                print()
            else:
                print(f"[OK] Hugging Face 토큰 확인됨 (토큰 시작: {hf_token[:7]}...)")
                print()
            
            # Hugging Face 로그인 시도 (토큰이 있으면)
            if hf_token:
                try:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=False)
                    print("[OK] Hugging Face 로그인 완료")
                    print()
                except Exception as login_error:
                    print(f"[경고] 로그인 실패: {login_error}")
                    print("토큰을 확인하거나 Gemma 라이선스에 동의했는지 확인하세요.")
                    print()
            
            # 모델 및 프로세서 로드 (TranslateGemma는 Processor 사용)
            print("프로세서 로딩 중...")
            if hf_token:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    token=hf_token
                )
            else:
                # 토큰이 없어도 시도 (이미 로그인되어 있을 수 있음)
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
            print()
            print("=" * 60)
            print("해결 방법 (순서대로 진행)")
            print("=" * 60)
            print()
            print("1. Hugging Face 토큰 설정 (PowerShell):")
            print('   $env:HF_TOKEN="여기에_토큰_붙여넣기"')
            print()
            print("2. Gemma 라이선스 동의 (필수!):")
            print("   - 브라우저에서 https://huggingface.co/google/translategemma-4b-it 방문")
            print("   - 'Agree and access repository' 버튼 클릭")
            print("   - 로그인되어 있어야 합니다")
            print()
            print("3. 토큰 권한 확인:")
            print("   - https://huggingface.co/settings/tokens 방문")
            print("   - '접근 가능한 모든 공개 제한 저장소의 콘텐츠에 대한 읽기 권한' 체크")
            print()
            print("4. 재시도:")
            print("   python test_translategemma_ko_to_vi.py")
            print()
            print("참고: Gemma 라이선스 동의는 필수입니다!")
            raise
    
    def translate(self, korean_text, source_lang='ko', target_lang='vi', max_length=32):
        """
        한국어 텍스트를 베트남어로 번역
        
        Args:
            korean_text: 번역할 한국어 텍스트
            source_lang: 원본 언어 코드 (ko = 한국어)
            target_lang: 목표 언어 코드 (vi = 베트남어)
            max_length: 최대 생성 길이 (상품명은 짧으므로 32로 줄임)
        
        Returns:
            번역된 베트남어 텍스트
        """
        if pd.isna(korean_text) or not str(korean_text).strip():
            return ""
        
        korean_text = str(korean_text).strip()
        
        try:
            # TranslateGemma chat template 형식 사용
            # User role의 content는 리스트 형식이어야 함
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": korean_text
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
            
            # 디바이스로 이동 및 dtype 변환
            dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
            inputs = {k: v.to(self.device).to(dtype) if v.dtype.is_floating_point else v.to(self.device) 
                     for k, v in inputs.items()}
            
            input_length = len(inputs['input_ids'][0])
            
            # 번역 생성 (속도 최적화: greedy decoding)
            with torch.no_grad():
                generation = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_length,
                    num_beams=1,  # greedy decoding (가장 빠름)
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # 반복 방지
                )
            
            # 디코딩 (입력 부분 제거)
            generation = generation[0][input_length:]
            translated = self.processor.decode(generation, skip_special_tokens=True)
            
            return translated.strip()
        except Exception as e:
            print(f"번역 오류 (텍스트: {korean_text[:50]}...): {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def translate_batch(self, texts, batch_size=1, source_lang='ko', target_lang='vi'):
        """
        여러 텍스트를 배치로 번역 (현재는 순차 처리, 향후 실제 배치 처리 구현 가능)
        
        Args:
            texts: 번역할 텍스트 리스트
            batch_size: 배치 크기 (현재는 1로 설정 - 순차 처리)
            source_lang: 원본 언어
            target_lang: 목표 언어
        
        Returns:
            번역된 텍스트 리스트
        """
        results = []
        
        # 순차 처리 (각 텍스트당 약 10-20초 예상)
        for i in tqdm(range(len(texts)), desc="번역 진행 중"):
            text = texts[i]
            translated = self.translate(text, source_lang, target_lang, max_length=32)
            results.append(translated)
            
            # GPU 메모리 정리 (매 5개마다)
            if self.device.type == 'cuda' and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        return results


def test_translate_csv(input_csv='products_all_categorized.csv', 
                       output_csv='products_all_categorized_translated_vi.csv',
                       sample_size=50,
                       model_size='4b',
                       columns_to_translate=['name', 'manufacturer', 'category_level1', 'category_level2', 'category_level3']):
    """
    CSV 파일에서 샘플 데이터를 번역하여 테스트
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
        sample_size: 테스트할 샘플 수
        model_size: 모델 크기 ('4b', '12b', '27b')
        columns_to_translate: 번역할 컬럼명 리스트
    """
    print("=" * 60)
    print("TranslateGemma 테스트: 한국어 → 베트남어")
    print("=" * 60)
    print()
    
    # CSV 로드
    print(f"CSV 파일 로딩: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"[오류] 파일을 찾을 수 없습니다: {input_csv}")
        return
    
    df = pd.read_csv(input_csv, encoding='utf-8')
    print(f"전체 데이터 개수: {len(df):,}개")
    print(f"컬럼: {list(df.columns)}")
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
    translator = TranslateGemmaTester(model_size=model_size)
    
    # 각 컬럼 번역
    for col in columns_to_translate:
        if col not in df_sample.columns:
            print(f"경고: 컬럼 '{col}'이 존재하지 않습니다. 건너뜁니다.")
            continue
        
        print(f"컬럼 '{col}' 번역 중...")
        translated_col = f"{col}_vi"  # 번역된 컬럼명
        
        # 번역 실행
        texts = df_sample[col].fillna("").astype(str).tolist()
        df_sample[translated_col] = translator.translate_batch(
            texts,
            batch_size=1,  # 순차 처리 (실제 배치는 TranslateGemma 구조상 복잡)
            source_lang='ko',
            target_lang='vi'
        )
        
        # 번역 결과 샘플 출력
        print(f"[OK] '{col}' → '{translated_col}' 번역 완료")
        print("번역 샘플 (처음 3개):")
        for idx in range(min(3, len(df_sample))):
            original = df_sample[col].iloc[idx]
            translated = df_sample[translated_col].iloc[idx]
            if original and translated:
                print(f"  [{idx+1}]")
                print(f"    원본 (한국어): {original[:80]}...")
                print(f"    번역 (베트남어): {translated[:80]}...")
        print()
    
    # 결과 저장
    print(f"결과 저장: {output_csv}")
    df_sample.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("[OK] 저장 완료")
    print()
    
    # 번역 결과 요약
    print("=" * 60)
    print("번역 결과 요약")
    print("=" * 60)
    print(f"총 처리: {len(df_sample):,}개")
    for col in columns_to_translate:
        if col in df_sample.columns:
            translated_col = f"{col}_vi"
            if translated_col in df_sample.columns:
                non_empty = df_sample[translated_col].notna() & (df_sample[translated_col] != "")
                print(f"  {col}: {non_empty.sum()}/{len(df_sample)}개 번역 완료")
    print()


def test_single_translation(text, model_size='4b'):
    """
    단일 텍스트 번역 테스트
    
    Args:
        text: 번역할 한국어 텍스트
        model_size: 모델 크기
    """
    print("=" * 60)
    print("단일 텍스트 번역 테스트")
    print("=" * 60)
    print()
    
    translator = TranslateGemmaTester(model_size=model_size)
    
    print(f"원본 (한국어): {text}")
    print()
    
    translated = translator.translate(text, source_lang='ko', target_lang='vi')
    
    print(f"번역 (베트남어): {translated}")
    print()


if __name__ == "__main__":
    # 테스트 옵션
    TEST_SINGLE = False  # True로 설정하면 단일 텍스트 테스트 (디버깅용)
    TEST_CSV = True      # True로 설정하면 CSV 파일 테스트
    
    if TEST_SINGLE:
        # 단일 텍스트 테스트
        test_single_translation(
            text="그린키즈 요술지팡이 이솝우화 (세트(20권))",
            model_size='4b'
        )
    
    if TEST_CSV:
        # CSV 파일 테스트 (샘플 50개만)
        test_translate_csv(
            input_csv='products_all_categorized.csv',
            output_csv='products_all_categorized_translated_vi.csv',
            sample_size=10,  # 테스트용으로 10개만 번역 (속도 테스트)
            model_size='4b',  # 4B 모델 사용 (가장 빠름)
            columns_to_translate=['name', 'manufacturer', 'category_level1', 'category_level2', 'category_level3']
        )
    
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print()
    print("참고:")
    print("- 모델명: google/translategemma-4b-it (또는 12b-it, 27b-it)")
    print("- Gemma 라이선스 동의 필요: https://huggingface.co/google/translategemma-4b-it")
    print("- 설치: pip install transformers torch accelerate")
    print("- Hugging Face 로그인: huggingface-cli login")
