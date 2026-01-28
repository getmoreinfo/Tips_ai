# translate_all_products.py
# 역할: products_all.csv 파일의 전체 데이터를 베트남어로 번역

import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime
from translation_preprocessor import TranslationPreProcessor
from translation_postprocessor import TranslationPostProcessor
import re

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
        """
        print("=" * 60)
        print("개선된 TranslateGemma 모델 로딩 중...")
        print("=" * 60)
        
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
            
            print("프로세서 로딩 중...")
            if hf_token:
                self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
            else:
                self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Decoder-only 모델을 위한 left padding 설정
            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer:
                self.processor.tokenizer.padding_side = 'left'
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
            print("모델 로딩 중... (시간이 걸릴 수 있습니다)")
            print("  - 모델이 로컬에 없으면 Hugging Face에서 다운로드됩니다")
            print("  - 다운로드 중에는 네트워크 상태에 따라 시간이 걸릴 수 있습니다")
            print("  - 로딩 중...")
            
            try:
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
            except Exception as load_error:
                print(f"[오류] 모델 로딩 중 예외 발생: {load_error}")
                print("  - 네트워크 연결을 확인하세요")
                print("  - GPU 메모리가 충분한지 확인하세요")
                raise
            
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
        """개선된 번역 함수 (단일 텍스트)"""
        results = self.translate_batch(
            [korean_text],
            brands=[original_brand] if original_brand else None,
            source_lang=source_lang,
            target_lang=target_lang,
            max_length=max_length,
            use_preprocessing=use_preprocessing
        )
        return results[0] if results else ""

    def _is_bad_translation(self, original_text: str, translated_text: str) -> bool:
        """
        간단 품질 게이트:
        - 모델이 종종 내뱉는 라벨/키워드 형태(THƯƠNG HIỆU, CHỦ KHÓA 등) 탐지
        - 과도한 대문자/구분자 나열 탐지
        - 공백/빈 문자열
        """
        if translated_text is None:
            return True
        t = str(translated_text).strip()
        if not t:
            return True

        # 금지 패턴(실제로 관측된 문제 케이스)
        banned_markers = [
            "THƯƠNG HIỆU",  # 브랜드 라벨
            "CHỦ KHÓA",     # 키워드 라벨
            "TỪ KHÓA",
            "CHỨC NĂNG",
            "GIẤU HÀNG",    # 의도치 않은 문구
        ]
        upper_t = t.upper()
        for m in banned_markers:
            if m in t or m in upper_t:
                return True

        # "라벨: 값, 라벨: 값" 같은 키워드 나열형
        if t.count(":") >= 2 and t.count(",") >= 2:
            return True

        # 대문자 비율이 지나치게 높으면(키워드 나열 가능성)
        letters = [ch for ch in t if ch.isalpha()]
        if letters:
            upper_letters = [ch for ch in letters if ch.isupper()]
            if (len(upper_letters) / max(1, len(letters))) > 0.7 and len(letters) >= 10:
                return True

        # 원문이 짧은데 번역이 과도하게 길면(설명문/나열문 가능성)
        if original_text is not None:
            o = str(original_text).strip()
            if o and len(t) > (len(o) * 3) and len(t) > 80:
                return True

        return False

    def _build_text_payload(self, text_to_translate: str, strict: bool) -> str:
        """
        Translategemma 입력 텍스트에 최소한의 가이드 라인을 주기 위한 래퍼.
        (실패 건 재시도에서만 strict=True 사용)
        """
        if not strict:
            return text_to_translate

        # 너무 길게 지시하면 품질이 흔들릴 수 있어, 짧게 고정 문구만 얹습니다.
        return (
            "Translate the product name to Vietnamese.\n"
            "Output ONLY the Vietnamese product name.\n"
            "Do NOT add labels like 'THƯƠNG HIỆU'/'CHỦ KHÓA' or keyword lists.\n"
            "Keep brand/model codes/numbers as-is.\n\n"
            f"{text_to_translate}"
        )
    
    def translate_batch(self, texts, brands=None, source_lang='ko', target_lang='vi',
                       max_length=32, use_preprocessing=True, actual_batch_size=8, max_retries=2):
        """
        배치 번역 함수 (실제 배치 처리로 속도 개선)
        
        Args:
            texts: 번역할 텍스트 리스트
            brands: 브랜드명 리스트 (선택사항)
            source_lang: 원본 언어
            target_lang: 목표 언어
            max_length: 최대 생성 길이
            use_preprocessing: 전처리 사용 여부
            actual_batch_size: 실제 모델 배치 크기 (GPU 메모리에 따라 조정)
        
        Returns:
            번역된 텍스트 리스트
        """
        if not texts:
            return []
        
        if brands is None:
            brands = [None] * len(texts)
        
        results = []
        
        # 배치 단위로 처리
        for batch_start in range(0, len(texts), actual_batch_size):
            batch_end = min(batch_start + actual_batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_brands = brands[batch_start:batch_end] if brands else [None] * len(batch_texts)
            
            # 전처리
            batch_messages = []
            batch_replacements = []
            batch_originals = []
            
            for i, text in enumerate(batch_texts):
                if pd.isna(text) or not str(text).strip():
                    batch_messages.append(None)
                    batch_replacements.append({})
                    batch_originals.append("")
                    continue
                
                original_text = str(text).strip()
                batch_originals.append(original_text)
                text_to_translate = original_text
                replacements = {}
                
                if use_preprocessing:
                    text_to_translate, replacements = self.preprocessor.preprocess(original_text)
                
                text_to_translate = self._build_text_payload(text_to_translate, strict=False)

                messages = [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text_to_translate
                    }]
                }]
                
                batch_messages.append(messages)
                batch_replacements.append(replacements)
            
            # None이 아닌 메시지만 필터링하여 배치 처리
            valid_indices = [i for i, msg in enumerate(batch_messages) if msg is not None]
            
            if not valid_indices:
                # 모두 빈 텍스트
                results.extend([""] * len(batch_texts))
                continue
            
            valid_messages = [batch_messages[i] for i in valid_indices]
            
            try:
                # 배치 처리: 개별적으로 토크나이징 후 패딩하여 배치 생성
                if len(valid_messages) == 0:
                    results.extend([""] * len(batch_texts))
                    continue
                
                tokenized_inputs = []
                input_lengths = []
                
                for msg in valid_messages:
                    # 개별 메시지 토크나이징
                    single_input = self.processor.apply_chat_template(
                        msg,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    input_lengths.append(len(single_input['input_ids'][0]))
                    tokenized_inputs.append(single_input)
                
                # 패딩하여 배치로 만들기 (LEFT PADDING for decoder-only models)
                max_length = max(input_lengths)
                batch_input_ids = []
                batch_attention_mask = []
                
                pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                
                for single_input in tokenized_inputs:
                    input_ids = single_input['input_ids'][0]
                    pad_length = max_length - len(input_ids)
                    
                    # LEFT PADDING (decoder-only 모델용)
                    if pad_length > 0:
                        # 왼쪽에 패딩 추가
                        padding = torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                        input_ids = torch.cat([padding, input_ids])
                        attention_mask = torch.cat([torch.zeros(pad_length, dtype=torch.long),
                                                   torch.ones(len(single_input['input_ids'][0]), dtype=torch.long)])
                    else:
                        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
                    
                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                
                # 배치 텐서 생성
                inputs = {
                    'input_ids': torch.stack(batch_input_ids),
                    'attention_mask': torch.stack(batch_attention_mask)
                }
                
                dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
                inputs = {k: v.to(self.device).to(dtype) if v.dtype.is_floating_point else v.to(self.device) 
                         for k, v in inputs.items()}
                
                # 배치 생성
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
                
                # 배치 결과 디코딩 (left padding 고려)
                batch_results = []
                max_input_len = max(input_lengths)
                
                for i, gen_seq in enumerate(generation):
                    # Left padding이므로, 전체 길이에서 실제 입력 길이를 빼야 함
                    # generation은 [PAD...PAD, INPUT_TOKENS..., NEW_TOKENS...] 형태
                    actual_input_len = input_lengths[i]
                    pad_length = max_input_len - actual_input_len
                    
                    # 패딩 부분과 원본 입력 부분을 제거하고 새로 생성된 부분만 추출
                    # gen_seq[pad_length + actual_input_len:] 하면 새로 생성된 토큰만 남음
                    new_tokens = gen_seq[pad_length + actual_input_len:]
                    translated = self.processor.decode(new_tokens, skip_special_tokens=True)
                    translated = translated.strip()
                    
                    # 후처리
                    orig_idx = valid_indices[i]
                    original_text = batch_originals[orig_idx]
                    replacements = batch_replacements[orig_idx]
                    brand = batch_brands[orig_idx]
                    
                    if use_preprocessing and replacements:
                        translated = self.preprocessor.restore(translated, replacements, use_standard_names=True)
                    
                    translated = self.postprocessor.process_translation(
                        original_text, translated, str(brand) if pd.notna(brand) else None
                    )

                    # 품질 게이트 → 실패 건만 재시도 → 그래도 실패면 원문 유지
                    if self._is_bad_translation(original_text, translated):
                        retry_ok = False
                        last = translated
                        for _ in range(max_retries):
                            try:
                                # strict 프롬프트로 재시도 (개별 처리)
                                retr = self.translate_single(
                                    original_text,
                                    source_lang=source_lang,
                                    target_lang=target_lang,
                                    max_length=max_length,
                                    original_brand=brand,
                                    use_preprocessing=use_preprocessing,
                                    strict=True,
                                )
                                if not self._is_bad_translation(original_text, retr):
                                    translated = retr
                                    retry_ok = True
                                    break
                                last = retr
                            except Exception:
                                pass
                        if not retry_ok:
                            # 최종 폴백: 원문 유지(가장 안전)
                            translated = str(original_text).strip() if original_text is not None else ""
                    
                    batch_results.append(translated)
                
                # 결과를 전체 인덱스에 매핑
                batch_translated = [""] * len(batch_texts)
                for i, result in zip(valid_indices, batch_results):
                    batch_translated[i] = result
                
                results.extend(batch_translated)
                
                # 배치 처리 성공 확인 (디버그용)
                if len(valid_messages) > 1:
                    # 배치 처리 성공 - 조용히 진행 (주석 처리 가능)
                    pass
                
            except Exception as e:
                # 배치 처리 실패 시 개별 처리로 폴백
                print(f"\n[경고] 배치 처리 실패, 개별 처리로 전환: {e}")
                for i, text in enumerate(batch_texts):
                    try:
                        brand = batch_brands[i] if i < len(batch_brands) else None
                        translated = self.translate_single(
                            text, source_lang, target_lang, max_length,
                            brand, use_preprocessing
                        )
                        results.append(translated)
                    except Exception as e2:
                        print(f"  개별 번역 오류: {e2}")
                        results.append("")
        
        return results
    
    def translate_single(self, korean_text, source_lang='ko', target_lang='vi',
                        max_length=32, original_brand=None, use_preprocessing=True, strict=False):
        """단일 텍스트 번역 (내부 함수)"""
        if pd.isna(korean_text) or not str(korean_text).strip():
            return ""
        
        original_text = str(korean_text).strip()
        text_to_translate = original_text
        replacements = {}
        
        if use_preprocessing:
            text_to_translate, replacements = self.preprocessor.preprocess(original_text)
        text_to_translate = self._build_text_payload(text_to_translate, strict=strict)
        
        try:
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text_to_translate
                }]
            }]
            
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
            inputs = {k: v.to(self.device).to(dtype) if v.dtype.is_floating_point else v.to(self.device) 
                     for k, v in inputs.items()}
            
            input_length = len(inputs['input_ids'][0])
            
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
            
            generation = generation[0][input_length:]
            translated = self.processor.decode(generation, skip_special_tokens=True)
            translated = translated.strip()
            
            if use_preprocessing and replacements:
                translated = self.preprocessor.restore(translated, replacements, use_standard_names=True)
            
            translated = self.postprocessor.process_translation(
                original_text, translated, str(original_brand) if pd.notna(original_brand) else None
            )
            
            return translated
        except Exception as e:
            print(f"번역 오류 (텍스트: {original_text[:50]}...): {e}")
            return ""


def translate_all_products(
    input_csv='products_all.csv',
    output_csv='products_all_translated_vi.csv',
    progress_file='translation_progress.json',
    model_size='4b',
    columns_to_translate=['name', 'manufacturer'],
    batch_size=100,  # 저장 주기
    actual_batch_size=8,  # 실제 모델 배치 크기 (GPU 메모리에 따라 조정)
    resume=True,
    limit=None  # 테스트용: 처리할 최대 행 수 (None이면 전체 처리)
):
    """
    products_all.csv 전체 데이터 번역
    
    Args:
        input_csv: 입력 CSV 파일
        output_csv: 출력 CSV 파일
        progress_file: 진행 상황 저장 파일 (중단 시 재개 가능)
        model_size: 모델 크기
        columns_to_translate: 번역할 컬럼 리스트
        batch_size: 배치 크기 (저장 주기)
        resume: 이전 진행 상황에서 재개할지 여부
    """
    print("=" * 60)
    print("전체 상품 데이터 베트남어 번역")
    print("=" * 60)
    print()
    
    # CSV 로드
    print(f"CSV 파일 로딩: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"[오류] 파일을 찾을 수 없습니다: {input_csv}")
        return
    
    df = pd.read_csv(input_csv, encoding='utf-8')
    total_rows = len(df)
    
    # 테스트용 제한 적용
    if limit is not None and limit > 0:
        df = df.head(limit)
        total_rows = len(df)
        print(f"[테스트 모드] 데이터 제한: {limit:,}개 행만 처리합니다.")
    
    print(f"전체 데이터: {total_rows:,}개")
    print(f"번역할 컬럼: {columns_to_translate}")
    print()
    
    # 진행 상황 로드
    start_idx = 0
    resume_column = None
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                start_idx = progress.get('last_processed_idx', 0)
                resume_column = progress.get('current_column')
                print(f"[재개] 이전 진행 상황 발견: {start_idx:,}번째 행부터 재개")
                print()
        except:
            print("[정보] 진행 상황 파일을 읽을 수 없습니다. 처음부터 시작합니다.")
            print()
    
    # 출력 파일 로드 (이미 번역된 부분이 있으면)
    if resume and os.path.exists(output_csv) and start_idx > 0:
        print(f"기존 번역 결과 로딩: {output_csv}")
        df_output = pd.read_csv(output_csv, encoding='utf-8')
        print(f"  기존 번역: {len(df_output):,}개 행")
        # df_output을 df에 병합 (번역된 컬럼만)
        for col in columns_to_translate:
            translated_col = f"{col}_vi"
            if translated_col in df_output.columns:
                df[translated_col] = df_output[translated_col]
        print()
    else:
        # 출력 파일 초기화
        for col in columns_to_translate:
            translated_col = f"{col}_vi"
            df[translated_col] = None
    
    # 번역기 초기화
    print("번역기 초기화 중...")
    translator = ImprovedTranslateGemmaTester(model_size=model_size)
    print()
    
    # 번역 실행
    print("=" * 60)
    print("번역 시작")
    print("=" * 60)
    print()
    
    total_to_translate = total_rows - start_idx
    total_items_to_translate = total_to_translate * len(columns_to_translate)
    
    # 실제 속도 측정을 위한 변수
    import time
    start_time = time.time()
    items_processed_for_timing = 0
    timing_start_idx = max(0, start_idx - 10)  # 처음 10개로 속도 측정
    
    processed_count = 0
    error_count = 0
    
    print(f"총 번역할 항목: {total_items_to_translate:,}개 (행 {total_to_translate:,}개 × 컬럼 {len(columns_to_translate)}개)")
    print(f"초기 속도 측정 중... (처음 몇 개 항목으로 실제 속도 계산)")
    print()
    
    try:
        # 배치 처리 최적화: 컬럼별로 배치 단위 번역
        for col in columns_to_translate:
            if col not in df.columns:
                print(f"경고: 컬럼 '{col}'이 존재하지 않습니다. 건너뜁니다.")
                continue
            
            translated_col = f"{col}_vi"
            print(f"\n{'='*60}")
            print(f"컬럼 '{col}' 번역 시작")
            print(f"{'='*60}")
            
            # 번역할 항목 수집 (배치 처리용)
            items_to_translate = []
            item_indices = []
            item_brands = []

            # 컬럼 단위 재개 처리:
            # - progress_file의 start_idx는 "중단 당시 진행 중이던 컬럼"에만 적용
            # - 다음 컬럼부터는 0부터 훑으면서 비어있는 번역만 채워야 누락이 없음
            col_start_idx = start_idx if (resume and resume_column == col) else 0
            
            for idx in range(col_start_idx, total_rows):
                # 이미 번역되어 있으면 건너뛰기
                if pd.notna(df.loc[idx, translated_col]):
                    continue
                
                row = df.iloc[idx]
                original_text = row[col]
                
                if pd.isna(original_text) or not str(original_text).strip():
                    df.loc[idx, translated_col] = ""
                    continue
                
                items_to_translate.append(str(original_text).strip())
                item_indices.append(idx)
                # 브랜드명 가져오기 (name 컬럼 번역 시)
                brand = row['manufacturer'] if col == 'name' and 'manufacturer' in row else None
                item_brands.append(str(brand) if pd.notna(brand) else None)
            
            if not items_to_translate:
                print(f"  모든 항목이 이미 번역되어 있습니다.")
                continue
            
            print(f"  번역할 항목: {len(items_to_translate):,}개")
            print(f"  배치 크기: {actual_batch_size}개")
            print()
            
            # 배치 단위로 번역
            pbar = tqdm(range(0, len(items_to_translate), actual_batch_size),
                       desc=f"'{col}' 번역 중", total=len(items_to_translate), unit='item')
            
            for batch_start in pbar:
                batch_end = min(batch_start + actual_batch_size, len(items_to_translate))
                batch_texts = items_to_translate[batch_start:batch_end]
                batch_brands = item_brands[batch_start:batch_end]
                batch_indices = item_indices[batch_start:batch_end]
                
                try:
                    # 배치 번역 실행
                    translated_results = translator.translate_batch(
                        batch_texts,
                        brands=batch_brands,
                        source_lang='ko',
                        target_lang='vi',
                        max_length=32,
                        use_preprocessing=True,
                        actual_batch_size=len(batch_texts),  # 실제 배치 크기 전달
                        max_retries=2
                    )
                    
                    # 결과 저장
                    for idx, translated in zip(batch_indices, translated_results):
                        df.loc[idx, translated_col] = translated
                        processed_count += 1
                        items_processed_for_timing += 1
                    
                    # tqdm 업데이트 (실제 처리된 항목 수만큼)
                    pbar.update(len(batch_texts))
                    
                    # 처음 10개 항목으로 실제 속도 측정 후 예상 시간 업데이트
                    if items_processed_for_timing == 10:
                        elapsed_time = time.time() - start_time
                        avg_time_per_item = elapsed_time / 10
                        remaining_items = total_items_to_translate - processed_count
                        estimated_remaining_time = (remaining_items * avg_time_per_item) / 60
                        print(f"\n[속도 측정 완료] 평균 {avg_time_per_item:.1f}초/항목")
                        print(f"예상 남은 시간: 약 {estimated_remaining_time:.1f}분 ({estimated_remaining_time/60:.1f}시간)")
                        print()
                    
                except Exception as e:
                    print(f"\n[오류] 배치 번역 실패 (인덱스 {batch_start}-{batch_end}): {e}")
                    # 개별 처리로 폴백
                    for idx, text, brand in zip(batch_indices, batch_texts, batch_brands):
                        try:
                            translated = translator.translate_single(
                                text, 'ko', 'vi', 32, brand, True
                            )
                            df.loc[idx, translated_col] = translated
                            processed_count += 1
                        except Exception as e2:
                            print(f"  개별 번역 오류 (행 {idx}): {e2}")
                            df.loc[idx, translated_col] = ""
                            error_count += 1
                
                # 배치마다 저장 (진행 상황 포함)
                if (batch_end) % (batch_size * len(columns_to_translate)) == 0 or batch_end == len(items_to_translate):
                    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                    
                    # 진행 상황 저장
                    progress = {
                        'last_processed_idx': max(batch_indices) + 1 if batch_indices else start_idx,
                        'total_rows': total_rows,
                        'processed_count': processed_count,
                        'error_count': error_count,
                        'current_column': col,
                        'last_updated': datetime.now().isoformat()
                    }
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress, f, indent=2, ensure_ascii=False)
                    
                    # 진행률 및 예상 남은 시간 계산
                    if items_processed_for_timing >= 10:
                        elapsed_time = time.time() - start_time
                        avg_time_per_item = elapsed_time / processed_count
                        remaining_items = total_items_to_translate - processed_count
                        estimated_remaining_time = (remaining_items * avg_time_per_item) / 60
                        progress_pct = (processed_count / total_items_to_translate) * 100
                        print(f"\n[저장] {processed_count:,}/{total_items_to_translate:,} 항목 완료 ({progress_pct:.1f}%, 오류: {error_count}개)")
                        print(f"  예상 남은 시간: 약 {estimated_remaining_time:.1f}분 ({estimated_remaining_time/60:.1f}시간)")
                    else:
                        print(f"\n[저장] 진행 상황 저장됨")
                
                # GPU 메모리 정리
                if translator.device.type == 'cuda' and batch_end % (actual_batch_size * 10) == 0:
                    torch.cuda.empty_cache()
            
            # 배치마다 저장 (진행 상황 포함)
            if (idx + 1) % batch_size == 0 or (idx + 1) == total_rows:
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                
                # 진행 상황 저장
                progress = {
                    'last_processed_idx': idx + 1,
                    'total_rows': total_rows,
                    'processed_count': processed_count,
                    'error_count': error_count,
                    'last_updated': datetime.now().isoformat()
                }
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress, f, indent=2, ensure_ascii=False)
                
                # 진행률 및 예상 남은 시간 계산
                if items_processed_for_timing >= 10:
                    elapsed_time = time.time() - start_time
                    avg_time_per_item = elapsed_time / processed_count
                    remaining_items = total_items_to_translate - processed_count
                    estimated_remaining_time = (remaining_items * avg_time_per_item) / 60
                    progress_pct = (processed_count / total_items_to_translate) * 100
                    print(f"\n[저장] {idx + 1:,}/{total_rows:,} 행 완료 ({progress_pct:.1f}%, 오류: {error_count}개)")
                    print(f"  예상 남은 시간: 약 {estimated_remaining_time:.1f}분 ({estimated_remaining_time/60:.1f}시간)")
                else:
                    print(f"\n[저장] {idx + 1:,}/{total_rows:,} 행 완료 (오류: {error_count}개)")
            
            # GPU 메모리 정리
            if translator.device.type == 'cuda' and (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n\n[중단] 사용자에 의해 중단되었습니다.")
        print(f"현재까지 {idx + 1:,}개 행 처리 완료")
        print(f"진행 상황이 저장되었습니다. 재개하려면 같은 명령어를 다시 실행하세요.")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        return
    
    # 최종 저장
    print("\n" + "=" * 60)
    print("번역 완료")
    print("=" * 60)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"결과 저장: {output_csv}")
    print(f"총 처리: {processed_count:,}개 항목")
    print(f"오류: {error_count}개")
    print()
    
    # 진행 상황 파일 삭제 (완료)
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"진행 상황 파일 삭제: {progress_file}")


if __name__ == "__main__":
    # 테스트용: 1000개만 처리하려면 limit=1000 설정
    # 전체 처리하려면 limit=None 또는 limit=0 설정
    translate_all_products(
        input_csv='products_all.csv',
        output_csv='products_all_translated_vi.csv',
        progress_file='translation_progress.json',
        model_size='4b',
        columns_to_translate=['name', 'manufacturer'],
        batch_size=100,  # 저장 주기 (100개마다 저장)
        actual_batch_size=8,  # 실제 모델 배치 크기 (GPU 메모리에 따라 조정: 4, 8, 16 등)
        resume=True,  # 중단 시 재개 가능
        limit=1000  # 테스트용: 1000개만 처리 (전체 처리하려면 None 또는 0)
    )
    
    print("=" * 60)
    print("전체 번역 완료!")
    print("=" * 60)
