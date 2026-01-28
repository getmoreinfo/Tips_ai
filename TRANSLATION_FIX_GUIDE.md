# TranslateGemma 번역 문제 해결 가이드

## 문제 요약

`products_all_categorized_translated_vi.csv`에서 발견된 번역 문제:

1. **브랜드명/고유명사 번역 문제** (10건)
   - 반달 → Hình mặt trăng (달의 모양) ❌
   - 헬로미니미 → Chào bạn nhỏ (안녕 작은 친구) ❌
   - 아가짱 → Bé yêu (사랑하는 아기) ❌
   - 새들베이비 → Sẻ bé (작은 새) ❌
   - 그린키즈 → Green Kids (의도적일 수 있으나 보존 권장)
   - 어스본 → Earthson (정확한 브랜드명 아님)
   - 지학사 → Nha xuất bản Địa Học (출판사명이 번역됨) ❌
   - 폴레드 → Poléd (음성학적 변형)

2. **키워드 오번역** (2건)
   - 고구마 → khoai tây (감자) ❌ → khoai lang (고구마) ✅
   - 기저귀 → 누락 ❌ → tã/bỉm ✅

3. **정보 누락** (1건)
   - "1년" → 누락 ❌ → "1 năm" ✅

---

## 해결 방안

### 방법 1: 기존 번역 결과 수정 (즉시 적용 가능)

기존에 번역된 `products_all_categorized_translated_vi.csv` 파일의 문제를 자동으로 수정합니다.

**실행 방법:**
```bash
python fix_existing_translations.py
```

**결과:**
- `products_all_categorized_translated_vi_fixed.csv` 파일 생성
- 브랜드명 복원, 키워드 오번역 수정, 숫자 누락 복원

**장점:**
- ✅ 즉시 적용 가능
- ✅ 기존 번역 결과 재활용
- ✅ 추가 번역 시간 불필요

**단점:**
- ⚠️ 완벽한 복원이 어려울 수 있음 (이미 번역된 결과를 수정하는 방식)

---

### 방법 2: 개선된 번역 시스템 사용 (권장)

번역 전에 고유명사를 보호하고, 번역 후 추가 수정을 적용하는 통합 시스템입니다.

**실행 방법:**
```bash
python test_translategemma_ko_to_vi_improved.py
```

**작동 원리:**

1. **전처리 단계** (`translation_preprocessor.py`)
   - 브랜드명을 플레이스홀더로 치환 (예: `그린키즈` → `BRAND_GREENKIDS`)
   - 특수 키워드 보호 (예: `고구마` → `KEYWORD_SWEETPOTATO`)

2. **번역 단계**
   - TranslateGemma로 번역 (플레이스홀더는 그대로 유지)

3. **후처리 단계**
   - 플레이스홀더를 표준명으로 복원 (예: `BRAND_GREENKIDS` → `Green Kids`)
   - 추가 문제 수정 (키워드 오번역, 숫자 누락 등)

**장점:**
- ✅ 가장 정확한 번역 품질
- ✅ 브랜드명 보존 보장
- ✅ 키워드 오번역 방지

**단점:**
- ⚠️ 전체 데이터를 다시 번역해야 함 (시간 소요)

---

### 방법 3: 수동 브랜드명 사전 확장

더 많은 브랜드명을 추가하여 정확도를 높일 수 있습니다.

**브랜드명 사전 파일 생성:**
```json
{
  "그린키즈": "Green Kids",
  "반달": "Bandal",
  "어스본": "Usborne",
  "지학사": "Jihak Publishing",
  "아가짱": "Agachan",
  "폴레드": "Poled",
  "에르고베이비": "Ergobaby",
  "새들베이비": "SaddleBaby",
  "헬로미니미": "Hello Mini"
}
```

**사용 방법:**
```python
# translation_preprocessor.py 또는 translation_postprocessor.py에서
processor = TranslationPreProcessor(brand_dict_path='brand_dict.json')
```

---

## 파일 구조

```
├── translation_preprocessor.py      # 번역 전 전처리 (고유명사 보호)
├── translation_postprocessor.py     # 번역 후 후처리 (문제 수정)
├── test_translategemma_ko_to_vi_improved.py  # 개선된 번역 시스템
├── fix_existing_translations.py     # 기존 번역 결과 수정
└── TRANSLATION_FIX_GUIDE.md         # 이 가이드
```

---

## 권장 워크플로우

### 시나리오 1: 기존 번역 결과가 있고 빠르게 수정하고 싶은 경우

```bash
# 1. 기존 번역 결과 수정
python fix_existing_translations.py

# 2. 수정된 결과 확인
# products_all_categorized_translated_vi_fixed.csv 파일 확인
```

### 시나리오 2: 새로 번역하거나 더 정확한 번역이 필요한 경우

```bash
# 1. 개선된 번역 시스템 사용
python test_translategemma_ko_to_vi_improved.py

# 2. 번역 결과 확인
# products_all_categorized_translated_vi_improved.csv 파일 확인
```

### 시나리오 3: 대량 데이터 처리

```python
# test_translategemma_ko_to_vi_improved.py 수정
test_translate_csv_improved(
    input_csv='products_all_categorized.csv',
    output_csv='products_all_categorized_translated_vi_improved.csv',
    sample_size=None,  # 전체 데이터 처리
    model_size='4b',
    columns_to_translate=['name', 'manufacturer', 'category_level1', 'category_level2', 'category_level3'],
    use_preprocessing=True
)
```

---

## 수정 사항 상세

### 브랜드명 복원 규칙

| 원본 | 잘못된 번역 | 올바른 번역 | 수정 방법 |
|------|------------|------------|----------|
| 반달 | Hình mặt trăng | Bandal | 후처리에서 패턴 매칭 후 교체 |
| 헬로미니미 | Chào bạn nhỏ | Hello Mini | 후처리에서 패턴 매칭 후 교체 |
| 아가짱 | Bé yêu | Agachan | 후처리에서 패턴 매칭 후 교체 |
| 새들베이비 | Sẻ bé | SaddleBaby | 후처리에서 패턴 매칭 후 교체 |
| 그린키즈 | Green Kids | Green Kids | ✅ 이미 정확 (선택적 보존) |
| 어스본 | Earthson | Usborne | 후처리에서 표준명으로 교체 |
| 지학사 | Nha xuất bản Địa Học | Jihak Publishing | 후처리에서 패턴 매칭 후 교체 |
| 폴레드 | Poléd | Poled | 후처리에서 표준명으로 교체 |

### 키워드 오번역 수정

| 원본 | 잘못된 번역 | 올바른 번역 | 수정 방법 |
|------|------------|------------|----------|
| 고구마 | khoai tây (감자) | khoai lang (고구마) | 후처리에서 문자열 교체 |
| 기저귀 | 누락 | tã/bỉm | 후처리에서 원본 확인 후 추가 |

### 숫자 누락 복원

| 원본 | 잘못된 번역 | 올바른 번역 | 수정 방법 |
|------|------------|------------|----------|
| 1년 정기구독 | hàng năm (매년) | 1 năm (1년) | 후처리에서 정규표현식으로 숫자 추출 후 복원 |

---

## 테스트 및 검증

### 수정 전후 비교

```python
import pandas as pd

# 원본 번역 결과
df_original = pd.read_csv('products_all_categorized_translated_vi.csv')

# 수정된 번역 결과
df_fixed = pd.read_csv('products_all_categorized_translated_vi_fixed.csv')

# 비교
for idx in range(min(10, len(df_original))):
    print(f"\n[{idx+1}]")
    print(f"원본: {df_original['name'].iloc[idx]}")
    print(f"번역 (수정 전): {df_original['name_vi'].iloc[idx]}")
    print(f"번역 (수정 후): {df_fixed['name_vi'].iloc[idx]}")
```

---

## 추가 개선 사항

### 1. 프롬프트 개선

TranslateGemma에 더 명확한 지시를 추가할 수 있습니다:

```python
# test_translategemma_ko_to_vi_improved.py의 translate 함수 수정
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": source_lang,
                "target_lang_code": target_lang,
                "text": f"다음 텍스트를 번역하세요. 브랜드명과 고유명사는 번역하지 말고 그대로 보존하세요:\n{text_to_translate}"
            }
        ]
    }
]
```

### 2. 브랜드명 자동 감지

데이터에서 자동으로 브랜드명을 추출하여 사전에 추가:

```python
# manufacturer 컬럼에서 고유값 추출
brands = df['manufacturer'].unique()
# 자동으로 브랜드명 사전에 추가
```

### 3. 번역 품질 자동 검증

번역 후 자동으로 문제를 감지하고 보고:

```python
# translation_quality_checker.py (향후 구현)
# - 브랜드명 번역 여부 확인
# - 키워드 오번역 감지
# - 숫자 누락 확인
```

---

## 결론

**즉시 적용 가능한 해결책:**
1. `fix_existing_translations.py` 실행하여 기존 번역 결과 수정

**장기적 해결책:**
1. `test_translategemma_ko_to_vi_improved.py` 사용하여 새로 번역
2. 브랜드명 사전 확장
3. 번역 품질 자동 검증 시스템 구축

**권장 사항:**
- 작은 샘플로 먼저 테스트
- 수정된 결과를 검증 후 전체 데이터에 적용
- 브랜드명 사전을 지속적으로 확장
