import pandas as pd
import sys

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

df = pd.read_csv('products_all_translated_vi.csv', encoding='utf-8-sig')

print("=" * 60)
print("번역 결과 확인")
print("=" * 60)
print(f"\n총 행 수: {len(df):,}개")
print(f"name_vi 비어있음: {df['name_vi'].isna().sum():,}개")
print(f"manufacturer_vi 비어있음: {df['manufacturer_vi'].isna().sum():,}개")

print("\n" + "=" * 60)
print("샘플 확인 (처음 15개)")
print("=" * 60)

for i in range(min(15, len(df))):
    print(f"\n[{i+1}]")
    print(f"  원본 이름: {df.iloc[i]['name']}")
    print(f"  번역 이름: {df.iloc[i]['name_vi']}")
    print(f"  원본 제조사: {df.iloc[i]['manufacturer']}")
    print(f"  번역 제조사: {df.iloc[i]['manufacturer_vi']}")

# 문제가 있는 번역 찾기
print("\n" + "=" * 60)
print("문제가 있는 번역 샘플")
print("=" * 60)

problematic = []
for i in range(len(df)):
    name_vi = str(df.iloc[i]['name_vi']) if pd.notna(df.iloc[i]['name_vi']) else ""
    if "THƯƠNG HIỆU" in name_vi or "CHỦ KHÓA" in name_vi or "GIẤU HÀNG" in name_vi:
        problematic.append(i)
        if len(problematic) <= 5:
            print(f"\n[행 {i+1}]")
            print(f"  원본: {df.iloc[i]['name']}")
            print(f"  번역: {name_vi}")

print(f"\n총 문제 번역: {len(problematic)}개")
