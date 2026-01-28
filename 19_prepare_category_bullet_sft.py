"""
19_prepare_category_bullet_sft.py
역할:
  - products_all.csv(상품+리뷰 요약 컬럼 포함)로부터
    "카테고리 리포트 하단 요약 불릿" 학습용 JSONL(SFT)을 생성한다.

출력(JSONL) 포맷:
  {"messages":[{"role":"system","content":"..."},{"role":"user","content":"{...json...}"},{"role":"assistant","content":"{...json...}"}]}

주의:
  - 현재 CSV에는 '매출'이 없어, 플랫폼/브랜드는 '리뷰 수(또는 분포 카운트)' 기반으로 요약한다.
  - 추후 Postgres의 실제 매출/판매량 테이블이 붙으면, 동일 스키마로 metrics만 교체하면 된다.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from ai_report_bullets_lib import aggregate_category, build_category_summary_bullets


SYSTEM_PROMPT = (
    "너는 상품/리뷰 집계 지표를 근거로 대시보드에 표시할 한국어 요약 불릿을 생성한다. "
    "반드시 유효한 JSON만 출력하고, JSON 외의 문장/마크다운/설명은 절대 출력하지 마라. "
    '출력은 반드시 {"summaryBullets":["문장","문장",...]} 형태의 JSON 객체여야 한다. '
    "중요: 필드명은 반드시 'summaryBullets' (복수형, s로 끝남)이어야 하며, 'summaryBulleted' 같은 다른 이름을 사용하면 안 된다. "
    "summaryBullets는 반드시 문자열 배열(리스트)이어야 하며, 단일 문자열이나 객체를 출력하면 안 된다. "
    "수치는 입력값을 그대로 사용하고, 없는 지표를 추정/과장하지 마라."
)


def _top_tags_list(tags: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    return [{"tag": k, "count": v} for k, v in tags]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="products_all.csv")
    ap.add_argument("--out_jsonl", default="training_category_bullets_sft.jsonl")
    # 기존 default=50이면 학습 샘플이 너무 적게 나올 수 있어 기본값을 낮춤
    ap.add_argument("--min_products", type=int, default=10, help="카테고리당 최소 상품 수")
    ap.add_argument("--max_categories", type=int, default=1000, help="최대 카테고리 수(학습 데이터 크기 제한)")
    ap.add_argument("--max_bullets", type=int, default=8)
    # 학습 데이터 확장: 카테고리 수가 적을 때 같은 카테고리에서 여러 샘플 생성
    ap.add_argument(
        "--samples_per_category",
        type=int,
        default=15,
        help="카테고리당 생성할 학습 샘플 수. 1이면 기존처럼 카테고리당 1개. 15면 카테고리당 15개(서브샘플로 다양화)",
    )
    ap.add_argument(
        "--subsample_ratio",
        type=float,
        default=0.85,
        help="samples_per_category>1일 때 각 샘플에서 사용할 상품 비율(0~1). 서로 다른 seed로 샘플링해 지표 분산",
    )
    ap.add_argument("--seed", type=int, default=42, help="재현을 위한 랜덤 시드")
    args = ap.parse_args()

    print("=" * 60)
    print("카테고리 불릿 SFT 데이터 생성")
    print("=" * 60)
    print(f"입력: {args.input_csv}")
    print(f"출력: {args.out_jsonl}")

    df = pd.read_csv(args.input_csv)

    required_cols = [
        "category_id",
        "category",
        "manufacturer",
        "review_count",
        "average_rating",
        "review_tags",
        "review_rating_distribution",
        "review_platform_distribution",
        "updated_at",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c}")

    # 카테고리별로 그룹화
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")
    df = df[df["category_id"].notna()]

    groups = df.groupby("category_id", sort=False)

    # 충분히 큰 카테고리만
    candidates: List[Tuple[int, pd.DataFrame]] = []
    for cat_id, g in groups:
        if len(g) >= args.min_products:
            candidates.append((int(cat_id), g))

    # 큰 카테고리 우선
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    candidates = candidates[: args.max_categories]

    n_samples_per_cat = max(1, args.samples_per_category)
    subsample_ratio = args.subsample_ratio
    random.seed(args.seed)

    print(f"대상 카테고리 수: {len(candidates):,}개 (min_products={args.min_products})")
    if n_samples_per_cat > 1:
        print(f"카테고리당 샘플 수: {n_samples_per_cat}개 (subsample_ratio={subsample_ratio})")

    written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for cat_id, g in candidates:
            category_path = str(g["category"].dropna().iloc[0]) if g["category"].notna().any() else ""
            n_rows = len(g)

            for sample_idx in range(n_samples_per_cat):
                if n_samples_per_cat > 1 and n_rows > args.min_products:
                    # 서로 다른 서브샘플로 지표 분산 → 다양한 (입력, 출력) 쌍
                    n_take = max(args.min_products, int(n_rows * subsample_ratio))
                    n_take = min(n_take, n_rows)
                    g_use = g.sample(n=n_take, random_state=args.seed + sample_idx)
                else:
                    g_use = g

                if len(g_use) < args.min_products:
                    continue

                metrics = aggregate_category(g_use, cat_id, category_path)
                bullets = build_category_summary_bullets(metrics, max_bullets=args.max_bullets)

                user_input = {
                    "reportType": "category_summary_bullets",
                    "categoryId": metrics.category_id,
                    "categoryName": metrics.category_name,
                    "metrics": {
                        "productCount": metrics.product_count,
                        "brandCount": metrics.brand_count,
                        "totalReviewCount": metrics.total_review_count,
                        "avgRatingWeighted": metrics.avg_rating_weighted,
                        "ratingHistogram": metrics.rating_hist,
                        "platformDistribution": metrics.platform_dist,
                        "topBrands": metrics.top_brands,
                        "topReviewTags": _top_tags_list(metrics.top_tags),
                        "period": {"from": metrics.period_from, "to": metrics.period_to},
                        "zeroReviewRatio": metrics.zero_review_ratio,
                        "lowRatingRatio": metrics.low_rating_ratio,
                        "top3BrandShare": metrics.top3_brand_share,
                        "price": {
                            "min": metrics.price_min,
                            "p25": metrics.price_p25,
                            "p50": metrics.price_p50,
                            "p75": metrics.price_p75,
                            "max": metrics.price_max,
                            "change6mMedian": metrics.price_change_6m_median,
                        },
                    },
                    "outputRequirements": {
                        "language": "ko",
                        "format": "json_only",
                        "maxBullets": args.max_bullets,
                    },
                }

                target = {
                    "categoryId": metrics.category_id,
                    "summaryBullets": bullets,
                }

                rec = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                    ]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"생성 완료: {written:,} examples -> {args.out_jsonl}")
    print("다음 단계: 20_train_category_bullet_lora.py 로 LoRA 학습")


if __name__ == "__main__":
    main()

