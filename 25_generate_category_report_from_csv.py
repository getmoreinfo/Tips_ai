"""
25_generate_category_report_from_csv.py

역할:
  - products_all.csv + reviews_all.csv 2개만으로
    카테고리 리포트 화면에 필요한 '전체 JSON'을 생성한다.

출력(예시):
  {
    "categoryId": 4,
    "categoryName": "...",
    "charts": {
      "brandTop10Donut": {"items":[{"name":"...", "value":123, "share":0.12}, ...]},
      "growthLine": {"unit":"estimatedRevenue", "points":[{"year":2020,"value":...,"growthRate":...}, ...]}
    },
    "summaries": {
      "marketOverviewSummary": "...",
      "growthSummary": "..."
    },
    "assumptions": {
      "proxy": "review_count",
      "unitsPerReview": 3.0,
      "priceRef": 22100.0
    }
  }
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import pandas as pd

from ai_report_bullets_lib import aggregate_category_with_reviews
from report_summary_lib import (
    build_growth_summary,
    build_market_overview_summary,
    calculate_yearly_growth,
)


def _pick_category(
    df_products: pd.DataFrame,
    category_id: Optional[int],
    category_contains: Optional[str],
) -> tuple[int, str, pd.DataFrame]:
    if (category_id is None) == (category_contains is None):
        raise ValueError("--category_id 또는 --category_contains 중 하나만 지정해야 합니다.")

    df = df_products.copy()
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")
    df = df[df["category_id"].notna()]

    if category_id is not None:
        cid = int(category_id)
        g = df[df["category_id"].astype(int) == cid].copy()
        if g.empty:
            raise ValueError(f"category_id={cid}에 해당하는 상품이 없습니다.")
        category_path = str(g["category"].dropna().iloc[0]) if g["category"].notna().any() else ""
        return cid, category_path, g

    # contains 모드
    needle = str(category_contains)
    g = df[df["category"].astype(str).str.contains(needle, na=False)].copy()
    if g.empty:
        raise ValueError(f"category_contains='{needle}'에 해당하는 상품이 없습니다.")
    # 가장 큰 카테고리 그룹 1개를 선택 (동일 문자열이 여러 레벨에 걸칠 수 있어 안전장치)
    best = None
    for cid, gg in g.groupby("category_id", sort=False):
        if best is None or len(gg) > len(best[2]):
            category_path = str(gg["category"].dropna().iloc[0]) if gg["category"].notna().any() else ""
            best = (int(cid), category_path, gg.copy())
    assert best is not None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV 기반 카테고리 리포트 전체 JSON 생성")
    ap.add_argument("--products_csv", default="products_all.csv")
    ap.add_argument("--reviews_csv", default="reviews_all.csv")
    ap.add_argument("--category_id", type=int, default=None)
    ap.add_argument("--category_contains", default=None)
    ap.add_argument("--units_per_review", type=float, default=3.0, help="리뷰 1건당 가상 판매량 계수")
    ap.add_argument("--start_year", type=int, default=2020)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--out_json", default=None, help="저장 경로(미지정 시 stdout)")
    args = ap.parse_args()

    df_products = pd.read_csv(args.products_csv)
    df_reviews = pd.read_csv(args.reviews_csv)

    cat_id, category_path, df_cat = _pick_category(
        df_products,
        category_id=args.category_id,
        category_contains=args.category_contains,
    )

    # 해당 카테고리 상품의 리뷰만 필터링
    product_ids = set(pd.to_numeric(df_cat["id"], errors="coerce").dropna().astype(int).tolist())
    # product_id가 숫자/문자 혼재일 수 있어, 동일 길이 boolean mask로 안전하게 필터링
    prod_ids_series = pd.to_numeric(df_reviews["product_id"], errors="coerce")
    mask = prod_ids_series.isin(product_ids)
    df_rev_cat = df_reviews[mask].copy()

    # 카테고리 메트릭
    metrics = aggregate_category_with_reviews(
        df_cat,
        cat_id,
        category_path,
        df_reviews=df_rev_cat if not df_rev_cat.empty else None,
    )

    # 가상 매출/판매량 기반 성장 데이터
    price_ref = metrics.price_p50 or metrics.price_p25 or metrics.price_p75
    growth_metrics = (
        calculate_yearly_growth(
            df_rev_cat,
            price_ref=price_ref,
            units_per_review=float(args.units_per_review),
            start_year=int(args.start_year),
            end_year=int(args.end_year),
        )
        if not df_rev_cat.empty
        else []
    )

    # 요약(그래프 하단 텍스트)
    market_overview = build_market_overview_summary(metrics, growth_metrics)
    growth_summary = build_growth_summary(growth_metrics)

    # 차트 데이터 (프론트에서 바로 쓰기 쉬운 형태)
    donut_items = [
        {"name": b.get("name"), "value": int(b.get("reviewCount", 0)), "share": float(b.get("share", 0.0))}
        for b in (metrics.top_brands[:10] if metrics.top_brands else [])
    ]
    growth_points = [
        {
            "year": g.year,
            "reviewCount": g.review_count,
            "estimatedUnits": g.estimated_units,
            "estimatedRevenue": g.estimated_revenue,
            "growthRate": g.growth_rate,
        }
        for g in growth_metrics
    ]

    result: Dict[str, Any] = {
        "categoryId": int(metrics.category_id),
        "categoryName": metrics.category_name,
        "charts": {
            "brandTop10Donut": {"items": donut_items},
            # 성장 그래프는 '가상 매출(estimatedRevenue)' 기준으로 쓰는 것을 기본으로 둠
            "growthLine": {"unit": "estimatedRevenue", "points": growth_points},
        },
        "summaries": {
            "marketOverviewSummary": market_overview,
            "growthSummary": growth_summary,
        },
        "assumptions": {
            "proxy": "review_count",
            "unitsPerReview": float(args.units_per_review),
            "priceRef": float(price_ref) if price_ref else None,
        },
    }

    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[OK] wrote: {args.out_json}")
    else:
        print(out)


if __name__ == "__main__":
    main()

