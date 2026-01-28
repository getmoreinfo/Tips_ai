# generate_category_bullets_from_db.py
# 역할: CSV 없이 danawa_crawlingdb에서 카테고리별 상품·리뷰를 직접 조회한 뒤
#       같은 분석(aggregate_category_with_reviews + build_category_summary_bullets)으로 불릿 생성.
#
# 사용 예:
#   python generate_category_bullets_from_db.py --category_contains 유모차
#   python generate_category_bullets_from_db.py --category_id 4

from __future__ import annotations

import argparse
import json
import sys

from ai_report_bullets_lib import (
    aggregate_category_with_reviews,
    build_category_summary_bullets,
    build_vietnam_market_insight_summary,
)
from db_category_loader import load_category_from_db


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DB에서 카테고리 상품·리뷰 직접 조회 후 불릿 생성 (CSV 불필요)"
    )
    ap.add_argument("--category_id", type=int, default=None, help="카테고리 ID")
    ap.add_argument("--category_contains", default=None, help="카테고리 경로에 포함된 문자열 (예: 유모차)")
    ap.add_argument("--max_bullets", type=int, default=8, help="최대 불릿 개수")
    ap.add_argument(
        "--vietnam_insight",
        action="store_true",
        help="베트남 시장 진출 관점의 해석 요약 추가 (컨설팅 리포트 톤)",
    )
    args = ap.parse_args()

    if not args.category_id and not args.category_contains:
        ap.error("--category_id 또는 --category_contains 중 하나는 필요합니다.")

    try:
        df_products, df_reviews, category_id, category_path = load_category_from_db(
            category_id=args.category_id,
            category_contains=args.category_contains,
        )
    except Exception as e:
        print(f"[ERROR] DB 조회 실패: {e}", file=sys.stderr)
        sys.exit(1)

    m = aggregate_category_with_reviews(
        df_products,
        category_id,
        category_path,
        df_reviews=df_reviews if not df_reviews.empty else None,
    )
    bullets = build_category_summary_bullets(m, max_bullets=args.max_bullets)
    out = {"summaryBullets": bullets, "categoryId": category_id, "categoryName": m.category_name}
    
    if args.vietnam_insight:
        vietnam_summary = build_vietnam_market_insight_summary(m)
        out["vietnamMarketInsight"] = vietnam_summary
    
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
