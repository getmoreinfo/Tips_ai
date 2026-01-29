"""
22_prepare_report_summary_sft.py
역할:
  - products_all.csv + reviews_all.csv로부터
    "커머스 시장 리포트 요약" 학습용 JSONL(SFT)을 생성한다.
  - marketOverviewSummary: 시장 구조 요약 (4~6문장)
  - growthSummary: 성장률 요약 (5~8문장)

출력(JSONL) 포맷:
  {"messages":[{"role":"system","content":"..."},{"role":"user","content":"{...json...}"},{"role":"assistant","content":"{...json...}"}]}
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai_report_bullets_lib import (
    aggregate_category_with_reviews,
    CategoryMetrics,
)
from ai_report_bullets_lib import _extract_keywords_from_review_texts  # type: ignore
from report_summary_lib import (
    build_market_overview_summary,
    build_growth_summary,
    calculate_yearly_growth,
)


SYSTEM_PROMPT = (
    "너는 커머스 시장 리포트를 작성하는 시니어 컨설팅 애널리스트다. "
    "제공되는 데이터와 지표는 이미 계산이 완료된 값이며, "
    "너의 역할은 계산이 아니라 '그래프와 지표가 의미하는 바를 "
    "의사결정자가 질문 없이 이해하도록 해설하는 것'이다.\n\n"
    "이 리포트는 Danawa 유아동 품목 데이터를 기반으로, "
    "해외(베트남) 진출을 검토하는 기업 의사결정자를 대상으로 한다. "
    "의사결정자는 기술이나 데이터 구조를 알지 못하며, "
    "리포트를 읽고 추가 질문 없이 다음 액션을 판단할 수 있어야 한다.\n\n"
    "핵심 전제: "
    "- Danawa에는 실제 판매량·매출액 데이터가 존재하지 않는다. "
    "- 따라서 본 리포트에서 사용되는 '매출', '점유율', '성장률'은 "
    "모두 '리뷰 수'를 판매/매출의 대리 지표(proxy)로 사용한 상대적 추정 결과임을 전제로 한다. "
    "- 이 전제는 리포트 초반 또는 각 요약의 첫 문장에서 명확하게 언급되어야 하며, "
    "실제 매출처럼 단정적으로 표현해서는 안 된다.\n\n"
    "작성 규칙: "
    "- 단순 수치 나열 금지 (그래프에 이미 보이는 정보 반복 금지) "
    "- '~로 판단됩니다', '~가 필요합니다' 같은 애매한 결론 금지 "
    "- 각 문장은 반드시 '의미 → 해석 → 시사점' 구조를 가질 것 "
    "- 실제 매출액, 금액 단위, 확정적 점유율 표현 금지 "
    "- 과장, 마케팅 문구, 낙관적 전망 금지 "
    "- 모든 문장은 한국어 존댓말, 컨설팅 리포트 톤 유지\n\n"
    "출력 형식: "
    "- marketOverviewSummary: 시장 구조 요약 (4~6문장, 단락형) "
    "- growthSummary: 성장률 요약 (5~8문장, 단락형) "
    "- 각 요약은 단락형 문장으로 출력하며, 불릿이나 번호는 사용하지 않는다."
)


def _truncate_text(s: Any, max_chars: int) -> str:
    ss = str(s) if s is not None else ""
    ss = " ".join(ss.split())
    if max_chars and max_chars > 0 and len(ss) > max_chars:
        return ss[: max_chars - 1] + "…"
    return ss


def _build_review_insights(
    df_rev_cat: pd.DataFrame,
    *,
    top_keywords: int,
    pos_examples: int,
    neg_examples: int,
    max_example_chars: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    리뷰 텍스트 기반 인사이트를 user_input에 넣기 위한 요약.
    - topKeywords: 상위 키워드(top_n)
    - positiveExamples / negativeExamples: 대표 리뷰 문장 샘플
    """
    if df_rev_cat is None or df_rev_cat.empty:
        return None
    if "review_text" not in df_rev_cat.columns:
        return None

    texts = (
        df_rev_cat["review_text"]
        .dropna()
        .astype(str)
        .map(lambda x: _truncate_text(x, max_example_chars))
        .tolist()
    )
    if not texts:
        return None

    kws = _extract_keywords_from_review_texts(texts, top_n=int(max(0, top_keywords)))

    df = df_rev_cat.copy()
    rating = pd.to_numeric(df.get("rating"), errors="coerce")
    df["_rating"] = rating
    df_pos = df[df["_rating"].notna() & (df["_rating"] >= 4)]
    df_neg = df[df["_rating"].notna() & (df["_rating"] <= 2)]

    rnd = random.Random(seed)

    def _sample_texts(dff: pd.DataFrame, n: int) -> List[str]:
        if n <= 0 or dff.empty:
            return []
        pool = (
            dff["review_text"]
            .dropna()
            .astype(str)
            .map(lambda x: _truncate_text(x, max_example_chars))
            .tolist()
        )
        if not pool:
            return []
        if len(pool) <= n:
            return pool
        return rnd.sample(pool, k=n)

    return {
        "topKeywords": list(kws),
        "positiveExamples": _sample_texts(df_pos, int(pos_examples)),
        "negativeExamples": _sample_texts(df_neg, int(neg_examples)),
        "hasReviewText": True,
        "reviewTextCount": int(len(texts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="products_all.csv")
    ap.add_argument("--reviews_csv", default="reviews_all.csv")
    ap.add_argument("--out_jsonl", default="training_report_summary_sft.jsonl")
    ap.add_argument("--min_products", type=int, default=10, help="카테고리당 최소 상품 수")
    ap.add_argument("--max_categories", type=int, default=1000, help="최대 카테고리 수")
    ap.add_argument(
        "--samples_per_category",
        type=int,
        default=10,
        help="카테고리당 생성할 학습 샘플 수",
    )
    ap.add_argument(
        "--subsample_ratio",
        type=float,
        default=0.85,
        help="서브샘플링 비율",
    )
    ap.add_argument(
        "--use_review_text",
        action="store_true",
        help="리뷰 텍스트(review_text) 기반 reviewInsights(topKeywords/대표리뷰)를 user_input에 포함",
    )
    ap.add_argument("--review_top_keywords", type=int, default=10, help="reviewInsights.topKeywords 개수")
    ap.add_argument("--review_pos_examples", type=int, default=3, help="긍정 리뷰 샘플 개수(평점>=4)")
    ap.add_argument("--review_neg_examples", type=int, default=2, help="부정 리뷰 샘플 개수(평점<=2)")
    ap.add_argument("--review_max_chars", type=int, default=180, help="대표 리뷰 샘플 최대 글자 수")
    ap.add_argument("--seed", type=int, default=42, help="재현을 위한 랜덤 시드")
    args = ap.parse_args()

    print("=" * 60)
    print("커머스 시장 리포트 요약 SFT 데이터 생성")
    print("=" * 60)
    print(f"입력: {args.input_csv}")
    print(f"리뷰 데이터: {args.reviews_csv}")
    print(f"출력: {args.out_jsonl}")

    # 데이터 로드
    df_products = pd.read_csv(args.input_csv)
    df_reviews = pd.read_csv(args.reviews_csv)
    print(f"상품 데이터: {len(df_products):,}개")
    print(f"리뷰 데이터: {len(df_reviews):,}개")

    # 필수 컬럼 확인
    required_product_cols = [
        "category_id",
        "category",
        "manufacturer",
        "review_count",
        "average_rating",
    ]
    for c in required_product_cols:
        if c not in df_products.columns:
            raise ValueError(f"상품 데이터에 필수 컬럼 누락: {c}")

    if "review_date" not in df_reviews.columns:
        raise ValueError("리뷰 데이터에 'review_date' 컬럼이 없습니다.")

    # 카테고리별로 그룹화
    df_products["category_id"] = pd.to_numeric(df_products["category_id"], errors="coerce")
    df_products = df_products[df_products["category_id"].notna()]

    groups = df_products.groupby("category_id", sort=False)

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
                    n_take = max(args.min_products, int(n_rows * subsample_ratio))
                    n_take = min(n_take, n_rows)
                    g_use = g.sample(n=n_take, random_state=args.seed + sample_idx)
                else:
                    g_use = g

                if len(g_use) < args.min_products:
                    continue

                # 해당 카테고리 상품의 리뷰만 필터링
                product_ids = set(g_use["id"].astype(int).tolist())
                df_rev_cat = df_reviews[df_reviews["product_id"].astype(int).isin(product_ids)].copy()

                # 메트릭스 계산
                metrics = aggregate_category_with_reviews(
                    g_use, cat_id, category_path, df_reviews=df_rev_cat if not df_rev_cat.empty else None
                )

                # 연도별 성장률 계산
                # Danawa에는 매출/판매량이 없으므로 '리뷰 수 -> (가상)매출/판매량'으로 환산한 성장률을 사용
                price_ref = metrics.price_p50 or metrics.price_p25 or metrics.price_p75
                growth_metrics = (
                    calculate_yearly_growth(df_rev_cat, price_ref=price_ref, units_per_review=3.0)
                    if not df_rev_cat.empty
                    else []
                )

                # 리포트 요약 생성 (템플릿 기반)
                market_overview = build_market_overview_summary(metrics, growth_metrics)
                growth_summary = build_growth_summary(growth_metrics)

                # 입력 데이터 구성
                user_input = {
                    "reportType": "market_report_summary",
                    "categoryId": metrics.category_id,
                    "categoryName": metrics.category_name,
                    "metrics": {
                        "productCount": metrics.product_count,
                        "brandCount": metrics.brand_count,
                        "totalReviewCount": metrics.total_review_count,
                        "avgRatingWeighted": metrics.avg_rating_weighted,
                        "ratingHistogram": metrics.rating_hist,
                        "zeroReviewRatio": metrics.zero_review_ratio,
                        "lowRatingRatio": metrics.low_rating_ratio,
                        "top3BrandShare": metrics.top3_brand_share,
                        "topBrands": metrics.top_brands[:10],
                        "topTags": [{"tag": k, "count": v} for k, v in metrics.top_tags[:10]],
                        "topSegments": [{"segment": s, "count": c} for s, c in metrics.top_segments[:10]],
                    },
                    "growthData": [
                        {
                            "year": g.year,
                            "reviewCount": g.review_count,
                            "estimatedUnits": g.estimated_units,
                            "estimatedRevenue": g.estimated_revenue,
                            "priceRef": g.price_ref,
                            "growthRate": g.growth_rate,
                        }
                        for g in growth_metrics
                    ],
                }

                # (선택) 리뷰 텍스트 인사이트 추가: 키워드 + 대표 리뷰(긍정/부정)
                if args.use_review_text:
                    ri = _build_review_insights(
                        df_rev_cat,
                        top_keywords=int(args.review_top_keywords),
                        pos_examples=int(args.review_pos_examples),
                        neg_examples=int(args.review_neg_examples),
                        max_example_chars=int(args.review_max_chars),
                        seed=int(args.seed) + int(sample_idx),
                    )
                    if ri:
                        user_input["reviewInsights"] = ri

                # 출력 데이터 구성
                target = {
                    "marketOverviewSummary": market_overview,
                    "growthSummary": growth_summary,
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
    print("다음 단계: 23_train_report_summary_lora.py 로 LoRA 학습")


if __name__ == "__main__":
    main()
