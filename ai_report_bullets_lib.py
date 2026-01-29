"""
ai_report_bullets_lib.py

카테고리 요약 불릿 생성을 위한 공용 유틸.
- products_all.csv (상품+리뷰 요약 컬럼) 또는 DB에서 로드한 DataFrame 기반 집계/불릿 생성
- aggregate_category_with_reviews: 리뷰 DataFrame(reviews_all 등)을 넘기면 해당 카테고리 리뷰만 분석
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def safe_literal_eval(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return default
    try:
        return ast.literal_eval(s)
    except Exception:
        return default


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def fmt_int(x: float | int) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"


def category_display_name(category_path: str) -> str:
    if not category_path:
        return "카테고리"
    if ">" in category_path:
        return category_path.split(">")[-1].strip()
    return category_path.strip()


def parse_updated_at(s: Any) -> Optional[datetime]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    ss = str(s).strip()
    if not ss or ss.lower() in {"nan", "none"}:
        return None
    try:
        return datetime.fromisoformat(ss)
    except Exception:
        return None


@dataclass
class CategoryMetrics:
    category_id: int
    category_name: str
    product_count: int
    total_review_count: int
    brand_count: int = 0
    avg_rating_weighted: Optional[float] = None
    rating_hist: Dict[str, int] = field(default_factory=lambda: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0})
    platform_dist: Dict[str, int] = field(default_factory=dict)
    top_brands: List[Dict[str, Any]] = field(default_factory=list)
    top_tags: List[Tuple[str, int]] = field(default_factory=list)
    period_from: Optional[str] = None
    period_to: Optional[str] = None
    zero_review_ratio: Optional[float] = None
    low_rating_ratio: Optional[float] = None
    top3_brand_share: Optional[float] = None
    price_p25: Optional[float] = None
    price_p50: Optional[float] = None
    price_p75: Optional[float] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    price_change_6m_median: Optional[float] = None
    top_segments: List[Tuple[str, int]] = field(default_factory=list)


def _parse_price_trend(value: Any) -> List[Dict[str, Any]]:
    data = safe_literal_eval(value, [])
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _extract_price_by_period(trend: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in trend:
        period = str(item.get("period", "")).strip()
        price = item.get("price")
        try:
            p = float(price)
        except Exception:
            continue
        if period and p > 0:
            out.setdefault(period, p)
    return out


def _extract_segments_from_specs(specs_value: Any) -> List[str]:
    segments: List[str] = []
    if specs_value is None or (isinstance(specs_value, float) and pd.isna(specs_value)):
        return segments
    spec_dict = safe_literal_eval(specs_value, {})
    if not isinstance(spec_dict, dict):
        return segments
    age_keys = ["연령", "사용연령", "적용연령", "대상연령"]
    for key in age_keys:
        if key in spec_dict:
            value = spec_dict[key]
            if isinstance(value, dict):
                for _k, sub_value in value.items():
                    if isinstance(sub_value, str) and sub_value.strip():
                        segments.append(f"연령: {sub_value.strip()}")
            elif isinstance(value, str) and value.strip():
                segments.append(f"연령: {value.strip()}")
    capacity_keys = ["용량", "규격", "크기", "사이즈"]
    for key in capacity_keys:
        if key in spec_dict:
            value = spec_dict[key]
            if isinstance(value, dict):
                for _k, sub_value in value.items():
                    if isinstance(sub_value, str) and re.search(r"\d+[mlkgL매개권조각세트팩]+", sub_value):
                        segments.append(f"용량: {sub_value.strip()}")
            elif isinstance(value, str) and re.search(r"\d+[mlkgL매개권조각세트팩]+", value):
                segments.append(f"용량: {value.strip()}")
    composition_keys = ["구성", "포함내용", "구성품"]
    for key in composition_keys:
        if key in spec_dict:
            value = spec_dict[key]
            if isinstance(value, dict):
                for _k, sub_value in value.items():
                    if isinstance(sub_value, str) and sub_value.strip() and len(sub_value.strip()) < 30:
                        segments.append(f"구성: {sub_value.strip()}")
            elif isinstance(value, str) and value.strip() and len(value.strip()) < 30:
                segments.append(f"구성: {value.strip()}")
    type_keywords = ["유기농", "무첨가", "무향", "프리미엄", "세트", "단품", "대용량", "소용량"]
    for key, value in spec_dict.items():
        if isinstance(value, dict):
            for _k, sub_value in value.items():
                if isinstance(sub_value, bool) and sub_value and _k in type_keywords:
                    segments.append(f"타입: {_k}")
    return segments


def _extract_segments_from_unit_price(unit_price_value: Any) -> List[str]:
    segments: List[str] = []
    if unit_price_value is None or (isinstance(unit_price_value, float) and pd.isna(unit_price_value)):
        return segments
    unit_str = str(unit_price_value).strip()
    match = re.search(r"원/([가-힣a-zA-Z]+)", unit_str)
    if match:
        segments.append(f"단위: {match.group(1)}")
    return segments


def _aggregate_segments(df_cat: pd.DataFrame) -> List[Tuple[str, int]]:
    all_segments: List[str] = []
    for v in df_cat.get("specifications", pd.Series([], dtype=object)).tolist():
        all_segments.extend(_extract_segments_from_specs(v))
    for v in df_cat.get("unit_price", pd.Series([], dtype=object)).tolist():
        all_segments.extend(_extract_segments_from_unit_price(v))
    if not all_segments:
        return []
    cnt = Counter(all_segments)
    return [(s, c) for s, c in cnt.most_common(10) if c >= 2]


def _extract_keywords_from_review_texts(
    texts: List[str], top_n: int = 10, min_len: int = 2, max_texts: int = 10000
) -> List[Tuple[str, int]]:
    """리뷰 텍스트에서 키워드(한글 2자 이상 등) 빈도 추출.
    
    메모리 효율을 위해 최대 max_texts개만 처리합니다.
    """
    # 메모리 효율: 너무 많은 텍스트는 샘플링
    if len(texts) > max_texts:
        import random
        texts = random.sample(texts, max_texts)
    
    # 청크 단위로 처리하여 메모리 사용량 제한
    chunk_size = 1000
    all_tokens = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        combined = " ".join(str(t).strip() for t in chunk if t is not None and str(t).strip())
        if not combined:
            continue
        
        ko_tokens = re.findall(r"[가-힣]{2,}", combined)
        en_tokens = [t.lower() for t in re.findall(r"[a-zA-Z]{2,}", combined)]
        chunk_tokens = [t for t in ko_tokens + en_tokens if t and not t.isdigit()]
        all_tokens.extend(chunk_tokens)
    
    if not all_tokens:
        return []
    cnt = Counter(all_tokens)
    return cnt.most_common(top_n)


def aggregate_category(df_cat: pd.DataFrame, category_id: int, category_path: str) -> CategoryMetrics:
    category_name = category_display_name(category_path)
    product_count = int(len(df_cat))
    brand_count = int(df_cat["manufacturer"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    total_review_count = int(pd.to_numeric(df_cat["review_count"], errors="coerce").fillna(0).sum())

    rating = pd.to_numeric(df_cat["average_rating"], errors="coerce")
    review_cnt = pd.to_numeric(df_cat["review_count"], errors="coerce").fillna(0)
    valid = rating.notna() & (review_cnt > 0)
    avg_rating_weighted = float((rating[valid] * review_cnt[valid]).sum() / review_cnt[valid].sum()) if valid.any() else None

    rating_hist: Dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for v in df_cat.get("review_rating_distribution", pd.Series([], dtype=object)).tolist():
        d = safe_literal_eval(v, {})
        if isinstance(d, dict):
            for k in ["1", "2", "3", "4", "5"]:
                rating_hist[k] += int(d.get(k, 0) or 0)

    platform_dist: Dict[str, int] = {}
    for v in df_cat.get("review_platform_distribution", pd.Series([], dtype=object)).tolist():
        d = safe_literal_eval(v, {})
        if isinstance(d, dict):
            for k, vv in d.items():
                platform_dist[str(k)] = platform_dist.get(str(k), 0) + int(vv or 0)

    df_brand = df_cat[df_cat["manufacturer"].fillna("").astype(str).str.strip() != ""].copy()
    brand_scores = df_brand.groupby(df_brand["manufacturer"].fillna("").astype(str).str.strip())["review_count"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()
    ).sort_values(ascending=False)
    top_brands_raw = brand_scores.head(10)
    brand_sum = float(brand_scores.sum()) if brand_scores.sum() > 0 else 0.0
    top_brands = [
        {"name": str(n), "reviewCount": int(c), "share": (float(c) / brand_sum) if brand_sum > 0 else 0.0}
        for n, c in top_brands_raw.items()
    ]

    tag_counts: Dict[str, int] = {}
    for v in df_cat.get("review_tags", pd.Series([], dtype=object)).tolist():
        tags = safe_literal_eval(v, [])
        if isinstance(tags, list):
            for t in tags:
                tt = str(t).strip()
                if tt:
                    tag_counts[tt] = tag_counts.get(tt, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    dt_list = [parse_updated_at(v) for v in df_cat.get("updated_at", pd.Series([], dtype=object)).tolist()]
    dt_list = [x for x in dt_list if x]
    period_from = min(dt_list).date().isoformat() if dt_list else None
    period_to = max(dt_list).date().isoformat() if dt_list else None

    zero_review_ratio = (review_cnt.fillna(0) == 0).sum() / product_count if product_count > 0 else None
    low_rating_ratio = (rating_hist["1"] + rating_hist["2"]) / total_review_count if total_review_count > 0 else None
    top3_brand_share = float(top_brands_raw.head(3).sum()) / brand_sum if brand_sum > 0 else None

    min_price = pd.to_numeric(df_cat.get("min_price"), errors="coerce")
    min_price = min_price[min_price.notna() & (min_price > 0)]
    if len(min_price) > 0:
        price_min = float(min_price.min())
        price_max = float(min_price.max())
        price_p25 = float(min_price.quantile(0.25))
        price_p50 = float(min_price.quantile(0.50))
        price_p75 = float(min_price.quantile(0.75))
    else:
        price_min = price_max = price_p25 = price_p50 = price_p75 = None

    changes: List[float] = []
    for v in df_cat.get("price_trend", pd.Series([], dtype=object)).tolist():
        trend = _parse_price_trend(v)
        by_period = _extract_price_by_period(trend)
        p1, p6 = by_period.get("1개월"), by_period.get("6개월")
        if p1 and p6 and p1 > 0 and p6 > 0:
            changes.append((p1 - p6) / p6)
    price_change_6m_median = float(pd.Series(changes).median()) if changes else None
    top_segments = _aggregate_segments(df_cat)

    return CategoryMetrics(
        category_id=int(category_id),
        category_name=category_name,
        product_count=product_count,
        brand_count=brand_count,
        total_review_count=total_review_count,
        avg_rating_weighted=avg_rating_weighted,
        rating_hist=rating_hist,
        platform_dist=platform_dist,
        top_brands=top_brands,
        top_tags=top_tags,
        period_from=period_from,
        period_to=period_to,
        zero_review_ratio=zero_review_ratio,
        low_rating_ratio=low_rating_ratio,
        top3_brand_share=top3_brand_share,
        price_p25=price_p25,
        price_p50=price_p50,
        price_p75=price_p75,
        price_min=price_min,
        price_max=price_max,
        price_change_6m_median=price_change_6m_median,
        top_segments=top_segments,
    )


def aggregate_category_with_reviews(
    df_cat: pd.DataFrame,
    category_id: int,
    category_path: str,
    df_reviews: Optional[pd.DataFrame] = None,
) -> CategoryMetrics:
    """리뷰 DataFrame이 있으면 해당 카테고리 상품의 리뷰만 모아 total_review_count, rating_hist, platform_dist, top_brands, top_tags 등을 리뷰 기준으로 계산."""
    if df_reviews is None or df_reviews.empty:
        return aggregate_category(df_cat, category_id, category_path)

    product_ids = set(df_cat["id"].astype(int).tolist())
    df_rev = df_reviews[df_reviews["product_id"].astype(int).isin(product_ids)]
    if df_rev.empty:
        return aggregate_category(df_cat, category_id, category_path)

    category_name = category_display_name(category_path)
    product_count = int(len(df_cat))
    brand_count = int(df_cat["manufacturer"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    total_review_count = int(len(df_rev))
    rating_col = pd.to_numeric(df_rev.get("rating"), errors="coerce").dropna()
    avg_rating_weighted = float(rating_col.mean()) if len(rating_col) > 0 else None
    rating_hist: Dict[str, int] = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for r in rating_col.astype(int).tolist():
        k = str(min(5, max(1, int(r))))
        rating_hist[k] = rating_hist.get(k, 0) + 1

    platform_dist: Dict[str, int] = {}
    for p in df_rev.get("platform", pd.Series([], dtype=object)).dropna().astype(str).tolist():
        p = p.strip()
        if p:
            platform_dist[p] = platform_dist.get(p, 0) + 1

    id_to_manu = df_cat.set_index("id")["manufacturer"].fillna("").astype(str).str.strip().to_dict()
    df_rev = df_rev.copy()
    df_rev["_manufacturer"] = df_rev["product_id"].astype(int).map(id_to_manu)
    df_rev = df_rev[df_rev["_manufacturer"] != ""]
    brand_scores = df_rev.groupby("_manufacturer").size().sort_values(ascending=False)
    brand_sum = float(brand_scores.sum()) if brand_scores.sum() > 0 else 0.0
    top_brands_raw = brand_scores.head(10)
    top_brands = [
        {"name": str(n), "reviewCount": int(c), "share": (float(c) / brand_sum) if brand_sum > 0 else 0.0}
        for n, c in top_brands_raw.items()
    ]

    texts = df_rev.get("review_text", pd.Series([], dtype=object)).dropna().astype(str).tolist()
    top_tags = _extract_keywords_from_review_texts(texts, top_n=10)

    dt_list = [parse_updated_at(v) for v in df_cat.get("updated_at", pd.Series([], dtype=object)).tolist()]
    dt_list = [x for x in dt_list if x]
    period_from = min(dt_list).date().isoformat() if dt_list else None
    period_to = max(dt_list).date().isoformat() if dt_list else None

    review_cnt = pd.to_numeric(df_cat["review_count"], errors="coerce").fillna(0)
    zero_review_ratio = (review_cnt == 0).sum() / product_count if product_count > 0 else None
    low_rating_ratio = (rating_hist["1"] + rating_hist["2"]) / total_review_count if total_review_count > 0 else None
    top3_brand_share = float(top_brands_raw.head(3).sum()) / brand_sum if brand_sum > 0 else None

    min_price = pd.to_numeric(df_cat.get("min_price"), errors="coerce")
    min_price = min_price[min_price.notna() & (min_price > 0)]
    if len(min_price) > 0:
        price_min = float(min_price.min())
        price_max = float(min_price.max())
        price_p25 = float(min_price.quantile(0.25))
        price_p50 = float(min_price.quantile(0.50))
        price_p75 = float(min_price.quantile(0.75))
    else:
        price_min = price_max = price_p25 = price_p50 = price_p75 = None

    changes = []
    for v in df_cat.get("price_trend", pd.Series([], dtype=object)).tolist():
        by_period = _extract_price_by_period(_parse_price_trend(v))
        p1, p6 = by_period.get("1개월"), by_period.get("6개월")
        if p1 and p6 and p1 > 0 and p6 > 0:
            changes.append((p1 - p6) / p6)
    price_change_6m_median = float(pd.Series(changes).median()) if changes else None
    top_segments = _aggregate_segments(df_cat)

    return CategoryMetrics(
        category_id=int(category_id),
        category_name=category_name,
        product_count=product_count,
        brand_count=brand_count,
        total_review_count=total_review_count,
        avg_rating_weighted=avg_rating_weighted,
        rating_hist=rating_hist,
        platform_dist=platform_dist,
        top_brands=top_brands,
        top_tags=top_tags,
        period_from=period_from,
        period_to=period_to,
        zero_review_ratio=zero_review_ratio,
        low_rating_ratio=low_rating_ratio,
        top3_brand_share=top3_brand_share,
        price_p25=price_p25,
        price_p50=price_p50,
        price_p75=price_p75,
        price_min=price_min,
        price_max=price_max,
        price_change_6m_median=price_change_6m_median,
        top_segments=top_segments,
    )


def build_category_summary_bullets(m: CategoryMetrics, max_bullets: int = 8) -> List[str]:
    bullets: List[str] = []
    if m.brand_count > 0:
        bullets.append(
            f"{m.category_name} 카테고리는 상품 {fmt_int(m.product_count)}개, 브랜드 {fmt_int(m.brand_count)}개가 등록되어 있으며, 누적 리뷰는 {fmt_int(m.total_review_count)}건입니다."
        )
    else:
        bullets.append(
            f"{m.category_name} 카테고리는 상품 {fmt_int(m.product_count)}개가 등록되어 있으며, 누적 리뷰는 {fmt_int(m.total_review_count)}건입니다."
        )

    if m.zero_review_ratio is not None:
        bullets.append(f"리뷰 0건 상품 비중은 {pct(m.zero_review_ratio)}입니다.")

    if m.avg_rating_weighted is not None and m.total_review_count > 0:
        if m.low_rating_ratio is not None:
            bullets.append(
                f"누적 리뷰 기준 가중 평균 평점은 {m.avg_rating_weighted:.2f}점이며, 1~2점 비중은 {pct(m.low_rating_ratio)}입니다."
            )
        else:
            bullets.append(f"누적 리뷰 기준 가중 평균 평점은 {m.avg_rating_weighted:.2f}점입니다.")

    if m.price_p50 is not None and m.price_p25 is not None and m.price_p75 is not None:
        line = f"가격 중앙값은 {fmt_int(m.price_p50)}원(가격 분위수 P25~P75: {fmt_int(m.price_p25)}~{fmt_int(m.price_p75)}원)입니다."
        if m.price_change_6m_median is not None:
            line += f" 최근 6개월 기준 제품별 가격 변화 중앙값은 {pct(m.price_change_6m_median)}입니다."
        bullets.append(line)

    total_platform = sum(m.platform_dist.values())
    if total_platform > 0:
        top2 = sorted(m.platform_dist.items(), key=lambda x: x[1], reverse=True)[:2]
        if len(top2) == 1:
            p1, c1 = top2[0]
            bullets.append(f"리뷰 분포 기준 상위 채널은 {p1}({pct(c1 / total_platform)})입니다.")
        else:
            (p1, c1), (p2, c2) = top2
            bullets.append(
                f"리뷰 분포 기준 상위 채널은 {p1}({pct(c1 / total_platform)}), {p2}({pct(c2 / total_platform)})입니다."
            )

    if m.top_brands:
        top3 = m.top_brands[:3]
        top3_str = ", ".join([f"{b['name']}({pct(float(b.get('share', 0.0)))})" for b in top3])
        if m.top3_brand_share is not None:
            bullets.append(f"리뷰 수 기준 상위 브랜드는 {top3_str}이며, 상위 3개 브랜드 집중도는 {pct(m.top3_brand_share)}입니다.")
        else:
            bullets.append(f"리뷰 수 기준 상위 브랜드는 {top3_str}입니다.")

    if m.top_segments:
        formatted = []
        for seg, _ in m.top_segments[:3]:
            if ":" in seg:
                _, seg_val = seg.split(":", 1)
                formatted.append(seg_val.strip())
            else:
                formatted.append(seg)
        if formatted:
            bullets.append(f"주요 세그먼트는 {', '.join(formatted)}입니다.")

    if m.top_tags:
        bullets.append(f"리뷰에서 자주 언급된 키워드는 {', '.join([k for k, _ in m.top_tags[:6]])}입니다.")

    if m.period_from and m.period_to:
        bullets.append(f"본 요약은 {m.period_from}~{m.period_to} 업데이트 데이터를 기반으로 생성했습니다.")

    return bullets[:max_bullets]


def build_vietnam_market_insight_summary(m: CategoryMetrics) -> str:
    """
    베트남 유아동 시장(Shopee) 진출을 고려하는 한국 기업을 위한 해석 요약.
    Danawa 데이터를 베트남 시장 진출 관점에서 재해석하여 의사결정에 도움이 되는 시사점을 제공.
    
    규칙:
    - 단순 통계 나열 금지
    - 한국 채널명/브랜드명 언급 금지
    - 매출 데이터 없음 → 리뷰 수를 대리 지표로 사용함을 명시
    - 5가지 관점 필수: 진입 난이도, 수요 연령대, 가격 포지셔닝, 구매 결정 요인, 경쟁 구도
    - 4~5문장, 컨설팅 리포트 톤
    """
    insights: List[str] = []
    
    # 1) 시장 진입 난이도 (포화 여부, 신규 진입 여지)
    if m.zero_review_ratio is not None and m.zero_review_ratio > 0.3:
        insights.append(
            f"리뷰 수 기준으로 추정한 시장 구조상, 리뷰가 없는 상품 비중이 {pct(m.zero_review_ratio)}에 달해 "
            f"신규 진입 여지가 상대적으로 넓은 편이며, 브랜드 수({fmt_int(m.brand_count)}개) 대비 "
            f"상품 수({fmt_int(m.product_count)}개)가 많지 않아 경쟁 강도는 중간 수준으로 판단됩니다."
        )
    elif m.brand_count > 100:
        insights.append(
            f"브랜드 수({fmt_int(m.brand_count)}개)가 많고 상품 수({fmt_int(m.product_count)}개)가 "
            f"상대적으로 적어 시장 진입 난이도가 높은 편이며, 리뷰 수 기준 추정 시 "
            f"기존 브랜드들의 영향력이 분산되어 있어 차별화된 포지셔닝이 중요합니다."
        )
    else:
        insights.append(
            f"브랜드 수({fmt_int(m.brand_count)}개)와 상품 수({fmt_int(m.product_count)}개)를 고려할 때 "
            f"시장 진입 난이도는 중간 수준이며, 리뷰 수 기준으로 추정한 시장 구조상 "
            f"신규 진입을 위한 여지가 존재합니다."
        )
    
    # 2) 핵심 수요 연령대/사용 단계
    if m.top_segments:
        age_segments = [s for s, _ in m.top_segments if "연령" in s or "개월" in s or "세" in s]
        if age_segments:
            age_str = ", ".join([s.split(":")[-1].strip() if ":" in s else s for s in age_segments[:2]])
            insights.append(
                f"주요 세그먼트 분석 결과, 핵심 수요는 {age_str} 연령대에 집중되어 있으며, "
                f"이 연령대를 타겟팅한 제품 개발 및 마케팅 전략이 효과적일 것으로 판단됩니다."
            )
        else:
            insights.append(
                f"세그먼트 분석 결과, 특정 연령대보다는 다양한 사용 단계에서 수요가 분산되어 있어 "
                f"범용성 높은 제품 포트폴리오 구성이 유리할 수 있습니다."
            )
    else:
        insights.append(
            f"세그먼트 정보가 제한적이나, 리뷰 수({fmt_int(m.total_review_count)}건)와 평균 평점 "
            f"({m.avg_rating_weighted:.2f}점)을 고려할 때 시장 전반의 수요는 안정적인 편입니다."
        )
    
    # 3) 가격 포지셔닝 시사점
    if m.price_p50 is not None and m.price_p25 is not None and m.price_p75 is not None:
        mid_krw = m.price_p50
        low_krw = m.price_p25
        high_krw = m.price_p75
        # 간단한 가격대 분류 (한국 시장 기준, 베트남에서는 환율/구매력 차이 고려 필요)
        if mid_krw < 50000:
            price_tier = "중저가"
        elif mid_krw < 150000:
            price_tier = "중가"
        else:
            price_tier = "프리미엄"
        
        insights.append(
            f"가격 분포 분석 결과, 중앙값 기준 {price_tier}대({fmt_int(mid_krw)}원, "
            f"P25~P75: {fmt_int(low_krw)}~{fmt_int(high_krw)}원)에 집중되어 있으며, "
            f"베트남 시장 진출 시 현지 구매력과 환율을 고려한 가격 전략 수립이 필요합니다."
        )
    
    # 4) 구매 결정 요인 (리뷰 키워드 해석)
    if m.top_tags:
        key_tags = [k for k, _ in m.top_tags[:4]]
        tag_str = ", ".join(key_tags)
        insights.append(
            f"리뷰 분석 결과, 소비자들이 가장 자주 언급하는 요소는 '{tag_str}' 등이며, "
            f"이러한 키워드가 반영된 제품 특성 및 마케팅 메시지가 구매 결정에 영향을 미치는 것으로 추정됩니다."
        )
    elif m.avg_rating_weighted is not None and m.avg_rating_weighted > 4.5:
        insights.append(
            f"평균 평점({m.avg_rating_weighted:.2f}점)이 높고 저평점 비중({pct(m.low_rating_ratio) if m.low_rating_ratio else '낮음'})이 낮아 "
            f"시장 전반의 제품 만족도가 높은 편이며, 품질과 신뢰성이 구매 결정의 핵심 요인으로 작용할 가능성이 높습니다."
        )
    else:
        insights.append(
            f"리뷰 데이터 분석 결과, 구매 결정 요인은 다양하게 분산되어 있어 "
            f"타겟 고객층별 맞춤형 마케팅 전략이 효과적일 수 있습니다."
        )
    
    # 5) 경쟁 구도 및 시장 집중도 시사점
    if m.top_brands and m.top3_brand_share is not None:
        top3_names = [b["name"] for b in m.top_brands[:3]]
        # 한국 브랜드명 제거 (일반명으로 대체하거나 생략)
        insights.append(
            f"리뷰 수 기준으로 추정한 시장 영향력 분석 결과, 상위 3개 브랜드의 집중도는 "
            f"{pct(m.top3_brand_share)}로 중간 수준이며, 시장이 완전히 독점적이지 않아 "
            f"차별화된 제품과 마케팅으로 시장 점유율 확보가 가능할 것으로 판단됩니다. "
            f"다만, 실제 매출 데이터가 없어 리뷰 수를 대리 지표로 사용한 추정치임을 고려해야 합니다."
        )
    else:
        insights.append(
            f"리뷰 수 기준으로 추정한 시장 구조상, 브랜드별 영향력이 상대적으로 분산되어 있어 "
            f"신규 진입 브랜드도 적절한 포지셔닝과 마케팅으로 시장 진입이 가능할 것으로 보입니다. "
            f"단, 실제 매출 데이터가 없어 리뷰 수를 대리 지표로 사용한 추정치임을 명시합니다."
        )
    
    return " ".join(insights)
