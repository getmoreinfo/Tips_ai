"""
report_summary_lib.py

커머스 시장 리포트 요약 생성을 위한 라이브러리.
- marketOverviewSummary: 시장 구조 요약 (4~6문장)
- growthSummary: 성장률 요약 (5~8문장)

베트남 진출 검토 기업 의사결정자를 위한 컨설팅 리포트 톤.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ai_report_bullets_lib import CategoryMetrics, pct, fmt_int


@dataclass
class GrowthMetrics:
    """연도별 성장률 지표"""
    year: int
    review_count: int  # 리뷰 수 (관측치)
    estimated_units: int  # (가상) 판매량 추정치: review_count 기반
    estimated_revenue: float  # (가상) 매출 추정치: estimated_units * price_ref
    growth_rate: Optional[float] = None  # 전년 대비 성장률 (%), estimated_revenue 기준
    price_ref: Optional[float] = None  # 추정에 사용한 대표 가격(중앙값 등)


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def estimate_units_and_revenue_from_reviews(
    review_count: int,
    price_ref: Optional[float],
    units_per_review: float = 3.0,
) -> Tuple[int, float]:
    """
    Danawa에는 판매/매출이 없으므로, 리뷰 수를 기반으로 가상 판매/매출을 생성한다.
    - review_count가 많을수록 units/revenue가 단조 증가하도록 '결정적(deterministic)'으로 계산한다.
    - units_per_review는 '리뷰 1건당 판매량'에 대한 가정(계수)이며, 운영 시 조정 가능한 파라미터다.
    """
    rc = int(max(0, review_count))
    upr = float(units_per_review) if units_per_review and units_per_review > 0 else 3.0
    units = int(round(rc * upr))
    p = _safe_float(price_ref)
    if p is None or p <= 0:
        # 가격이 없을 때는 0원으로 두면 그래프가 죽으므로 1을 사용해 상대 비교만 가능하게 둔다.
        p = 1.0
    revenue = float(units) * float(p)
    return units, revenue


def calculate_yearly_growth(
    df_reviews: pd.DataFrame,
    price_ref: Optional[float] = None,
    units_per_review: float = 3.0,
    start_year: int = 2020,
    end_year: int = 2025,
) -> List[GrowthMetrics]:
    """
    리뷰 데이터에서 연도별 리뷰 수와 성장률 계산.
    
    Args:
        df_reviews: 리뷰 DataFrame (review_date 컬럼 필요)
        start_year: 시작 연도
        end_year: 종료 연도
    
    Returns:
        연도별 GrowthMetrics 리스트
    """
    if df_reviews.empty:
        return []

    # 날짜 컬럼 선택 (DB/CSV 모두 대응)
    date_col = None
    for c in ["review_date", "first_seen_at", "last_seen_at", "created_at", "updated_at"]:
        if c in df_reviews.columns:
            date_col = c
            break
    if date_col is None:
        return []

    df_reviews = df_reviews.copy()
    df_reviews[date_col] = pd.to_datetime(df_reviews[date_col], errors="coerce")
    df_reviews = df_reviews[df_reviews[date_col].notna()]
    if df_reviews.empty:
        return []

    df_reviews["year"] = df_reviews[date_col].dt.year
    
    # 연도별 리뷰 수 집계
    yearly_counts = df_reviews.groupby("year").size().to_dict()
    
    # GrowthMetrics 생성 (가상 판매/매출 생성 포함)
    growth_metrics = []
    prev_revenue: Optional[float] = None
    
    for year in range(start_year, end_year + 1):
        count = yearly_counts.get(year, 0)
        units, revenue = estimate_units_and_revenue_from_reviews(
            int(count),
            price_ref=price_ref,
            units_per_review=units_per_review,
        )
        growth_rate: Optional[float] = None
        
        if prev_revenue is not None and prev_revenue > 0:
            growth_rate = ((revenue - prev_revenue) / prev_revenue) * 100
        
        growth_metrics.append(GrowthMetrics(
            year=year,
            review_count=int(count),
            estimated_units=int(units),
            estimated_revenue=float(revenue),
            growth_rate=growth_rate,
            price_ref=_safe_float(price_ref),
        ))
        prev_revenue = revenue
    
    return growth_metrics


def build_market_overview_summary(
    metrics: CategoryMetrics,
    growth_metrics: List[GrowthMetrics],
) -> str:
    """
    '해석 중심 요약(Insight Summary)' 생성 (5~7문장).

    구조(문장 단위):
    1) 전제 문장
    2) 시장 한 줄 결론
    3) 경쟁 구조 요약
    4) 시장 성숙도 및 품질 수준
    5) 소비자 의사결정 방식
    6) 실행 시사점
    """
    summary_parts: List[str] = []

    # 1) 전제 문장
    if metrics.category_name:
        summary_parts.append(
            f"본 분석은 {metrics.category_name} 카테고리에서 수집된 리뷰 수를 실제 매출 대신 판매 규모와 점유를 가늠하기 위한 대리 지표로 활용해 구조를 추정한 결과입니다."
        )
    else:
        summary_parts.append(
            "본 분석은 해당 카테고리의 리뷰 수를 실제 매출 대신 판매 규모와 점유를 가늠하기 위한 대리 지표로 활용해 구조를 추정한 결과입니다."
        )

    # 2) 시장 한 줄 결론 (진입 난이도 + 성숙/성장 뉘앙스, 핵심 수치 1개 포함)
    top3 = metrics.top3_brand_share or 0.0
    zero_ratio = metrics.zero_review_ratio if metrics.zero_review_ratio is not None else 0.0

    # 상위 3개 브랜드 비중 하나만 핵심 수치로 사용
    top3_text = pct(top3) if top3 > 0 else "상당한 비중"
    if top3 >= 0.65:
        comp_phrase = f"리뷰 기준 상위 3개 브랜드가 전체의 {top3_text}를 차지해 리더 영향력이 매우 강한 구조로, 신규 브랜드 입장에서는 진입 장벽이 상당히 높은 시장입니다."
    elif top3 >= 0.45:
        comp_phrase = f"리뷰 기준 상위 3개 브랜드 비중이 {top3_text} 수준으로 리더 그룹과 후발 주자가 공존해 진입 자체는 가능하지만, 상위 그룹과의 격차를 줄이기 위한 전략이 필수적인 시장입니다."
    else:
        comp_phrase = f"상위 3개 브랜드 비중이 {top3_text}에 그쳐 브랜드 지배력이 한쪽으로 과도하게 쏠려 있지 않은 분산형 구조로, 차별화 포인트만 명확하다면 신규 진입 여지가 비교적 넓은 시장입니다."

    if zero_ratio >= 0.35:
        maturity_phrase = f"동시에 리뷰 0건 상품 비중이 {pct(zero_ratio)} 수준으로 적지 않아 아직 검증되지 않은 시도와 빈 칸이 많은 성장·전환 단계로 볼 수 있습니다."
    elif zero_ratio <= 0.15:
        maturity_phrase = f"동시에 리뷰 0건 상품 비중이 {pct(zero_ratio)}로 높지 않아 이미 검증과 경쟁이 상당히 진행된 성숙 시장에 가깝습니다."
    else:
        maturity_phrase = "검증된 상품과 초기 단계 상품이 섞여 있는 과도기 시장으로, 성장성과 경쟁 강도가 동시에 존재하는 환경입니다."

    summary_parts.append(f"전체적으로 보면 {comp_phrase} 동시에 {maturity_phrase}")

    # 3) 경쟁 구조 요약 (브랜드 집중 + HHI 해석)
    hhi: Optional[float] = None
    if metrics.top_brands:
        try:
            hhi = sum((float(b.get("share", 0.0)) * 100) ** 2 for b in metrics.top_brands)
        except Exception:
            hhi = None

    if hhi is not None:
        hhi_int = int(round(hhi))
        if hhi >= 5000:
            hhi_interp = f"상위 10개 브랜드 기준 HHI 지수는 약 {hhi_int}로, 상위 몇 개 브랜드가 사실상 시장을 주도하는 독점에 가까운 구조입니다."
        elif hhi >= 2500:
            hhi_interp = f"상위 10개 브랜드 기준 HHI 지수는 약 {hhi_int}로, 일정 수의 리더 브랜드가 시장을 강하게 쥐고 있고 나머지 다수 브랜드는 틈새 영역에서 경쟁하는 고집중 구조입니다."
        else:
            hhi_interp = f"상위 10개 브랜드 기준 HHI 지수는 약 {hhi_int} 수준으로, 여러 브랜드가 비슷한 비중으로 경쟁하는 구조에 가까워 단일 브랜드의 영향력이 절대적으로 크지는 않습니다."
    else:
        hhi_interp = "리더 그룹과 후발 주자가 공존하지만, 특정 소수 브랜드에 모든 수요가 쏠린 극단적 구조는 아닙니다."

    summary_parts.append(
        "경쟁 구도 측면에서는 리뷰 기준 상위 브랜드들이 일정 비중을 차지하면서도 "
        "완전 독점보다는 리더 그룹과 후발 주자가 구분되는 구조이며, "
        f"{hhi_interp}"
    )

    # 4) 시장 성숙도 및 품질 수준 (리뷰 0건 + 평점, 문단당 핵심 수치 1~2개)
    if metrics.avg_rating_weighted is not None:
        rating_text = f"전반적인 가중 평균 평점은 {metrics.avg_rating_weighted:.1f}점 수준으로, "
        if metrics.low_rating_ratio is not None and metrics.low_rating_ratio > 0.1:
            rating_text += (
                "만족도는 대체로 높지만 저평점 비중도 무시할 수 없어 세부 상품군에 따라 품질 편차와 서비스 리스크가 함께 존재하는 시장입니다."
            )
        else:
            rating_text += (
                "저평점 비중이 높지 않아 전반적으로는 품질과 경험에 대한 만족도가 안정적인 편에 속합니다."
            )
        summary_parts.append(rating_text)

    # 5) 소비자 의사결정 방식 (세그먼트/키워드 기반, 리뷰 규모 1개 수치 포함)
    if metrics.top_segments or metrics.top_tags:
        segment_keywords: List[str] = []
        if metrics.top_segments:
            segment_keywords.extend([s[0] for s in metrics.top_segments[:3]])
        if metrics.top_tags:
            segment_keywords.extend([t[0] for t in metrics.top_tags[:3]])

        if segment_keywords:
            # 전체 리뷰 규모를 한 번만 언급
            total_reviews_text = (
                f"{fmt_int(metrics.total_review_count)}건 내외의 리뷰 텍스트를 기준으로 볼 때 "
                if metrics.total_review_count
                else ""
            )

            keyword_text = (
                f"주요 스펙과 세그먼트, 리뷰 키워드를 종합하면 소비자들은 "
                f"{', '.join(segment_keywords[:3])} 등 구체적인 사용 상황과 체감 효용을 기준으로 상품을 비교하는 경향이 강해, "
                "단순 브랜드 인지도보다는 '우리 아이에게 맞는 상황과 조건'을 얼마나 설득력 있게 제시하는지가 의사결정의 핵심 축이 되는 시장입니다."
            )
            summary_parts.append(total_reviews_text + keyword_text)

    # 6) 실행 시사점 (행동 가능한 한 문단)
    action_clauses: List[str] = []
    if zero_ratio >= 0.3:
        action_clauses.append(
            "초기에는 소수의 핵심 상품에 리뷰를 집중적으로 모아 '검증된 선택지'라는 신뢰를 빠르게 만드는 것이 중요합니다"
        )
    else:
        action_clauses.append(
            "이미 리뷰가 풍부한 기존 상품과 정면 승부하기보다는 덜 포화된 세부 세그먼트에서 차별화된 포지셔닝을 선점하는 전략이 필요합니다"
        )

    if metrics.top_segments:
        action_clauses.append(
            "타깃 연령·사용 상황·휴대성 등 구체적인 사용 시나리오를 상품명과 상세 페이지 전반에 일관되게 녹여내야 합니다"
        )
    else:
        action_clauses.append(
            "상품 특징을 추상적인 슬로건이 아니라 실제 사용 장면과 연결된 메시지로 풀어내는 것이 효과적입니다"
        )

    exec_text = (
        "신규 진입을 고려할 경우 "
        + " 그리고 ".join(action_clauses)
        + " 또한 리뷰 콘텐츠는 단순 별점이 아니라 구체적인 사용 경험과 사진을 중심으로 설계해, 리뷰 자체가 마케팅 자산이 되도록 관리하는 것이 필요합니다."
    )
    summary_parts.append(exec_text)

    return " ".join(summary_parts)


def build_growth_summary(
    growth_metrics: List[GrowthMetrics],
) -> str:
    """
    성장률 요약 생성 (5~8문장).
    
    프롬프트 요구사항:
    - 리뷰 수 기반 추정 성장률임을 명시
    - 2020~2025 전체 흐름 요약
    - 최고 성장 구간과 하락 구간의 의미
    - 최근 1~2년 모멘텀 해석
    - 데이터 기반 한계 또는 주의점
    """
    if not growth_metrics:
        return "연도별 성장 데이터가 부족하여 성장 추세를 분석할 수 없습니다."
    
    summary_parts = [
        "성장 관련 해석은 실제 매출 데이터가 아니라, "
        "연도별 리뷰 수를 기반으로 가상의 판매량·매출을 산정한 뒤 그 변화를 성장률로 환산한 추정치라는 점을 전제로 이해하셔야 합니다."
    ]
    
    # 전체 흐름 요약
    _total_reviews = sum(g.review_count for g in growth_metrics if 2020 <= g.year <= 2025)
    growth_rates = [g.growth_rate for g in growth_metrics if g.growth_rate is not None]
    
    if growth_rates:
        max_growth = max(growth_rates)
        min_growth = min(growth_rates)
        _avg_growth = sum(growth_rates) / len(growth_rates)
        
        # 최고 성장 구간 찾기
        max_growth_year = None
        for g in growth_metrics:
            if g.growth_rate == max_growth:
                max_growth_year = g.year
                break
        
        flow_text = (
            f"2020년부터 2025년까지의 흐름을 보면, 리뷰 관측 기준상 "
            f"초기에는 저변이 점차 넓어지는 완만한 성장 국면을 거쳐, "
            f"특정 시점 이후 리뷰 증가 속도가 뚜렷이 가속된 구간과 성장세가 둔화·조정되는 구간이 교차하며 "
            f"전반적으로는 장기 성장성 위에 중간 중간 변동성이 낀 패턴으로 요약됩니다."
        )
        summary_parts.append(flow_text)
        
        # 최고 성장 구간 해석
        if max_growth_year:
            growth_text = (
                f"리뷰 기준 최고 성장 구간은 {max_growth_year}년 전후로, "
                f"단기간에 신규 수요와 제품 구성이 빠르게 유입된 시기라 볼 수 있으며, "
                f"이는 채널 입점 확대나 제품 포트폴리오 다변화 등이 복합적으로 반영된 결과일 가능성이 있지만, "
                f"외부 요인을 단정적으로 특정하기보다는 '리뷰 노출과 소비자 관심이 집중된 시기' 정도로 해석하는 것이 안전합니다."
            )
            summary_parts.append(growth_text)
        
        # 하락 구간 해석
        if min_growth < 0:
            decline_text = (
                f"반대로 리뷰 증가세가 눈에 띄게 둔화된 구간은 실제 수요 위축뿐 아니라 "
                f"리뷰 작성 행태 변화, 캠페인 종료, 노출 정책 변경 등 다양한 요인이 얽혀 있을 수 있어, "
                f"단일 원인으로 해석하기보다 '성장 모멘텀이 일시적으로 약해진 관측 구간'으로 보는 것이 합리적입니다."
            )
            summary_parts.append(decline_text)
        
        # 최근 1~2년 모멘텀
        recent_years = [g for g in growth_metrics if g.year >= 2023]
        if len(recent_years) >= 2:
            recent_growth = [g.growth_rate for g in recent_years if g.growth_rate is not None]
            if recent_growth:
                recent_avg = sum(recent_growth) / len(recent_growth)
                momentum_text = (
                    f"최근 1~2년 데이터를 기준으로 보면, 리뷰 기준 성장세는 "
                    f"과거 고성장 국면에 비해 속도가 다소 조정된 상태에서, "
                    f"완만한 증가 또는 횡보 구간을 오가며 단기 급등보다는 안정적인 체질을 만들어가는 단계로 읽힙니다."
                )
                summary_parts.append(momentum_text)
        
        # 데이터 기반 한계
        limit_text = (
            f"다만 리뷰 날짜 분포를 보면 특정 기간에 캠페인·프로모션성 리뷰가 집중된 흔적이 있을 수 있고, "
            f"짧은 기간의 급격한 변동 구간은 실제 수요의 구조적 변화라기보다는 관측상의 왜곡일 위험도 존재하므로, "
            f"단기 스파이크보다 여러 해에 걸친 중장기 추세를 중심으로 시장의 '온도'를 판단하는 것이 필요합니다."
        )
        summary_parts.append(limit_text)
    
    return " ".join(summary_parts)
