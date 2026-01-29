"""
24_generate_report_summary.py
역할: 학습된 LoRA 모델을 사용하여 커머스 시장 리포트 요약 생성

사용 예:
  python 24_generate_report_summary.py --model_dir results_report/qwen2.5-3b-lora-report-summary --category_contains 유모차
  python 24_generate_report_summary.py --model_dir results_report/qwen2.5-3b-lora-report-summary --category_id 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_report_bullets_lib import aggregate_category_with_reviews
from db_category_loader import load_category_from_db
from report_summary_lib import calculate_yearly_growth

# 프록시 설정 제거
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """텍스트에서 첫 번째 JSON 객체 추출"""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_to_result_json(raw: str) -> Dict[str, Any]:
    """
    모델이 JSON을 안 지키고 마크다운/불릿으로 출력했을 때 최소한의 형태로 복구.
    - marketOverviewSummary / growthSummary 키를 항상 채움
    """
    raw = raw.strip()
    # 아주 단순한 섹션 분리 휴리스틱
    mo = ""
    gr = ""

    lower = raw.lower()
    # 헤더 기반 분리
    if "market overview" in lower and "growth" in lower:
        # Market Overview ~ Growth 사이
        i = lower.find("market overview")
        j = lower.find("growth")
        mo = raw[i:j].strip()
        gr = raw[j:].strip()
    else:
        # 구분 실패하면 전체를 market에 넣고 growth는 빈 문장
        mo = raw
        gr = ""

    # 마크다운/불릿 제거: 라인 단위로 불릿/헤더 제거 후 문장 합치기
    def _cleanup(s: str) -> str:
        lines = []
        for ln in s.splitlines():
            t = ln.strip()
            if not t:
                continue
            if t.startswith("#"):
                continue
            if t.startswith("- "):
                t = t[2:].strip()
            # 굵게/코드 등 단순 제거
            t = t.replace("**", "").replace("`", "")
            lines.append(t)
        # 단락형으로 합치기
        out = " ".join(lines)
        out = " ".join(out.split())
        return out

    mo = _cleanup(mo)
    gr = _cleanup(gr)

    # 성장 요약이 비어있으면 안내 문구
    if not gr:
        gr = "연도별 성장 데이터가 충분하지 않아 성장 추세를 해석하기 어렵습니다."

    return {"marketOverviewSummary": mo, "growthSummary": gr}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="학습된 모델로 커머스 시장 리포트 요약 생성"
    )
    ap.add_argument("--model_dir", default=None, help="학습된 LoRA 모델 디렉토리 (--template_only일 때는 선택사항)")
    ap.add_argument("--category_id", type=int, default=None, help="카테고리 ID")
    ap.add_argument("--category_contains", default=None, help="카테고리 경로에 포함된 문자열")
    ap.add_argument("--template_only", action="store_true", help="템플릿 기반 생성 (모델 없이)")
    ap.add_argument("--max_length", type=int, default=4096, help="최대 생성 길이")
    args = ap.parse_args()

    if not args.category_id and not args.category_contains:
        ap.error("--category_id 또는 --category_contains 중 하나는 필요합니다.")
    
    if not args.template_only and not args.model_dir:
        ap.error("--template_only가 아닐 때는 --model_dir이 필요합니다.")

    # 데이터 로드
    try:
        df_products, df_reviews, category_id, category_path = load_category_from_db(
            category_id=args.category_id,
            category_contains=args.category_contains,
        )
    except Exception as e:
        print(f"[ERROR] DB 조회 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 메트릭스 계산
    metrics = aggregate_category_with_reviews(
        df_products,
        category_id,
        category_path,
        df_reviews=df_reviews if not df_reviews.empty else None,
    )

    # 연도별 성장률 계산
    # Danawa에는 매출/판매량이 없으므로 '리뷰 수 -> (가상)매출/판매량'으로 환산한 성장률을 사용
    price_ref = metrics.price_p50 or metrics.price_p25 or metrics.price_p75
    growth_metrics = (
        calculate_yearly_growth(df_reviews, price_ref=price_ref, units_per_review=3.0)
        if not df_reviews.empty
        else []
    )

    # 템플릿 모드
    if args.template_only:
        from report_summary_lib import build_market_overview_summary, build_growth_summary

        market_overview = build_market_overview_summary(metrics, growth_metrics)
        growth_summary = build_growth_summary(growth_metrics)

        result = {
            "categoryId": category_id,
            "categoryName": metrics.category_name,
            "marketOverviewSummary": market_overview,
            "growthSummary": growth_summary,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 모델 모드
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60)

    # 메타데이터 로드
    metadata_path = os.path.join(args.model_dir, "training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            base_model = metadata.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    else:
        base_model = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Base model: {base_model}")
    print(f"LoRA adapter: {args.model_dir}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # device_map="auto"는 일부 Windows/드라이버 환경에서 크래시(출력 없는 종료)를 유발할 수 있어
        # 추론에서는 단순하게 명시적으로 cuda로 올리는 방식이 더 안정적임.
        if device == "cuda":
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            base_model_obj.to("cuda")
        else:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        model = PeftModel.from_pretrained(base_model_obj, args.model_dir)
        model.eval()
        print("모델 로딩 완료\n")
    except Exception as e:
        print(f"[ERROR] 모델 로딩 실패: {e}", file=sys.stderr)
        raise

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

    # 시스템 프롬프트 (학습 시와 동일)
    system_prompt = (
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
        "출력 형식(중요): "
        "반드시 JSON 객체 1개만 출력하라. 마크다운, 제목, 불릿, 설명 문장을 절대 출력하지 마라. "
        "반드시 아래 키 2개를 포함해야 한다: "
        "marketOverviewSummary (문자열), growthSummary (문자열). "
        "예시는 다음과 같다: "
        '{"marketOverviewSummary":"...","growthSummary":"..."}'
    )

    # 프롬프트 구성
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 생성
    print("=" * 60)
    print("리포트 요약 생성 중...")
    print("=" * 60)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_length,
            # JSON 스키마 안정성을 위해 최대한 결정적으로 생성
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = generated_text[len(prompt_text) :].strip()

    # JSON 추출
    result_json = _extract_json(answer)
    if not result_json:
        # JSON을 못 지키면 출력 텍스트를 최소한으로 복구해 JSON으로 반환
        result_json = _coerce_to_result_json(answer)

    result_json["categoryId"] = category_id
    result_json["categoryName"] = metrics.category_name
    print(json.dumps(result_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
