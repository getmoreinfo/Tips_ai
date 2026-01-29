"""
26_evaluate_report_summary.py
역할: 학습된 LoRA 모델의 리포트 요약 품질을 평가

- 템플릿 기반 출력 vs 모델 출력 비교
- 여러 카테고리에 대해 일괄 평가
- 품질 지표 계산 (JSON 형식 준수, 문장 길이, 키워드 포함 등)

사용 예:
  python 26_evaluate_report_summary.py --model_dir results_report/qwen2.5-3b-lora-report-summary
  python 26_evaluate_report_summary.py --model_dir results_report/qwen2.5-3b-lora-report-summary --categories 유모차 "그림/동화/놀이책"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_report_bullets_lib import aggregate_category_with_reviews
from db_category_loader import load_category_from_db
from report_summary_lib import (
    build_growth_summary,
    build_market_overview_summary,
    calculate_yearly_growth,
)

# 프록시 설정 제거
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


@dataclass
class QualityMetrics:
    """품질 지표"""
    json_valid: bool  # JSON 형식 준수 여부
    has_market_overview: bool  # marketOverviewSummary 키 존재
    has_growth_summary: bool  # growthSummary 키 존재
    market_overview_length: int  # 시장 구조 요약 문자 수
    growth_summary_length: int  # 성장 요약 문자 수
    market_overview_sentences: int  # 시장 구조 요약 문장 수
    growth_summary_sentences: int  # 성장 요약 문장 수
    contains_proxy_disclaimer: bool  # "대리 지표" 또는 "proxy" 언급 여부
    contains_key_numbers: int  # 핵심 수치(%, 점수 등) 포함 개수
    template_similarity: Optional[float] = None  # 템플릿과의 유사도 (간단한 키워드 기반)


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


def _count_sentences(text: str) -> int:
    """문장 수 계산 (간단한 휴리스틱)"""
    if not text:
        return 0
    # 마침표, 느낌표, 물음표로 문장 구분
    sentences = [s.strip() for s in text.replace("。", ".").split(".") if s.strip()]
    sentences += [s.strip() for s in text.split("!") if s.strip()]
    sentences += [s.strip() for s in text.split("?") if s.strip()]
    return max(len(sentences), 1)  # 최소 1개


def _count_key_numbers(text: str) -> int:
    """핵심 수치 개수 계산 (% 포함, 소수점 포함 숫자 등)"""
    import re
    # % 포함 숫자, 소수점 포함 숫자 (예: 4.7점, 45.5%)
    patterns = [
        r'\d+\.\d+%',  # 45.5%
        r'\d+%',  # 45%
        r'\d+\.\d+점',  # 4.7점
        r'\d+\.\d+',  # 831.5 같은 숫자
    ]
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text)
        count += len(matches)
    return count


def evaluate_quality(model_output: Dict[str, Any], template_output: Dict[str, Any]) -> QualityMetrics:
    """모델 출력의 품질 지표 계산"""
    json_valid = isinstance(model_output, dict) and "marketOverviewSummary" in model_output and "growthSummary" in model_output
    
    market_overview = model_output.get("marketOverviewSummary", "")
    growth_summary = model_output.get("growthSummary", "")
    
    # 템플릿 출력과 비교
    template_mo = template_output.get("marketOverviewSummary", "")
    template_gr = template_output.get("growthSummary", "")
    
    # 간단한 키워드 기반 유사도 (공통 단어 비율)
    def _simple_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0
    
    mo_sim = _simple_similarity(market_overview, template_mo)
    gr_sim = _simple_similarity(growth_summary, template_gr)
    avg_sim = (mo_sim + gr_sim) / 2 if (mo_sim > 0 or gr_sim > 0) else None
    
    return QualityMetrics(
        json_valid=json_valid,
        has_market_overview=bool(market_overview),
        has_growth_summary=bool(growth_summary),
        market_overview_length=len(market_overview),
        growth_summary_length=len(growth_summary),
        market_overview_sentences=_count_sentences(market_overview),
        growth_summary_sentences=_count_sentences(growth_summary),
        contains_proxy_disclaimer=("대리 지표" in market_overview or "대리 지표" in growth_summary or "proxy" in market_overview.lower() or "proxy" in growth_summary.lower()),
        contains_key_numbers=_count_key_numbers(market_overview) + _count_key_numbers(growth_summary),
        template_similarity=avg_sim,
    )


def generate_with_model(
    model,
    tokenizer,
    device: str,
    user_input: Dict[str, Any],
    system_prompt: str,
    max_length: int = 4096,
) -> Dict[str, Any]:
    """모델로 리포트 요약 생성"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = generated_text[len(prompt_text) :].strip()
    
    result_json = _extract_json(answer)
    if not result_json:
        # JSON 파싱 실패 시 최소한의 구조로 복구
        result_json = {
            "marketOverviewSummary": answer[:500] if answer else "",
            "growthSummary": answer[500:] if len(answer) > 500 else "",
        }
    
    return result_json


def evaluate_category(
    category_id: Optional[int],
    category_contains: Optional[str],
    model,
    tokenizer,
    device: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """단일 카테고리에 대해 템플릿 vs 모델 비교 평가"""
    # 데이터 로드
    try:
        df_products, df_reviews, cat_id, category_path = load_category_from_db(
            category_id=category_id,
            category_contains=category_contains,
        )
    except Exception as e:
        return {"error": str(e)}
    
    # 메트릭스 계산
    metrics = aggregate_category_with_reviews(
        df_products,
        cat_id,
        category_path,
        df_reviews=df_reviews if not df_reviews.empty else None,
    )
    
    price_ref = metrics.price_p50 or metrics.price_p25 or metrics.price_p75
    growth_metrics = (
        calculate_yearly_growth(df_reviews, price_ref=price_ref, units_per_review=3.0)
        if not df_reviews.empty
        else []
    )
    
    # 템플릿 기반 생성
    template_mo = build_market_overview_summary(metrics, growth_metrics)
    template_gr = build_growth_summary(growth_metrics)
    template_output = {
        "marketOverviewSummary": template_mo,
        "growthSummary": template_gr,
    }
    
    # 모델 기반 생성
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
    
    model_output = generate_with_model(model, tokenizer, device, user_input, system_prompt)
    
    # 품질 평가
    quality = evaluate_quality(model_output, template_output)
    
    return {
        "categoryId": cat_id,
        "categoryName": metrics.category_name,
        "template": template_output,
        "model": model_output,
        "quality": {
            "json_valid": quality.json_valid,
            "has_market_overview": quality.has_market_overview,
            "has_growth_summary": quality.has_growth_summary,
            "market_overview_length": quality.market_overview_length,
            "growth_summary_length": quality.growth_summary_length,
            "market_overview_sentences": quality.market_overview_sentences,
            "growth_summary_sentences": quality.growth_summary_sentences,
            "contains_proxy_disclaimer": quality.contains_proxy_disclaimer,
            "contains_key_numbers": quality.contains_key_numbers,
            "template_similarity": quality.template_similarity,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="리포트 요약 모델 품질 평가")
    ap.add_argument("--model_dir", required=True, help="학습된 LoRA 모델 디렉토리")
    ap.add_argument(
        "--categories",
        nargs="+",
        default=["유모차", "그림/동화/놀이책"],
        help="평가할 카테고리 목록 (category_contains로 검색)",
    )
    ap.add_argument("--max_length", type=int, default=4096, help="최대 생성 길이")
    ap.add_argument("--out_json", default="evaluation_report_summary.json", help="평가 결과 저장 경로")
    args = ap.parse_args()
    
    print("=" * 60)
    print("리포트 요약 모델 품질 평가")
    print("=" * 60)
    print(f"모델 디렉토리: {args.model_dir}")
    print(f"평가 카테고리: {args.categories}")
    print()
    
    # 모델 로드
    print("모델 로딩 중...")
    metadata_path = os.path.join(args.model_dir, "training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            base_model = metadata.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    else:
        base_model = "Qwen/Qwen2.5-3B-Instruct"
    
    print(f"Base model: {base_model}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 토크나이저 로드: 먼저 모델 디렉토리에서 시도, 없으면 베이스 모델에서 로드
    tokenizer_path = args.model_dir
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        tokenizer_path = base_model
        print(f"토크나이저를 베이스 모델에서 로드: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
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
        
        # 절대 경로로 변환하여 로컬 파일임을 명확히 함
        model_dir_abs = os.path.abspath(args.model_dir)
        if not os.path.exists(model_dir_abs):
            raise ValueError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir_abs}")
        
        # adapter_config.json 파일 존재 확인
        adapter_config_path = os.path.join(model_dir_abs, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise ValueError(
                f"LoRA 어댑터 설정 파일을 찾을 수 없습니다: {adapter_config_path}\n"
                f"모델이 제대로 학습되었는지 확인하세요."
            )
        
        model = PeftModel.from_pretrained(base_model_obj, model_dir_abs, local_files_only=True)
        model.eval()
        print("모델 로딩 완료\n")
    except Exception as e:
        print(f"[ERROR] 모델 로딩 실패: {e}", file=sys.stderr)
        raise
    
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
    
    # 각 카테고리 평가
    results = []
    for cat_name in args.categories:
        print(f"\n{'=' * 60}")
        print(f"카테고리 평가: {cat_name}")
        print("=" * 60)
        
        result = evaluate_category(
            category_id=None,
            category_contains=cat_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            system_prompt=system_prompt,
        )
        
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            continue
        
        results.append(result)
        
        # 간단한 요약 출력
        q = result["quality"]
        print(f"JSON 형식 준수: {q['json_valid']}")
        print(f"시장 구조 요약 길이: {q['market_overview_length']}자 ({q['market_overview_sentences']}문장)")
        print(f"성장 요약 길이: {q['growth_summary_length']}자 ({q['growth_summary_sentences']}문장)")
        print(f"대리 지표 언급: {q['contains_proxy_disclaimer']}")
        print(f"핵심 수치 포함: {q['contains_key_numbers']}개")
        if q["template_similarity"] is not None:
            print(f"템플릿 유사도: {q['template_similarity']:.2%}")
    
    # 전체 통계
    print("\n" + "=" * 60)
    print("전체 통계")
    print("=" * 60)
    
    if results:
        json_valid_rate = sum(1 for r in results if r["quality"]["json_valid"]) / len(results)
        proxy_disclaimer_rate = sum(1 for r in results if r["quality"]["contains_proxy_disclaimer"]) / len(results)
        avg_mo_length = sum(r["quality"]["market_overview_length"] for r in results) / len(results)
        avg_gr_length = sum(r["quality"]["growth_summary_length"] for r in results) / len(results)
        avg_key_numbers = sum(r["quality"]["contains_key_numbers"] for r in results) / len(results)
        avg_similarity = sum(
            r["quality"]["template_similarity"] for r in results if r["quality"]["template_similarity"] is not None
        ) / max(1, sum(1 for r in results if r["quality"]["template_similarity"] is not None))
        
        print(f"평가 카테고리 수: {len(results)}개")
        print(f"JSON 형식 준수율: {json_valid_rate:.1%}")
        print(f"대리 지표 언급율: {proxy_disclaimer_rate:.1%}")
        print(f"평균 시장 구조 요약 길이: {avg_mo_length:.0f}자")
        print(f"평균 성장 요약 길이: {avg_gr_length:.0f}자")
        print(f"평균 핵심 수치 포함: {avg_key_numbers:.1f}개")
        print(f"평균 템플릿 유사도: {avg_similarity:.2%}")
    
    # 결과 저장
    output = {
        "model_dir": args.model_dir,
        "base_model": base_model,
        "categories": args.categories,
        "results": results,
        "summary": {
            "total_categories": len(results),
            "json_valid_rate": json_valid_rate if results else 0.0,
            "proxy_disclaimer_rate": proxy_disclaimer_rate if results else 0.0,
            "avg_market_overview_length": avg_mo_length if results else 0.0,
            "avg_growth_summary_length": avg_gr_length if results else 0.0,
            "avg_key_numbers": avg_key_numbers if results else 0.0,
            "avg_template_similarity": avg_similarity if results else 0.0,
        },
    }
    
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n평가 결과 저장: {args.out_json}")


if __name__ == "__main__":
    main()
