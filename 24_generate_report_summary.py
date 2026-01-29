"""
24_generate_report_summary.py
역할: 학습된 LoRA 모델을 사용하여 커머스 시장 리포트 요약 생성

사용 예 (DB):
  python 24_generate_report_summary.py --model_dir results_report/qwen2.5-3b-lora-report-summary --category_contains 유모차
  python 24_generate_report_summary.py --model_dir results_report/qwen2.5-3b-lora-report-summary --category_id 4

사용 예 (CSV):
  python 24_generate_report_summary.py --model_dir .../qwen2.5-7b-lora-report-summary --products_csv products_all.csv --reviews_csv reviews_all.csv --category_contains 유모차
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_report_bullets_lib import aggregate_category_with_reviews
from ai_report_bullets_lib import _extract_keywords_from_review_texts  # type: ignore
from report_summary_lib import calculate_yearly_growth


def _pick_category_from_df(
    df_products: pd.DataFrame,
    category_id: Optional[int],
    category_contains: Optional[str],
) -> Tuple[int, str, pd.DataFrame]:
    """CSV 기반 카테고리 선택 (25와 동일 로직)."""
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

    needle = str(category_contains)
    g = df[df["category"].astype(str).str.contains(needle, na=False)].copy()
    if g.empty:
        raise ValueError(f"category_contains='{needle}'에 해당하는 상품이 없습니다.")
    best = None
    for cid, gg in g.groupby("category_id", sort=False):
        if best is None or len(gg) > len(best[2]):
            cp = str(gg["category"].dropna().iloc[0]) if gg["category"].notna().any() else ""
            best = (int(cid), cp, gg.copy())
    assert best is not None
    return best

# 프록시 설정 제거
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """텍스트에서 첫 번째 JSON 객체 추출"""
    text = text.strip()
    # Qwen 계열이 가끔 끝 토큰을 그대로 출력
    text = text.replace("<|im_end|>", "").strip()
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


def _looks_like_json_object_string(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    ss = s.strip().replace("<|im_end|>", "").strip()
    while ss.endswith('"') or ss.endswith("\\"):
        ss = ss[:-1].strip()
    return ss.startswith("{") and (
        ss.endswith("}") or "\"marketOverviewSummary\"" in ss or "\"growthSummary\"" in ss
    )


def _clean_summary_text(s: str, max_sentences: int = 12) -> str:
    """
    모델이 출력한 요약 문자열 정리:
    - <|im_end|> 제거, 그 이후 버림
    - 중국어 블록 제거 또는 그 이전까지만 유지
    - 첫 번째 한국어 요약 블록 우선 사용, 중복 문장 정리
    """
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    # 토큰 및 그 이후 버리기
    if "<|im_end|>" in s:
        s = s.split("<|im_end|>")[0].strip()
    s = s.replace("<|im_end|>", "").strip()

    # 중국어 등장 지점에서 잘라서 앞쪽(한국어)만 사용
    for sep in ("与你提供的", "对应的", "分析如下", "市场概况", "增长概况", "群与"):
        if sep in s:
            idx = s.find(sep)
            if idx > 50:
                s = s[:idx].strip()
            else:
                s = s[idx + len(sep) :].strip()
                for start in ("본 ", "그림", "전체", "리뷰", "이 ", "연도", "대리"):
                    j = s.find(start)
                    if j != -1:
                        s = s[j:]
                        break
            break

    # "{\"marketOverviewSummary\":\"한국어...群중국어" 형태: 첫 번째 값만 추출 (파싱 불가 시)
    if s.strip().startswith("{") and ("marketOverviewSummary" in s or "growthSummary" in s):
        key = "marketOverviewSummary"
        for key_pattern in (f'"{key}"', f'\\"{key}\\"'):
            pos = s.find(key_pattern)
            if pos != -1:
                break
        if pos != -1:
            # 값 시작: ": " 또는 \":\" 다음의 따옴표 뒤 (JSON 파싱된 문자열은 \" 형태)
            after_key = pos + len(key_pattern)
            val_start = -1
            for q in ('\\":\\"', '": "', '":"', ':\\"'):
                i = s.find(q, after_key)
                if i != -1:
                    val_start = i + len(q)
                    break
            if val_start > 0:
                end_markers = ["群", "与", "与你", "\", \"growthSummary\"", "\\\", \\\"growthSummary\\\"", "\", \"marketOverviewSummary\""]
                val_end = len(s)
                for m in end_markers:
                    i = s.find(m, val_start)
                    if i != -1:
                        val_end = min(val_end, i)
                chunk = s[val_start:val_end].strip()
                if len(chunk) > 30 and any("\uac00" <= c <= "\ud7a3" for c in chunk):
                    s = chunk
        if s.strip().startswith("{"):
            try:
                start = s.find("{")
                depth = 0
                for i in range(start, len(s)):
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                        if depth == 0:
                            obj = json.loads(s[start : i + 1])
                            if isinstance(obj.get("marketOverviewSummary"), str):
                                s = obj["marketOverviewSummary"]
                            elif isinstance(obj.get("growthSummary"), str):
                                s = obj["growthSummary"]
                            break
            except Exception:
                pass

    # 문장 단위로 나누어 중복·과도한 반복 제거 후 재결합 (숫자 내 . 은 유지)
    raw_sentences = re.split(r"(?<=[가-힣a-zA-Z])\s*[.。]\s+", s)
    raw_sentences = [t.strip() for t in raw_sentences if t.strip()]
    sentences = []
    seen = set()
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        key = sent[:50]
        if key in seen:
            continue
        seen.add(key)
        sentences.append(sent)
        if len(sentences) >= max_sentences:
            break

    out = ". ".join(sentences)
    if out and not out.endswith("."):
        out = out + "."
    return " ".join(out.split())


def _normalize_result_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    모델 출력이 깨지는 케이스를 최대한 복구:
    - marketOverviewSummary 값이 JSON 문자열(\"{...}\") 형태로 다중 중첩
    - 문자열 안에 <|im_end|>, 중국어 혼입
    """
    out: Dict[str, Any] = dict(obj)

    # 1) 다중 중첩 JSON 문자열 풀기 (최대 3단계), 끝의 <|im_end|>·따옴표 제거 후 파싱 시도
    for _ in range(3):
        changed = False
        for k in ("marketOverviewSummary", "growthSummary"):
            v = out.get(k)
            if not isinstance(v, str):
                continue
            v_strip = v.strip().replace("<|im_end|>", "").strip()
            while v_strip.endswith('"') or v_strip.endswith("\\"):
                v_strip = v_strip[:-1].strip()
            if _looks_like_json_object_string(v_strip):
                try:
                    nested = json.loads(v_strip)
                    if isinstance(nested, dict):
                        for nk in ("marketOverviewSummary", "growthSummary"):
                            if nk in nested and isinstance(nested[nk], str):
                                out[nk] = nested[nk]
                                changed = True
                        break
                except Exception:
                    pass
        if not changed:
            break

    # 2) 문자열 정리: 토큰 제거, 중국어 구간 제거, 중복 문장 정리
    for k in ("marketOverviewSummary", "growthSummary"):
        v = out.get(k)
        if isinstance(v, str):
            out[k] = _clean_summary_text(v)

    # 3) 최소 키 보장
    if "marketOverviewSummary" not in out or not isinstance(out["marketOverviewSummary"], str):
        out["marketOverviewSummary"] = ""
    if "growthSummary" not in out or not isinstance(out["growthSummary"], str):
        out["growthSummary"] = ""

    return out

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


def _review_text_column(df: pd.DataFrame) -> Optional[str]:
    """리뷰 본문 컬럼명 찾기 (review_text 우선, 다른 이름 허용)."""
    for col in ("review_text", "review_content", "content", "text", "body"):
        if col in df.columns:
            return col
    return None


def _build_review_insights_for_inference(
    df_reviews: pd.DataFrame,
    *,
    top_keywords: int = 10,
    pos_examples: int = 3,
    neg_examples: int = 2,
    max_example_chars: int = 180,
) -> Optional[Dict[str, Any]]:
    if df_reviews is None or df_reviews.empty:
        return None
    text_col = _review_text_column(df_reviews)
    if text_col is None:
        return None

    def _trunc(s: Any) -> str:
        ss = str(s) if s is not None else ""
        ss = " ".join(ss.split())
        if max_example_chars and max_example_chars > 0 and len(ss) > max_example_chars:
            return ss[: max_example_chars - 1] + "…"
        return ss

    texts = df_reviews[text_col].dropna().astype(str).map(_trunc).tolist()
    texts = [t for t in texts if len(t.strip()) > 0]
    if not texts:
        return None

    kws = _extract_keywords_from_review_texts(texts, top_n=int(max(0, top_keywords)))

    rating = pd.to_numeric(df_reviews.get("rating"), errors="coerce")
    df_pos = df_reviews[rating.notna() & (rating >= 4)]
    df_neg = df_reviews[rating.notna() & (rating <= 2)]

    def _take_top(dff: pd.DataFrame, n: int) -> List[str]:
        if n <= 0 or dff.empty or text_col not in dff.columns:
            return []
        arr = dff[text_col].dropna().astype(str).map(_trunc).tolist()
        return [t for t in arr if t.strip()][:n]

    return {
        "topKeywords": list(kws),
        "positiveExamples": _take_top(df_pos, int(pos_examples)),
        "negativeExamples": _take_top(df_neg, int(neg_examples)),
        "hasReviewText": True,
        "reviewTextCount": int(len(texts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="학습된 모델로 커머스 시장 리포트 요약 생성"
    )
    ap.add_argument("--model_dir", default=None, help="학습된 LoRA 모델 디렉토리 (--template_only일 때는 선택사항)")
    ap.add_argument("--category_id", type=int, default=None, help="카테고리 ID")
    ap.add_argument("--category_contains", default=None, help="카테고리 경로에 포함된 문자열")
    ap.add_argument("--products_csv", default=None, help="상품 CSV (--reviews_csv와 함께 사용 시 DB 대신 CSV 사용)")
    ap.add_argument("--reviews_csv", default=None, help="리뷰 CSV (--products_csv와 함께 사용 시 DB 대신 CSV 사용)")
    ap.add_argument("--template_only", action="store_true", help="템플릿 기반 생성 (모델 없이)")
    ap.add_argument("--max_length", type=int, default=4096, help="최대 생성 길이")
    args = ap.parse_args()

    if not args.category_id and not args.category_contains:
        ap.error("--category_id 또는 --category_contains 중 하나는 필요합니다.")
    
    if not args.template_only and not args.model_dir:
        ap.error("--template_only가 아닐 때는 --model_dir이 필요합니다.")

    use_csv = bool(args.products_csv and args.reviews_csv)

    # 데이터 로드
    if use_csv:
        try:
            df_products = pd.read_csv(args.products_csv)
            df_reviews = pd.read_csv(args.reviews_csv)
            cat_id, category_path, df_cat = _pick_category_from_df(
                df_products,
                category_id=args.category_id,
                category_contains=args.category_contains,
            )
            product_ids = set(pd.to_numeric(df_cat["id"], errors="coerce").dropna().astype(int).tolist())
            prod_ids_series = pd.to_numeric(df_reviews["product_id"], errors="coerce")
            df_rev_cat = df_reviews[prod_ids_series.isin(product_ids)].copy()
            df_products = df_cat
            df_reviews = df_rev_cat
            category_id = cat_id
        except Exception as e:
            print(f"[ERROR] CSV 로드/카테고리 선택 실패: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            from db_category_loader import load_category_from_db

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

    # 토크나이저: adapter 디렉터리에는 vocab 등이 없을 수 있으므로 base_model(메타데이터)에서 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
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

    # 리뷰 텍스트 인사이트: 22(SFT)와 입력 구조를 맞춰 모델이 리뷰 근거를 활용하도록 함
    ri = _build_review_insights_for_inference(df_reviews)
    if ri:
        user_input["reviewInsights"] = ri

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
    else:
        result_json = _normalize_result_obj(result_json)

    result_json["categoryId"] = category_id
    result_json["categoryName"] = metrics.category_name
    print(json.dumps(result_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
