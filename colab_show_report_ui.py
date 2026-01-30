"""
Colab에서 학습된 모델 + 학습 데이터로 리포트를 생성하고, HTML UI로 노트북에 표시합니다.

=== Colab 노트북에서 실행 순서 ===

[셀 1] 경로 설정 (필요 시 수정)
  import os
  os.environ["COLAB_REPO_ROOT"] = "/content/ai-tr"   # clone한 폴더
  os.environ["COLAB_MODEL_DIR"] = "results_report/qwen2.5-7b-lora-report-summary"
  os.environ["COLAB_PRODUCTS_CSV"] = "products_all.csv"
  os.environ["COLAB_REVIEWS_CSV"] = "reviews_all.csv"
  os.environ["COLAB_CATEGORY"] = "아기띠"   # 또는 "그림/동화/놀이책"

[셀 2] UI 실행 (22→23→24 완료 후, 같은 노트북 또는 clone된 폴더에서)
  %run colab_show_report_ui.py

=> 위 셀 출력에 리포트 HTML(차트·요약)이 표시됩니다.

실행 전: 22→23→24까지 한 번 실행해 두고, MODEL_DIR·CSV 경로가 맞아야 합니다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# ========== Colab에서 수정할 설정 ==========
# Colab에서 clone한 루트(예: /content/ai-tr)로 맞춰 주세요.
REPO_ROOT = os.environ.get("COLAB_REPO_ROOT", os.getcwd())
MODEL_DIR = os.environ.get("COLAB_MODEL_DIR", "results_report/qwen2.5-7b-lora-report-summary")
PRODUCTS_CSV = os.environ.get("COLAB_PRODUCTS_CSV", "products_all.csv")
REVIEWS_CSV = os.environ.get("COLAB_REVIEWS_CSV", "reviews_all.csv")
CATEGORY = os.environ.get("COLAB_CATEGORY", "아기띠")

# 25번 결과(차트+템플릿 요약) / 24번 결과(모델 요약) 임시 파일
REPORT_FULL_JSON = "/tmp/colab_report_full.json"
SUMMARY_MODEL_JSON = "/tmp/colab_summary_model.json"


def _run(cmd: list[str], cwd: str) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"실패: {' '.join(cmd)}\n{r.stderr or r.stdout}")
    return r.stdout or ""


def _build_html(report: dict) -> str:
    """리포트 JSON을 pm_demo 스타일 HTML 문자열로 만듦."""
    report_js = json.dumps(report, ensure_ascii=False)
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI-TR · Colab 리포트</title>
  <style>
    :root { --bg:#f6f8fa; --panel:#fff; --border:#d8dee4; --text:#1f2328; --muted:#656d76; --accent:#0969da; --accent-bg:#ddf4ff; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", "Noto Sans KR", sans-serif; line-height: 1.5; }
    .container { max-width: 960px; margin: 0 auto; padding: 24px 20px 48px; }
    header { background: var(--panel); border-bottom: 1px solid var(--border); padding: 20px 24px; }
    header h1 { margin: 0 0 6px; font-size: 1.5rem; font-weight: 700; }
    header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
    section { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; margin-top: 20px; }
    section h2 { margin: 0 0 14px; font-size: 1.1rem; font-weight: 600; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
    .summary-text { font-size: 0.9rem; line-height: 1.65; white-space: pre-wrap; margin: 10px 0 0; }
    .meta { font-size: 0.8rem; color: var(--muted); margin-top: 8px; }
    .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
    .chart-wrap { background: var(--bg); border-radius: 8px; padding: 16px; }
    .chart-wrap h3 { margin: 0 0 12px; font-size: 0.85rem; color: var(--muted); font-weight: 600; }
    canvas { width: 100%; height: 240px; display: block; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 12px; margin-top: 10px; font-size: 0.8rem; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .legend .dot { width: 8px; height: 8px; border-radius: 50%; }
    .notice { background: var(--accent-bg); border: 1px solid rgba(9,105,218,0.3); border-radius: 8px; padding: 12px 14px; margin-top: 16px; font-size: 0.9rem; }
  </style>
</head>
<body>
  <header><div class="container"><h1>AI-TR: 카테고리 리포트 (Colab)</h1><p>학습 모델 + 학습 데이터로 생성 · Qwen2.5-7B LoRA</p></div></header>
  <main class="container">
    <section>
      <h2>리포트: <span id="catName"></span></h2>
      <p class="meta">아래는 24번(모델 요약) + 25번(차트) 결과를 합쳐 표시한 것입니다. 리뷰 수를 대리 지표(proxy)로 사용한 상대적 구조 해석 결과입니다.</p>
      <div class="chart-row">
        <div class="chart-wrap"><h3>Top10 브랜드 (리뷰 기반 점유율)</h3><canvas id="donut"></canvas><div id="donutLegend" class="legend"></div></div>
        <div class="chart-wrap"><h3>성장 추세 (가상 매출)</h3><canvas id="line"></canvas><div id="lineMeta" class="meta"></div></div>
      </div>
      <h3 style="margin:18px 0 8px;font-size:0.95rem;">시장 개요 요약</h3><div id="marketSummary" class="summary-text"></div>
      <h3 style="margin:18px 0 8px;font-size:0.95rem;">성장 요약</h3><div id="growthSummary" class="summary-text"></div>
      <div class="notice">※ 본 리포트의 매출·점유·성장 수치는 실제 거래 데이터가 아니라, 리뷰 수를 대리 지표(proxy)로 활용한 상대적 구조 해석 결과입니다.</div>
    </section>
  </main>
  <script>
    const PALETTE = ["#0969da","#cf222e","#1a7f37","#9a6700","#8250df","#c24c00","#0e8a16","#fb8500","#656d76","#ff6b6b"];
    const report = """ + report_js + """;
    document.getElementById("catName").textContent = report.categoryName || "-";
    function fmtNum(n){ if(n==null||Number.isNaN(n)) return "-"; return new Intl.NumberFormat("ko-KR").format(n); }
    function fmtPct(x){ if(x==null||Number.isNaN(x)) return "-"; return (x*100).toFixed(1)+"%"; }
    function setupCanvas(canvas){ const dpr=Math.max(1,window.devicePixelRatio||1); const rect=canvas.getBoundingClientRect(); canvas.width=Math.floor(rect.width*dpr); canvas.height=Math.floor(rect.height*dpr); const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr); return {ctx,w:rect.width,h:rect.height}; }
    function drawDonut(canvas, items){
      const {ctx,w,h}=setupCanvas(canvas); ctx.clearRect(0,0,w,h);
      const cx=w/2,cy=h/2,rOut=Math.min(w,h)*0.38,rIn=rOut*0.6;
      const total=items.reduce((s,it)=>s+(it.value||0),0)||1; let a=-Math.PI/2;
      ctx.beginPath(); ctx.arc(cx,cy,rOut,0,Math.PI*2); ctx.arc(cx,cy,rIn,0,Math.PI*2,true); ctx.fillStyle="#eaeef2"; ctx.fill("evenodd");
      items.forEach((it,i)=>{ const frac=(it.value||0)/total, da=frac*Math.PI*2; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,rOut,a,a+da); ctx.closePath(); ctx.arc(cx,cy,rIn,a+da,a,true); ctx.fillStyle=PALETTE[i%PALETTE.length]; ctx.fill("evenodd"); a+=da; });
      ctx.fillStyle="#1f2328"; ctx.font="600 14px system-ui,sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("TOP10",cx,cy-6);
      ctx.fillStyle="#656d76"; ctx.font="12px system-ui,sans-serif"; ctx.fillText("브랜드 점유",cx,cy+10);
    }
    function drawLine(canvas, points){
      const {ctx,w,h}=setupCanvas(canvas); ctx.clearRect(0,0,w,h);
      if(!points||points.length===0){ ctx.fillStyle="#656d76"; ctx.font="13px system-ui,sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText("성장 데이터 없음",w/2,h/2); return; }
      const padL=36,padR=12,padT=12,padB=28,x0=padL,x1=w-padR,y0=h-padB,y1=padT;
      const ys=points.map(p=>Number(p.estimatedRevenue||0)), minY=Math.min(...ys,0), maxY=Math.max(...ys,1), span=Math.max(1,maxY-minY), n=points.length, maxX=Math.max(1,n-1);
      function pxX(i){ return x0+(x1-x0)*i/maxX; }
      function pxY(v){ return y0-(y0-y1)*(v-minY)/span; }
      ctx.strokeStyle="#d8dee4"; ctx.lineWidth=1;
      for(let i=0;i<=4;i++){ const y=y1+(y0-y1)*i/4; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke(); }
      ctx.strokeStyle=PALETTE[0]; ctx.lineWidth=2.5; ctx.beginPath();
      points.forEach((p,i)=>{ const x=pxX(i), y=pxY(Number(p.estimatedRevenue||0)); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
      ctx.fillStyle="#fff"; points.forEach((p,i)=>{ const x=pxX(i), y=pxY(Number(p.estimatedRevenue||0)); ctx.beginPath(); ctx.arc(x,y,3.5,0,Math.PI*2); ctx.fill(); ctx.strokeStyle=PALETTE[0]; ctx.lineWidth=2; ctx.stroke(); });
      ctx.fillStyle="#656d76"; ctx.font="11px system-ui,sans-serif"; ctx.textAlign="center"; ctx.textBaseline="top";
      points.forEach((p,i)=>{ ctx.fillText(String(p.year),pxX(i),y0+6); });
    }
    const items = report.charts && report.charts.brandTop10Donut && report.charts.brandTop10Donut.items ? report.charts.brandTop10Donut.items : [];
    drawDonut(document.getElementById("donut"), items);
    const leg = document.getElementById("donutLegend"); leg.innerHTML = "";
    items.forEach((it,i)=>{ const s=document.createElement("span"); s.innerHTML='<span class="dot" style="background:'+PALETTE[i%PALETTE.length]+'"></span> '+it.name+' '+fmtNum(it.value)+' ('+fmtPct(it.share)+')'; leg.appendChild(s); });
    const points = report.charts && report.charts.growthLine && report.charts.growthLine.points ? report.charts.growthLine.points : [];
    drawLine(document.getElementById("line"), points);
    const last = points[points.length-1];
    document.getElementById("lineMeta").textContent = last ? "최근: "+last.year+"년 · 추정 매출 "+fmtNum(Math.round(last.estimatedRevenue))+" · YoY "+(last.growthRate!=null?last.growthRate.toFixed(1)+"%":"-") : "성장 데이터 없음.";
    const summaries = report.summaries || {};
    document.getElementById("marketSummary").textContent = summaries.marketOverviewSummary || "-";
    document.getElementById("growthSummary").textContent = summaries.growthSummary || "-";
  </script>
</body>
</html>"""


def main() -> None:
    os.chdir(REPO_ROOT)
    model_dir_abs = os.path.abspath(MODEL_DIR)
    products_abs = os.path.abspath(PRODUCTS_CSV)
    reviews_abs = os.path.abspath(REVIEWS_CSV)

    if not os.path.isdir(model_dir_abs):
        print(f"[경고] 모델 디렉터리가 없습니다: {model_dir_abs}")
        print("  먼저 23번 LoRA 학습을 실행한 뒤, MODEL_DIR을 맞춰 주세요.")
    if not os.path.isfile(products_abs):
        print(f"[경고] 상품 CSV가 없습니다: {products_abs}")
    if not os.path.isfile(reviews_abs):
        print(f"[경고] 리뷰 CSV가 없습니다: {reviews_abs}")

    # 1) 25번: 차트 + 템플릿 요약으로 전체 리포트 JSON 생성
    print("25번 실행 중 (차트·템플릿 요약)...")
    _run([
        sys.executable,
        "25_generate_category_report_from_csv.py",
        "--products_csv", products_abs,
        "--reviews_csv", reviews_abs,
        "--category_contains", CATEGORY,
        "--out_json", REPORT_FULL_JSON,
    ], REPO_ROOT)

    # 2) 24번: 학습된 모델로 요약만 생성 (--out_json)
    print("24번 실행 중 (모델 요약)...")
    _run([
        sys.executable,
        "24_generate_report_summary.py",
        "--model_dir", model_dir_abs,
        "--products_csv", products_abs,
        "--reviews_csv", reviews_abs,
        "--category_contains", CATEGORY,
        "--out_json", SUMMARY_MODEL_JSON,
    ], REPO_ROOT)

    # 3) 합치기: 25 결과에 24 요약 덮어쓰기
    with open(REPORT_FULL_JSON, "r", encoding="utf-8") as f:
        report_full = json.load(f)
    with open(SUMMARY_MODEL_JSON, "r", encoding="utf-8") as f:
        summary_model = json.load(f)
    report_full.setdefault("summaries", {})["marketOverviewSummary"] = summary_model.get("marketOverviewSummary", "")
    report_full.setdefault("summaries", {})["growthSummary"] = summary_model.get("growthSummary", "")

    # 4) HTML로 노트북에 표시
    html = _build_html(report_full)
    try:
        from IPython.display import HTML, display
        display(HTML(html))
        print("HTML UI를 위 셀 출력에 표시했습니다.")
    except ImportError:
        with open("/tmp/colab_report_ui.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("IPython이 없어 파일로 저장했습니다: /tmp/colab_report_ui.html")
        print("Colab 노트북에서는 위 셀을 실행한 뒤, 다음을 한 셀에 넣어 실행하세요:")
        print("  from IPython.display import HTML, display")
        print("  with open('/tmp/colab_report_ui.html') as f: display(HTML(f.read()))")


if __name__ == "__main__":
    main()
