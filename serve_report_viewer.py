from __future__ import annotations

import argparse
import os
import socketserver
import sys
from http.server import SimpleHTTPRequestHandler


def main() -> None:
    ap = argparse.ArgumentParser(description="report_viewer.html 로컬 서버")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # 워크스페이스 루트에서 실행되는 것을 전제로, 현재 폴더를 그대로 서빙한다.
    root = os.getcwd()
    print(f"[OK] serving: {root}")
    print(f"[OK] PM 데모(한 페이지): http://127.0.0.1:{args.port}/pm_demo.html")
    print(f"[OK] 리포트 뷰어: http://127.0.0.1:{args.port}/report_viewer.html?json=report_stroller.json")
    print("     (json 파라미터를 다른 파일로 바꾸면 됩니다)")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    try:
        with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
            httpd.serve_forever()
    except OSError as e:
        print(f"[ERROR] 서버 시작 실패: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

