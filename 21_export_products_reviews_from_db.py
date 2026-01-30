"""
21_export_products_reviews_from_db.py

역할:
  - .env의 DB 설정(PGHOST, PGPORT, PGUSER, PGPASSWORD, PGSCHEMA, PGTABLE, PGREVIEWSTABLE)으로
    danawa_crawlingdb에 접속해,
  - 상품·리뷰 전체를 products_all.csv, reviews_all.csv 로 추출한다.

사용:
  python 21_export_products_reviews_from_db.py
  python 21_export_products_reviews_from_db.py --all          # 카테고리 필터 없이 전부
  python 21_export_products_reviews_from_db.py --out_dir ./data
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def _schema() -> str:
    return os.getenv("PGSCHEMA", "public1")


def _products_table() -> str:
    return os.getenv("PGTABLE", "products")


def _reviews_table() -> str:
    return os.getenv("PGREVIEWSTABLE", "reviews")


def _get_config() -> dict:
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    if not all([host, port, user, password]):
        raise RuntimeError(
            ".env에 PGHOST, PGPORT, PGUSER, PGPASSWORD가 필요합니다."
        )
    return {
        "host": host,
        "port": int(port) if str(port).isdigit() else 5432,
        "database": "danawa_crawlingdb",
        "user": user,
        "password": password,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DB에서 products_all.csv, reviews_all.csv 추출 (.env 사용)"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="카테고리 필터 없이 상품/리뷰 전부 추출 (기본: min_products 이상인 카테고리만)",
    )
    ap.add_argument(
        "--min_products",
        type=int,
        default=10,
        help="--all 미사용 시, 카테고리당 최소 상품 수 (기본 10)",
    )
    ap.add_argument(
        "--max_categories",
        type=int,
        default=10_000,
        help="--all 미사용 시, 추출할 최대 카테고리 수 (기본 10000)",
    )
    ap.add_argument(
        "--out_dir",
        default=".",
        help="CSV 저장 디렉터리 (기본: 현재 디렉터리)",
    )
    ap.add_argument(
        "--products_csv",
        default="products_all.csv",
        help="상품 CSV 파일명 (기본 products_all.csv)",
    )
    ap.add_argument(
        "--reviews_csv",
        default="reviews_all.csv",
        help="리뷰 CSV 파일명 (기본 reviews_all.csv)",
    )
    args = ap.parse_args()

    cfg = _get_config()
    schema = _schema()
    ptab = _products_table()
    rtab = _reviews_table()

    print("=" * 60)
    print("DB → products_all.csv / reviews_all.csv 추출")
    print("=" * 60)
    print(f"DB: {cfg['host']}:{cfg['port']} / danawa_crawlingdb")
    print(f"Schema: {schema}  Tables: {ptab}, {rtab}")
    print(f"출력 디렉터리: {args.out_dir}")

    conn = psycopg2.connect(**cfg)

    try:
        if args.all:
            q_prod = f"SELECT * FROM {schema}.{ptab}"
            df_products = pd.read_sql(q_prod, conn)
            product_ids = df_products["id"].astype(int).dropna().unique().tolist()
            print(f"상품 전체: {len(df_products):,}건")
        else:
            q_cats = f"""
                SELECT category_id, COUNT(*) AS cnt
                FROM {schema}.{ptab}
                WHERE category_id IS NOT NULL AND category IS NOT NULL
                GROUP BY category_id
                HAVING COUNT(*) >= %s
                ORDER BY cnt DESC
                LIMIT %s
            """
            df_cats = pd.read_sql(
                q_cats, conn, params=(args.min_products, args.max_categories)
            )
            cat_ids = df_cats["category_id"].astype(int).tolist()
            if not cat_ids:
                print("조건에 맞는 카테고리가 없습니다. --all 로 전부 추출하거나 --min_products 를 낮추세요.")
                return

            placeholders = ",".join(["%s"] * len(cat_ids))
            q_prod = f"SELECT * FROM {schema}.{ptab} WHERE category_id IN ({placeholders})"
            df_products = pd.read_sql(q_prod, conn, params=tuple(cat_ids))
            product_ids = df_products["id"].astype(int).dropna().unique().tolist()
            print(f"카테고리 {len(cat_ids):,}개, 상품 {len(df_products):,}건")

        if not product_ids:
            print("추출된 상품이 없어 리뷰를 조회하지 않습니다.")
            df_reviews = pd.DataFrame()
        else:
            placeholders = ",".join(["%s"] * len(product_ids))
            q_rev = f"SELECT * FROM {schema}.{rtab} WHERE product_id IN ({placeholders})"
            df_reviews = pd.read_sql(q_rev, conn, params=tuple(product_ids))
            print(f"리뷰 {len(df_reviews):,}건")

    finally:
        conn.close()

    os.makedirs(args.out_dir, exist_ok=True)
    path_prod = os.path.join(args.out_dir, args.products_csv)
    path_rev = os.path.join(args.out_dir, args.reviews_csv)

    df_products.to_csv(path_prod, index=False, encoding="utf-8")
    df_reviews.to_csv(path_rev, index=False, encoding="utf-8")

    print(f"\n[OK] {path_prod}")
    print(f"[OK] {path_rev}")
    print("다음: 22_prepare_report_summary_sft.py 또는 25_generate_category_report_from_csv.py 에서 CSV 사용")


if __name__ == "__main__":
    main()
