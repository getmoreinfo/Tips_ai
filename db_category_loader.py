# db_category_loader.py
# 역할: danawa_crawlingdb에서 카테고리별 상품·리뷰를 직접 조회해 DataFrame으로 반환.
#       CSV 없이 분석 파이프라인(aggregate_category_with_reviews 등)에 넘길 수 있음.

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def _schema() -> str:
    return os.getenv("PGSCHEMA", "public1")


def _products_table() -> str:
    """상품 테이블명. .env의 PGTABLE과 동일하면 호환됨."""
    return os.getenv("PGTABLE", "products")


def _reviews_table() -> str:
    """리뷰 테이블명. .env에 PGREVIEWSTABLE이 없으면 기본 'reviews'."""
    return os.getenv("PGREVIEWSTABLE", "reviews")


def get_crawlingdb_config() -> dict:
    """danawa_crawlingdb 접속 설정 (.env PGHOST 등 사용)."""
    return {
        "host": os.getenv("PGHOST"),
        "port": os.getenv("PGPORT"),
        "database": "danawa_crawlingdb",
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
    }


def load_category_from_db(
    category_id: Optional[int] = None,
    category_contains: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, str]:
    """
    DB에서 해당 카테고리 상품 + 그 상품들의 리뷰만 조회해 (df_products, df_reviews, category_id, category_path) 반환.

    - category_contains가 있으면: category 컬럼에 해당 문자열이 포함된 행만 사용 (예: "유모차").
    - 없으면: category_id로만 필터.

    반환:
      df_products: 해당 카테고리 상품 전체 (products_all.csv와 동일한 컬럼 구성)
      df_reviews:  해당 상품 id에 대한 리뷰만 (reviews_all.csv와 동일한 컬럼 구성)
      category_id:  사용된 category_id (집계/표시용)
      category_path: 대표 카테고리 경로 문자열 (첫 행 기준)
    """
    cfg = get_crawlingdb_config()
    if not all([cfg["host"], cfg["port"], cfg["user"], cfg["password"]]):
        raise RuntimeError(
            "DB 연결 정보가 부족합니다. .env에 PGHOST, PGPORT, PGUSER, PGPASSWORD를 설정하세요."
        )
    conn = psycopg2.connect(**cfg)

    try:
        schema = _schema()
        ptab = _products_table()
        rtab = _reviews_table()
        if category_contains:
            # category 문자열에 키워드 포함된 행만
            q = f"""
                SELECT * FROM {schema}.{ptab}
                WHERE category IS NOT NULL AND category LIKE %s
            """
            df_products = pd.read_sql(q, conn, params=(f"%{category_contains}%",))
            if df_products.empty:
                raise ValueError(
                    f"category에 '{category_contains}'가 포함된 데이터를 DB에서 찾지 못했습니다."
                )
            category_id = int(df_products["category_id"].iloc[0])
        else:
            if category_id is None:
                raise ValueError("category_id 또는 category_contains 중 하나는 필요합니다.")
            q = f"""
                SELECT * FROM {schema}.{ptab}
                WHERE category_id = %s
            """
            df_products = pd.read_sql(q, conn, params=(int(category_id),))
            if df_products.empty:
                raise ValueError(
                    f"category_id={category_id} 데이터를 DB에서 찾지 못했습니다."
                )

        category_path = (
            str(df_products["category"].dropna().iloc[0])
            if df_products["category"].notna().any()
            else ""
        )
        product_ids = df_products["id"].astype(int).tolist()
        if not product_ids:
            df_reviews = pd.DataFrame()
        else:
            # IN 절용 플레이스홀더 (보안상 %s 사용)
            placeholders = ",".join(["%s"] * len(product_ids))
            q_rev = f"""
                SELECT * FROM {schema}.{rtab}
                WHERE product_id IN ({placeholders})
            """
            df_reviews = pd.read_sql(q_rev, conn, params=tuple(product_ids))

        return df_products, df_reviews, int(category_id), category_path
    finally:
        conn.close()
