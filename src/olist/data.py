"""Build one row per delivered order from the nine Olist tables.

Columns are split by when they are known, because the three analyses need
different information sets:

- known at purchase: customer and seller location, distance, price, freight,
  weight, category, number of items and sellers, promised delivery window,
  purchase date. These are the confounders for the causal analysis and the
  only inputs allowed in the delay-risk model.
- known after delivery: whether and how late the order arrived (treatment).
- outcomes: review score, and whether the customer bought again.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
DATASET = "olistbr/brazilian-ecommerce"


def download(raw: Path = RAW) -> None:
    if (raw / "olist_orders_dataset.csv").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    subprocess.run([str(kaggle), "datasets", "download", "-d", DATASET, "-p", str(raw), "--unzip"], check=True)


def _read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RAW / f"olist_{name}_dataset.csv", **kw)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build() -> pd.DataFrame:
    ts = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]
    orders = _read("orders", parse_dates=ts)
    orders = orders[orders.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"])
    customers = _read("customers")
    sellers = _read("sellers")
    items = _read("order_items")
    products = _read("products")
    cat = pd.read_csv(RAW / "product_category_name_translation.csv")
    reviews = _read("order_reviews", parse_dates=["review_creation_date", "review_answer_timestamp"])
    geo = _read("geolocation").groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()

    products = products.merge(cat, on="product_category_name", how="left")
    items = items.merge(products[["product_id", "product_weight_g", "product_category_name_english"]],
                        on="product_id", how="left").merge(sellers, on="seller_id", how="left")
    items = items.join(geo, on="seller_zip_code_prefix")
    # Order-level aggregates; the main seller is the one with the largest share of the price.
    main = items.sort_values("price", ascending=False).drop_duplicates("order_id")
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"), n_sellers=("seller_id", "nunique"),
        price=("price", "sum"), freight=("freight_value", "sum"), weight_g=("product_weight_g", "sum"),
    )
    main = main.set_index("order_id")[["seller_id", "seller_state", "product_category_name_english",
                                       "geolocation_lat", "geolocation_lng"]]
    main.columns = ["seller_id", "seller_state", "category", "seller_lat", "seller_lng"]

    df = orders.merge(customers, on="customer_id").join(agg, on="order_id").join(main, on="order_id")
    df = df.join(geo.rename(columns={"geolocation_lat": "cust_lat", "geolocation_lng": "cust_lng"}),
                 on="customer_zip_code_prefix")
    df["distance_km"] = haversine_km(df.seller_lat, df.seller_lng, df.cust_lat, df.cust_lng)
    df["same_state"] = (df.seller_state == df.customer_state).astype(int)
    df["promised_days"] = (df.order_estimated_delivery_date - df.order_purchase_timestamp.dt.normalize()).dt.days
    df["freight_share"] = df.freight / (df.price + df.freight)
    df["purchase_month"] = df.order_purchase_timestamp.dt.to_period("M").astype(str)
    df["purchase_dow"] = df.order_purchase_timestamp.dt.dayofweek
    # Seller experience: orders the main seller had shipped before this one (no look-ahead).
    df = df.sort_values("order_purchase_timestamp")
    df["seller_prior_orders"] = df.groupby("seller_id").cumcount()

    # Treatment
    df["late_days"] = (df.order_delivered_customer_date.dt.normalize() - df.order_estimated_delivery_date).dt.days
    df["late"] = (df.late_days > 0).astype(int)
    df["delivery_days"] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.total_seconds() / 86400

    # Outcome 1: last answered review per order
    rv = reviews.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    df = df.merge(rv[["order_id", "review_score", "review_creation_date"]], on="order_id", how="left")
    # A review created before delivery cannot reflect the delay.
    df["review_after_delivery"] = (df.review_creation_date >= df.order_delivered_customer_date.dt.normalize()).astype(int)

    # Outcome 2: does the same person (customer_unique_id) order again, and when
    df = df.sort_values("order_purchase_timestamp")
    nxt = df.groupby("customer_unique_id").order_purchase_timestamp.shift(-1)
    df["days_to_next_order"] = (nxt - df.order_purchase_timestamp).dt.total_seconds() / 86400
    df["order_seq"] = df.groupby("customer_unique_id").cumcount()
    end = df.order_purchase_timestamp.max()
    df["days_observed"] = (end - df.order_purchase_timestamp).dt.total_seconds() / 86400
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/orders.parquet"))
    args = p.parse_args()
    download()
    df = build()
    df.to_parquet(args.out, index=False)
    print(f"{len(df):,} delivered orders, {df.late.mean():.3%} late, "
          f"{df.review_score.notna().mean():.1%} reviewed, {df.customer_unique_id.nunique():,} customers")


if __name__ == "__main__":
    main()
