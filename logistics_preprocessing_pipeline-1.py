"""
Week 2: Logistics Data Collection, Cleaning and Preprocessing
Dataset: Brazilian E-Commerce Public Dataset by Olist

Usage:
1. Download the Olist CSV files from:
   https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Extract the CSV files into a folder named "data/olist".
3. Install dependencies:
   pip install pandas numpy scikit-learn
4. Run:
   python logistics_preprocessing_pipeline.py

Outputs are written to output/:
- cleaned_order_level.csv
- cleaned_order_items.csv
- data_quality_report.csv
- preprocessing_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/olist")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Download and extract the Olist dataset "
            "into data/olist before running this script."
        )
    return pd.read_csv(path)


def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def missing_report(df, table_name):
    return pd.DataFrame({
        "table": table_name,
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 2) for c in df.columns
        ],
    })


def iqr_flag(series):
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index)
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.lt(lower) | series.gt(upper)


def main():
    # 1. Collection / loading
    orders = standardize_columns(load_csv("olist_orders_dataset.csv"))
    items = standardize_columns(load_csv("olist_order_items_dataset.csv"))
    customers = standardize_columns(load_csv("olist_customers_dataset.csv"))
    products = standardize_columns(load_csv("olist_products_dataset.csv"))

    initial_counts = {
        "orders_rows_loaded": len(orders),
        "items_rows_loaded": len(items),
        "customers_rows_loaded": len(customers),
        "products_rows_loaded": len(products),
    }

    # 2. Remove exact duplicate rows
    duplicates = {
        "orders_exact_duplicates": int(orders.duplicated().sum()),
        "items_exact_duplicates": int(items.duplicated().sum()),
        "customers_exact_duplicates": int(customers.duplicated().sum()),
        "products_exact_duplicates": int(products.duplicated().sum()),
    }
    orders = orders.drop_duplicates().copy()
    items = items.drop_duplicates().copy()
    customers = customers.drop_duplicates().copy()
    products = products.drop_duplicates().copy()

    # 3. Resolve order primary-key duplicates conservatively.
    # Report them, then keep first row only for this educational pipeline.
    duplicate_order_ids = int(orders["order_id"].duplicated().sum())
    orders = orders.drop_duplicates(subset=["order_id"], keep="first").copy()

    # 4. Parse date columns; invalid strings become NaT and are counted in QA.
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(orders[column], errors="coerce")

    # 5. Missing-value treatment: keep delivery dates missing when not delivered.
    # Do not impute identifiers or delivery timestamps.
    if "customer_city" in customers.columns:
        customers["customer_city"] = (
            customers["customer_city"].fillna("unknown").astype("string").str.strip()
        )
    if "product_weight_g" in products.columns:
        products["weight_missing"] = products["product_weight_g"].isna().astype("int8")
        median_weight = products["product_weight_g"].median()
        products["product_weight_g"] = products["product_weight_g"].fillna(median_weight)

    # 6. Join tables. Item-level output contains one row per order item.
    order_level = orders.merge(
        customers, on="customer_id", how="left", validate="many_to_one"
    )
    item_product = items.merge(
        products, on="product_id", how="left", validate="many_to_one"
    )
    item_level = order_level.merge(item_product, on="order_id", how="left")

    # 7. Domain validation flags (do not silently delete potential exceptions).
    if "freight_value" in item_level.columns:
        item_level["invalid_freight_flag"] = item_level["freight_value"].lt(0)
        item_level["freight_outlier_flag"] = iqr_flag(item_level["freight_value"])
    if "price" in item_level.columns:
        item_level["invalid_price_flag"] = item_level["price"].le(0)
        item_level["price_outlier_flag"] = iqr_flag(item_level["price"])
    if "product_weight_g" in item_level.columns:
        item_level["weight_outlier_flag"] = iqr_flag(item_level["product_weight_g"])

    # 8. Delivery metrics at order grain. Use delivered orders only.
    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    delivered = delivered.dropna(subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
    ])
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    if "order_estimated_delivery_date" in delivered.columns:
        delivered["delivery_delay_days"] = (
            delivered["order_delivered_customer_date"]
            - delivered["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86400

    delivered["delivery_time_outlier_flag"] = iqr_flag(delivered["delivery_days"])

    # 9. Build data quality report for source and output tables.
    reports = [
        missing_report(orders, "orders"),
        missing_report(items, "order_items"),
        missing_report(customers, "customers"),
        missing_report(products, "products"),
        missing_report(delivered, "delivered_orders"),
    ]
    quality = pd.concat(reports, ignore_index=True)
    quality.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    # 10. Save clean outputs.
    order_level.to_csv(OUTPUT_DIR / "cleaned_order_level.csv", index=False)
    item_level.to_csv(OUTPUT_DIR / "cleaned_order_items.csv", index=False)
    delivered.to_csv(OUTPUT_DIR / "cleaned_delivered_orders.csv", index=False)

    # 11. Correct scaling demonstration: split first, fit only on training set.
    numeric_features = [
        c for c in ["price", "freight_value", "product_weight_g"]
        if c in item_level.columns
    ]
    if numeric_features:
        model_frame = item_level[numeric_features].replace(
            [np.inf, -np.inf], np.nan
        )
        model_frame = model_frame.fillna(model_frame.median(numeric_only=True))
        if len(model_frame) >= 5:
            train, test = train_test_split(
                model_frame, test_size=0.20, random_state=42
            )
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train)
            test_scaled = scaler.transform(test)
            # Save only the demonstration outputs; source rows remain unchanged.
            pd.DataFrame(
                train_scaled, columns=numeric_features
            ).to_csv(OUTPUT_DIR / "scaled_training_features.csv", index=False)
            pd.DataFrame(
                test_scaled, columns=numeric_features
            ).to_csv(OUTPUT_DIR / "scaled_test_features.csv", index=False)

    # 12. Summary of measured results from this run.
    summary = {
        **initial_counts,
        **duplicates,
        "duplicate_order_ids_after_exact_dedup": duplicate_order_ids,
        "orders_after_cleaning": len(orders),
        "items_after_cleaning": len(items),
        "delivered_orders_with_valid_dates": len(delivered),
        "negative_freight_records": (
            int(item_level["invalid_freight_flag"].sum())
            if "invalid_freight_flag" in item_level else 0
        ),
        "nonpositive_price_records": (
            int(item_level["invalid_price_flag"].sum())
            if "invalid_price_flag" in item_level else 0
        ),
        "delivery_time_outliers_flagged": int(
            delivered["delivery_time_outlier_flag"].sum()
        ),
    }
    with (OUTPUT_DIR / "preprocessing_summary.txt").open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("Preprocessing completed.")
    print(pd.Series(summary).to_string())
    print(f"\nFiles saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
