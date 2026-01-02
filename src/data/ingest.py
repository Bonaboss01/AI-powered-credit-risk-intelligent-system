# src/data/ingest.py
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic import (
    MediumScaleConfig,
    generate_accounts,
    generate_customers,
    generate_weekly_snapshot,
    generate_transactions_for_week,
    save_week_data,
)
from src.data.validate import ValidationRule, run_validation
from src.data.versioning import DatasetVersion, parse_run_date, raw_week_dir


def _rules():
    customers_rule = ValidationRule(
        required_columns=["customer_id", "age", "region", "employment_status", "income_band", "created_date"],
        max_null_rate={"age": 0.0, "region": 0.0, "employment_status": 0.0, "income_band": 0.0},
        numeric_ranges={"age": (18, 90)},
    )

    accounts_rule = ValidationRule(
        required_columns=["account_id", "customer_id", "product_type", "credit_limit", "apr", "opened_date"],
        max_null_rate={"credit_limit": 0.0, "apr": 0.0},
        numeric_ranges={"credit_limit": (500, 25000), "apr": (0.01, 0.60)},
    )

    snapshot_rule = ValidationRule(
        required_columns=[
            "snapshot_date",
            "account_id",
            "balance",
            "utilisation",
            "min_due",
            "payment_made",
            "payment_amount",
            "days_past_due",
            "missed_payment_flag",
            "default_flag",
        ],
        max_null_rate={"balance": 0.0, "utilisation": 0.0, "days_past_due": 0.0},
        numeric_ranges={"utilisation": (0.0, 1.0), "days_past_due": (0, 180)},
    )

    tx_rule = ValidationRule(
        required_columns=["transaction_id", "account_id", "transaction_date", "amount", "type"],
        max_null_rate={"amount": 0.0, "type": 0.0},
        numeric_ranges={},
    )
    return customers_rule, accounts_rule, snapshot_rule, tx_rule


def main():
    parser = argparse.ArgumentParser(description="Weekly parameterised data ingestion (synthetic generator).")
    parser.add_argument("--run-date", type=str, required=True, help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks to generate from run-date (inclusive).")
    parser.add_argument("--data-raw-dir", type=str, default="data/raw", help="Raw data directory.")
    args = parser.parse_args()

    run_dt = parse_run_date(args.run_date)
    cfg = MediumScaleConfig()

    # Generate base (customers/accounts) once for consistency across weeks
    customers = generate_customers(cfg)
    accounts = generate_accounts(cfg, customers)

    # Validate base tables
    customers_rule, accounts_rule, snapshot_rule, tx_rule = _rules()
    run_validation(customers, "customers", customers_rule)
    run_validation(accounts, "accounts", accounts_rule)

    state = None

    for w in range(args.weeks):
        current_date = run_dt + timedelta(days=7 * w)
        version = DatasetVersion.from_run_date(current_date)

        week_dir = raw_week_dir(args.data_raw_dir, version)

        snapshot, state = generate_weekly_snapshot(
            cfg=cfg,
            customers=customers,
            accounts=accounts,
            snapshot_date=current_date,
            week_index=w,
            prior_state=state,
        )
        tx = generate_transactions_for_week(cfg, accounts, snapshot, current_date, week_index=w)

        # Validate weekly tables
        run_validation(snapshot, "weekly_account_snapshot", snapshot_rule)
        run_validation(tx, "transactions", tx_rule)

        # Save weekly package
        save_week_data(week_dir, customers, accounts, snapshot, tx)

        print(f"✅ Wrote weekly dataset: {version.tag} -> {week_dir}")


if __name__ == "__main__":
    main()
