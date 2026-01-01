#!/usr/bin/env python3
"""
Synthetic credit application data generator (weekly snapshots).

Default output is Parquet (recommended).

Examples:
  # 1M rows, Parquet
  python scripts/generate_data.py --run-date 2025-12-21 --n-rows 1000000 --seed 42

  # Another week (new file)
  python scripts/generate_data.py --run-date 2025-12-28 --n-rows 1000000 --seed 43

Environment variables (optional alternative):
  RUN_DATE=2025-12-21 N_ROWS=1000000 SEED=42 python scripts/generate_data.py

Scenario knobs (optional):
  DEFAULT_RATE_SHIFT=0.00   # +0.05 increases default probability (stress)
  INCOME_SHIFT=0.00         # +0.10 increases incomes by 10%
  UNEMPLOYMENT_SHIFT=0.00   # +0.03 increases unemployment probability by 3pp
  FRAUD_SHIFT=0.00          # +0.05 increases KYC mismatch probability by 5pp

Output:
  data/raw/credit_applications_snapshot_YYYYMMDD.parquet  (default)
  or .csv if --format csv
"""
from __future__ import annotations

import os
import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Scenario:
    default_rate_shift: float = 0.0
    income_shift: float = 0.0
    unemployment_shift: float = 0.0
    fraud_shift: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-date", default=os.getenv("RUN_DATE", ""), help="YYYY-MM-DD (snapshot date)")
    p.add_argument("--n-rows", type=int, default=int(os.getenv("N_ROWS", "100000")), help="rows per snapshot (1,000,000+ ok)")
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    p.add_argument("--outdir", default="data/raw")
    p.add_argument("--format", choices=["parquet", "csv"], default=os.getenv("OUT_FORMAT", "parquet"), help="output format")
    p.add_argument("--compression", default=os.getenv("PARQUET_COMPRESSION", "snappy"), help="parquet compression (snappy|gzip|brotli|zstd|none)")
    return p.parse_args()


def get_scenario() -> Scenario:
    return Scenario(
        default_rate_shift=float(os.getenv("DEFAULT_RATE_SHIFT", "0.0")),
        income_shift=float(os.getenv("INCOME_SHIFT", "0.0")),
        unemployment_shift=float(os.getenv("UNEMPLOYMENT_SHIFT", "0.0")),
        fraud_shift=float(os.getenv("FRAUD_SHIFT", "0.0")),
    )


def weighted_choice(rng: np.random.Generator, items, probs, size: int):
    return rng.choice(items, size=size, p=probs)


def generate_snapshot(run_date: datetime, n_rows: int, seed: int, scenario: Scenario) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # IDs
    customer_id = rng.integers(1_000_000, 9_999_999, size=n_rows, endpoint=True)
    application_id = np.arange(1, n_rows + 1) + 10_000_000

    # Dates: past 24 months up to run_date
    start = run_date - timedelta(days=730)
    app_ts = rng.integers(int(start.timestamp()), int(run_date.timestamp()), size=n_rows)
    application_date = pd.to_datetime(app_ts, unit="s").normalize()

    # Demographics / stability
    age = np.clip(rng.normal(36, 10, size=n_rows).round().astype(int), 18, 75)
    region = weighted_choice(
        rng,
        ["London", "South East", "South West", "Midlands", "North", "Scotland", "Wales", "Northern Ireland"],
        [0.18, 0.14, 0.10, 0.18, 0.22, 0.09, 0.06, 0.03],
        n_rows,
    )
    residential_status = weighted_choice(rng, ["rent", "mortgage", "own", "family"], [0.42, 0.33, 0.12, 0.13], n_rows)
    time_at_address_months = np.clip(rng.exponential(36, size=n_rows).round().astype(int), 0, 240)

    # Employment: reweight by unemployment_shift
    base_unemp = 0.08
    unemp = float(np.clip(base_unemp + scenario.unemployment_shift, 0.0, 0.50))
    remaining = 1.0 - unemp
    # employed, self_employed, student, retired (relative weights)
    others = np.array([0.68, 0.12, 0.05, 0.07], dtype=float)
    others = others / others.sum() * remaining
    emp_probs = [others[0], others[1], unemp, others[2], others[3]]
    employment_status = weighted_choice(
        rng, ["employed", "self_employed", "unemployed", "student", "retired"], emp_probs, n_rows
    )

    time_in_job_months = np.clip(rng.exponential(30, size=n_rows).round().astype(int), 0, 240)
    education_level = weighted_choice(rng, ["secondary", "college", "bachelors", "masters", "phd"], [0.28, 0.23, 0.32, 0.15, 0.02], n_rows)

    # Income (+ income shift)
    annual_income = np.clip(rng.lognormal(mean=np.log(32000), sigma=0.5, size=n_rows), 8000, 250000) * (1.0 + scenario.income_shift)
    other_income = np.clip(rng.lognormal(mean=np.log(2500), sigma=0.9, size=n_rows), 0, 80000) * (1.0 + scenario.income_shift)
    annual_income = np.clip(annual_income, 8000, 250000).round(2)
    other_income = np.clip(other_income, 0, 80000).round(2)
    total_income = (annual_income + other_income).round(2)
    monthly_income = (total_income / 12.0).round(2)

    # Product + terms
    product_type = weighted_choice(
        rng,
        ["personal_loan", "credit_card", "auto_loan", "payday_like", "mortgage_topup"],
        [0.45, 0.30, 0.12, 0.08, 0.05],
        n_rows,
    )
    loan_amount = np.zeros(n_rows, dtype=float)
    term_months = np.zeros(n_rows, dtype=int)

    for pt in ["personal_loan", "credit_card", "auto_loan", "payday_like", "mortgage_topup"]:
        idx = np.where(product_type == pt)[0]
        m = len(idx)
        if m == 0:
            continue
        if pt == "personal_loan":
            loan_amount[idx] = np.clip(rng.lognormal(np.log(5500), 0.6, size=m), 500, 50000)
            term_months[idx] = rng.choice([12, 18, 24, 36, 48, 60], p=[0.12, 0.10, 0.24, 0.24, 0.18, 0.12], size=m)
        elif pt == "credit_card":
            loan_amount[idx] = np.clip(rng.lognormal(np.log(2200), 0.7, size=m), 200, 20000)
            term_months[idx] = 0
        elif pt == "auto_loan":
            loan_amount[idx] = np.clip(rng.lognormal(np.log(12000), 0.5, size=m), 2000, 60000)
            term_months[idx] = rng.choice([24, 36, 48, 60, 72], p=[0.10, 0.24, 0.28, 0.24, 0.14], size=m)
        elif pt == "payday_like":
            loan_amount[idx] = np.clip(rng.lognormal(np.log(350), 0.4, size=m), 100, 1500)
            term_months[idx] = rng.choice([1, 2, 3, 6], p=[0.35, 0.30, 0.20, 0.15], size=m)
        else:
            loan_amount[idx] = np.clip(rng.lognormal(np.log(18000), 0.55, size=m), 3000, 100000)
            term_months[idx] = rng.choice([60, 84, 120, 180], p=[0.20, 0.25, 0.35, 0.20], size=m)

    loan_amount = np.round(loan_amount, 2)

    # Bureau / behaviour
    credit_history_months = np.clip(rng.normal(84, 48, size=n_rows).round().astype(int), 0, 360)
    num_open_accounts = np.clip(rng.poisson(6, size=n_rows), 0, 35)
    num_closed_accounts = np.clip(rng.poisson(9, size=n_rows), 0, 80)
    num_delinquencies_12m = np.clip(rng.poisson(0.6, size=n_rows), 0, 12)
    num_delinquencies_24m = np.clip(num_delinquencies_12m + rng.poisson(0.5, size=n_rows), 0, 24)
    utilization_rate = np.clip(rng.beta(2, 4, size=n_rows), 0, 1)

    hard_inquiries_6m = np.clip(rng.poisson(0.8, size=n_rows), 0, 12)
    hard_inquiries_12m = np.clip(hard_inquiries_6m + rng.poisson(0.6, size=n_rows), 0, 24)
    bankruptcies = rng.binomial(1, 0.02, size=n_rows)
    collections = np.clip(rng.poisson(0.15, size=n_rows), 0, 8)

    monthly_debt_payments = np.clip(rng.lognormal(np.log(650), 0.7, size=n_rows), 0, 10000)
    dti = np.clip(monthly_debt_payments / (monthly_income + 1e-6), 0, 3)
    savings_balance = np.clip(rng.lognormal(np.log(1200), 1.0, size=n_rows), 0, 200000)

    # Fraud / KYC proxies
    device_trust_score = np.clip(rng.normal(72, 15, size=n_rows), 0, 100)
    email_domain_risk = weighted_choice(rng, ["low", "medium", "high"], [0.86, 0.11, 0.03], n_rows)
    ip_risk = weighted_choice(rng, ["low", "medium", "high"], [0.90, 0.08, 0.02], n_rows)
    kyc_mismatch_prob = float(np.clip(0.03 + scenario.fraud_shift, 0.0, 0.30))
    kyc_match = rng.binomial(1, 1.0 - kyc_mismatch_prob, size=n_rows)

    # Risk linear signal
    risk_linear = (
        0.9 * (dti - 0.35)
        + 0.8 * (utilization_rate - 0.35)
        + 0.25 * num_delinquencies_12m
        + 0.12 * hard_inquiries_6m
        + 0.8 * bankruptcies
        + 0.15 * collections
        + 0.25 * (employment_status == "unemployed").astype(int)
        + 0.12 * (residential_status == "rent").astype(int)
        - 0.10 * np.log1p(savings_balance / 1000)
        - 0.06 * np.log1p(total_income / 10000)
        - 0.08 * np.log1p(credit_history_months / 12)
    )

    # Pricing (APR)
    base_apr = np.select(
        [
            product_type == "personal_loan",
            product_type == "credit_card",
            product_type == "auto_loan",
            product_type == "payday_like",
            product_type == "mortgage_topup",
        ],
        [0.12, 0.22, 0.09, 1.50, 0.06],
        default=0.15,
    )
    apr = np.clip(base_apr + 0.08 * sigmoid(risk_linear) * 4, 0.02, 2.50)

    bureau_score = np.clip(
        720 - 110 * sigmoid(risk_linear) - 25 * num_delinquencies_12m + rng.normal(0, 35, size=n_rows),
        300,
        850,
    ).round().astype(int)

    # Outcome: default within 12 months (+ scenario shift)
    p_default = sigmoid(
        -2.2
        + 1.8 * (dti - 0.35)
        + 1.6 * (utilization_rate - 0.35)
        + 0.45 * num_delinquencies_12m
        + 0.25 * hard_inquiries_6m
        + 1.8 * bankruptcies
        + 0.35 * collections
        + 0.20 * (employment_status == "unemployed").astype(int)
        + 0.12 * (employment_status == "student").astype(int)
        + 0.10 * (residential_status == "rent").astype(int)
        + 0.08 * (product_type == "payday_like").astype(int)
        - 0.18 * np.log1p(total_income / 10000)
        - 0.15 * np.log1p(savings_balance / 1000)
        - 0.08 * np.log1p(credit_history_months / 12)
        - 0.10 * (time_in_job_months / 24)
        - 0.08 * (time_at_address_months / 24)
        + rng.normal(0, 0.25, size=n_rows)
    )
    p_default = np.clip(p_default + scenario.default_rate_shift, 0.0001, 0.9999)
    default_12m = (rng.uniform(size=n_rows) < p_default).astype(int)

    pd_bucket = pd.cut(p_default, bins=[-0.01, 0.02, 0.05, 0.10, 0.20, 1.01], labels=["A", "B", "C", "D", "E"]).astype(str)

    df = pd.DataFrame(
        {
            "application_id": application_id,
            "customer_id": customer_id,
            "application_date": application_date,
            "product_type": product_type,
            "region": region,
            "age": age,
            "education_level": education_level,
            "employment_status": employment_status,
            "time_in_job_months": time_in_job_months,
            "residential_status": residential_status,
            "time_at_address_months": time_at_address_months,
            "annual_income": np.round(annual_income, 2),
            "other_income": np.round(other_income, 2),
            "total_income": np.round(total_income, 2),
            "monthly_income": monthly_income,
            "loan_amount": np.round(loan_amount, 2),
            "term_months": term_months,
            "monthly_debt_payments": np.round(monthly_debt_payments, 2),
            "dti": np.round(dti, 4),
            "savings_balance": np.round(savings_balance, 2),
            "credit_history_months": credit_history_months,
            "num_open_accounts": num_open_accounts,
            "num_closed_accounts": num_closed_accounts,
            "num_delinquencies_12m": num_delinquencies_12m,
            "num_delinquencies_24m": num_delinquencies_24m,
            "utilization_rate": np.round(utilization_rate, 4),
            "hard_inquiries_6m": hard_inquiries_6m,
            "hard_inquiries_12m": hard_inquiries_12m,
            "bankruptcies": bankruptcies,
            "collections": collections,
            "bureau_score": bureau_score,
            "apr": np.round(apr, 4),
            "device_trust_score": np.round(device_trust_score, 1),
            "email_domain_risk": email_domain_risk,
            "ip_risk": ip_risk,
            "kyc_match": kyc_match,
            "p_default_true": np.round(p_default, 6),
            "pd_bucket": pd_bucket,
            "default_12m": default_12m,
        }
    )
    return df


def main() -> None:
    args = parse_args()
    scenario = get_scenario()

    if args.run_date:
        run_date = datetime.strptime(args.run_date, "%Y-%m-%d")
    else:
        run_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    if args.n_rows < 1:
        raise SystemExit("--n-rows must be >= 1")

    df = generate_snapshot(run_date=run_date, n_rows=args.n_rows, seed=args.seed, scenario=scenario)

    os.makedirs(args.outdir, exist_ok=True)
    base_name = f"credit_applications_snapshot_{run_date.strftime('%Y%m%d')}"

    if args.format == "csv":
        out_path = os.path.join(args.outdir, base_name + ".csv")
        df.to_csv(out_path, index=False)
    else:
        # Parquet (recommended)
        # Requires: pyarrow (preferred) or fastparquet
        out_path = os.path.join(args.outdir, base_name + ".parquet")
        compression = None if args.compression.lower() in {"none", ""} else args.compression
        df.to_parquet(out_path, index=False, compression=compression)

    print("✅ Wrote snapshot:", out_path)
    print("Rows:", f"{len(df):,}", "Columns:", df.shape[1])
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
