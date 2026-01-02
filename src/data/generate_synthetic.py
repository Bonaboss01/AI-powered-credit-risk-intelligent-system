# src/data/generate_synthetic.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class MediumScaleConfig:
    n_customers: int = 200_000
    accounts_per_customer_mean: float = 2.0
    n_weeks: int = 26

    # drift knobs
    drift_start_week: int = 8
    drift_strength: float = 0.10  # 0.0 to 0.3 typical

    seed: int = 42


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _income_band(r: np.random.Generator, n: int) -> np.ndarray:
    bands = np.array(["<20k", "20-35k", "35-50k", "50-75k", "75k+"])
    probs = np.array([0.20, 0.35, 0.25, 0.15, 0.05])
    return r.choice(bands, size=n, p=probs)


def _employment(r: np.random.Generator, n: int) -> np.ndarray:
    cats = np.array(["employed", "self_employed", "unemployed", "student", "retired"])
    probs = np.array([0.62, 0.10, 0.10, 0.10, 0.08])
    return r.choice(cats, size=n, p=probs)


def generate_customers(cfg: MediumScaleConfig) -> pd.DataFrame:
    r = _rng(cfg.seed)
    customer_id = np.arange(1, cfg.n_customers + 1)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": r.integers(18, 75, size=cfg.n_customers),
            "region": r.choice(["London", "Midlands", "North", "Scotland", "Wales", "NI"], size=cfg.n_customers),
            "employment_status": _employment(r, cfg.n_customers),
            "income_band": _income_band(r, cfg.n_customers),
            "created_date": pd.to_datetime("2019-01-01")
            + pd.to_timedelta(r.integers(0, 5 * 365, size=cfg.n_customers), unit="D"),
        }
    )
    return df


def generate_accounts(cfg: MediumScaleConfig, customers: pd.DataFrame) -> pd.DataFrame:
    r = _rng(cfg.seed + 1)

    n = cfg.n_customers
    apc = r.poisson(lam=cfg.accounts_per_customer_mean, size=n)
    apc = np.clip(apc, 1, 6)

    account_rows = int(apc.sum())
    customer_ids = np.repeat(customers["customer_id"].to_numpy(), apc)

    product_type = r.choice(["credit_card", "personal_loan"], size=account_rows, p=[0.70, 0.30])

    credit_limit = np.where(
        product_type == "credit_card",
        r.normal(loc=3500, scale=1500, size=account_rows),
        r.normal(loc=8000, scale=3000, size=account_rows),
    )
    credit_limit = np.clip(credit_limit, 500, 25_000).round(0).astype(int)

    apr = np.where(
        product_type == "credit_card",
        r.normal(loc=0.28, scale=0.06, size=account_rows),
        r.normal(loc=0.18, scale=0.05, size=account_rows),
    )
    apr = np.clip(apr, 0.05, 0.49)

    opened_date = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        r.integers(0, 5 * 365, size=account_rows), unit="D"
    )

    df = pd.DataFrame(
        {
            "account_id": np.arange(1, account_rows + 1),
            "customer_id": customer_ids,
            "product_type": product_type,
            "credit_limit": credit_limit,
            "apr": apr,
            "opened_date": opened_date,
        }
    )
    return df


def _risk_base(customers: pd.DataFrame, accounts: pd.DataFrame) -> pd.Series:
    cust = customers.set_index("customer_id")
    a = accounts.join(cust, on="customer_id", how="left")

    income_map = {"<20k": 0.35, "20-35k": 0.22, "35-50k": 0.16, "50-75k": 0.10, "75k+": 0.06}
    emp_map = {"employed": 0.12, "self_employed": 0.16, "unemployed": 0.30, "student": 0.20, "retired": 0.14}

    base = (
        a["income_band"].map(income_map).astype(float)
        + a["employment_status"].map(emp_map).astype(float)
        + (a["age"].astype(float) < 25) * 0.06
    )
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return base


def generate_weekly_snapshot(
    cfg: MediumScaleConfig,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    snapshot_date: date,
    week_index: int,
    prior_state: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = _rng(cfg.seed + 100 + week_index)
    n = len(accounts)

    base_risk = _risk_base(customers, accounts).to_numpy()

    drift = 0.0
    if week_index >= cfg.drift_start_week:
        drift = cfg.drift_strength * ((week_index - cfg.drift_start_week + 1) / (cfg.n_weeks - cfg.drift_start_week + 1))

    if prior_state is None:
        utilisation = np.clip(r.beta(2.2, 5.0, size=n) + (base_risk * 0.20) + drift * 0.15, 0, 1)
        balance = (utilisation * accounts["credit_limit"].to_numpy()).round(2)
        days_past_due = r.choice([0, 0, 0, 0, 5, 15, 30, 60], size=n, p=[0.55, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.01])
    else:
        utilisation = np.clip(
            prior_state["utilisation"].to_numpy()
            + r.normal(0, 0.05, size=n)
            + (base_risk * 0.02)
            + drift * 0.03,
            0,
            1,
        )
        balance = (utilisation * accounts["credit_limit"].to_numpy()).round(2)

        dpd_prev = prior_state["days_past_due"].to_numpy()
        step = r.choice([0, 0, 0, 5, 10, 15, 30], size=n, p=[0.58, 0.14, 0.08, 0.08, 0.06, 0.04, 0.02])
        worsen_prob = np.clip(0.05 + base_risk * 0.20 + drift * 0.25, 0, 0.45)
        worsen = r.random(n) < worsen_prob
        days_past_due = np.where(
            worsen,
            np.clip(dpd_prev + step, 0, 180),
            np.maximum(dpd_prev - r.choice([0, 0, 5, 10], size=n, p=[0.70, 0.15, 0.10, 0.05]), 0),
        )

    min_due = np.maximum(15, (balance * r.uniform(0.02, 0.06, size=n))).round(2)

    pay_prob = np.clip(0.92 - base_risk * 0.35 - drift * 0.20, 0.35, 0.97)
    payment_made = (r.random(n) < pay_prob).astype(int)

    payment_amount = np.where(
        payment_made == 1,
        np.minimum(balance, min_due * r.uniform(0.9, 2.5, size=n)),
        0.0,
    ).round(2)

    missed_payment_flag = (payment_made == 0).astype(int)

    default_prob = np.clip(0.003 + (days_past_due >= 90) * 0.08 + base_risk * 0.02 + drift * 0.03, 0, 0.25)
    default_flag = (r.random(n) < default_prob).astype(int)

    snap = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(snapshot_date),
            "account_id": accounts["account_id"].to_numpy(),
            "balance": balance,
            "utilisation": utilisation.round(4),
            "min_due": min_due,
            "payment_made": payment_made,
            "payment_amount": payment_amount,
            "days_past_due": days_past_due.astype(int),
            "missed_payment_flag": missed_payment_flag,
            "default_flag": default_flag,
        }
    )

    state = snap[["account_id", "balance", "utilisation", "days_past_due"]].copy()
    return snap, state


def generate_transactions_for_week(
    cfg: MediumScaleConfig,
    accounts: pd.DataFrame,
    snapshot: pd.DataFrame,
    snapshot_date: date,
    week_index: int,
) -> pd.DataFrame:
    r = _rng(cfg.seed + 500 + week_index)

    sample_frac = 0.25
    sampled = snapshot.sample(frac=sample_frac, random_state=cfg.seed + week_index)

    n = len(sampled)
    tx_counts = r.poisson(lam=2.2, size=n)
    tx_counts = np.clip(tx_counts, 0, 6)

    account_ids = np.repeat(sampled["account_id"].to_numpy(), tx_counts)
    total_tx = len(account_ids)
    if total_tx == 0:
        return pd.DataFrame(columns=["transaction_id", "account_id", "transaction_date", "amount", "type"])

    tx_type = r.choice(["purchase", "payment", "fee", "cash_advance"], size=total_tx, p=[0.72, 0.20, 0.06, 0.02])

    amt = np.where(
        tx_type == "purchase",
        r.gamma(shape=2.0, scale=35.0, size=total_tx),
        np.where(
            tx_type == "payment",
            r.gamma(shape=2.0, scale=55.0, size=total_tx),
            np.where(tx_type == "fee", r.uniform(5, 25, size=total_tx), r.uniform(50, 250, size=total_tx)),
        ),
    )
    amt = np.round(amt, 2)
    amt = np.where(tx_type == "payment", -amt, amt)

    start = pd.to_datetime(snapshot_date) - pd.to_timedelta(6, unit="D")
    tx_dates = start + pd.to_timedelta(r.integers(0, 7, size=total_tx), unit="D")

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, total_tx + 1),
            "account_id": account_ids,
            "transaction_date": tx_dates,
            "amount": amt,
            "type": tx_type,
        }
    )
    return df


def save_week_data(
    out_dir: Path,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    snapshot: pd.DataFrame,
    tx: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out_dir / "customers.csv", index=False)
    accounts.to_csv(out_dir / "accounts.csv", index=False)
    snapshot.to_csv(out_dir / "weekly_account_snapshot.csv", index=False)
    tx.to_csv(out_dir / "transactions.csv", index=False)
