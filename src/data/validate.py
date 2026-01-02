# src/data/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ValidationRule:
    required_columns: List[str]
    max_null_rate: Dict[str, float]          # e.g. {"income_band": 0.02}
    numeric_ranges: Dict[str, Tuple[float, float]]  # e.g. {"balance": (0, 20000)}


def _null_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def validate_schema(df: pd.DataFrame, table_name: str, rule: ValidationRule) -> List[str]:
    errors: List[str] = []

    missing = [c for c in rule.required_columns if c not in df.columns]
    if missing:
        errors.append(f"[{table_name}] Missing required columns: {missing}")

    return errors


def validate_nulls(df: pd.DataFrame, table_name: str, rule: ValidationRule) -> List[str]:
    errors: List[str] = []
    for col, max_rate in rule.max_null_rate.items():
        if col not in df.columns:
            continue
        rate = _null_rate(df[col])
        if rate > max_rate:
            errors.append(f"[{table_name}] Null rate too high for '{col}': {rate:.3f} > {max_rate:.3f}")
    return errors


def validate_ranges(df: pd.DataFrame, table_name: str, rule: ValidationRule) -> List[str]:
    errors: List[str] = []
    for col, (lo, hi) in rule.numeric_ranges.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s[(~s.isna()) & ((s < lo) | (s > hi))]
        if len(bad) > 0:
            errors.append(
                f"[{table_name}] Out-of-range values in '{col}': {len(bad)} rows outside [{lo}, {hi}]"
            )
    return errors


def run_validation(df: pd.DataFrame, table_name: str, rule: ValidationRule) -> None:
    errors: List[str] = []
    errors += validate_schema(df, table_name, rule)
    errors += validate_nulls(df, table_name, rule)
    errors += validate_ranges(df, table_name, rule)

    if errors:
        msg = "\n".join(errors)
        raise ValueError(f"Validation failed:\n{msg}")
