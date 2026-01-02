# src/data/versioning.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetVersion:
    run_date: date
    year: int
    iso_week: int

    @property
    def tag(self) -> str:
        # Example: credit_data_2026_W01
        return f"credit_data_{self.year}_W{self.iso_week:02d}"

    @staticmethod
    def from_run_date(run_date: date) -> "DatasetVersion":
        iso = run_date.isocalendar()
        return DatasetVersion(run_date=run_date, year=iso.year, iso_week=iso.week)


def parse_run_date(run_date_str: str) -> date:
    return datetime.strptime(run_date_str, "%Y-%m-%d").date()


def raw_week_dir(data_raw_dir: str | Path, version: DatasetVersion) -> Path:
    return Path(data_raw_dir) / version.tag


def processed_week_dir(data_processed_dir: str | Path, version: DatasetVersion) -> Path:
    return Path(data_processed_dir) / version.tag


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
