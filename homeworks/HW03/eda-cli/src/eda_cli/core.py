from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


# ---------- dataclasses ----------

@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


# ---------- core EDA ----------

def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


# ---------- quality heuristics ----------

def compute_quality_flags(summary: DatasetSummary, missing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Простые эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    - слишком много колонок;
    - константные колонки;
    - высокая кардинальность категориальных признаков;
    - НОВОЕ: колонки с очень низкой вариативностью;
    - НОВОЕ: колонки с критически большим числом пропусков.
    """
    flags: Dict[str, Any] = {}

    # Базовые флаги по датасету
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # Константные колонки
    constant_columns: List[str] = [
        col.name for col in summary.columns if col.unique == 1
    ]
    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns

    # Высокая кардинальность категориальных признаков
    high_cardinality_threshold = 50
    high_cardinality_columns: List[str] = []
    for col in summary.columns:
        if col.dtype == "object" and col.unique is not None:
            if col.unique > high_cardinality_threshold:
                high_cardinality_columns.append(col.name)

    flags["has_high_cardinality_categoricals"] = len(high_cardinality_columns) > 0
    flags["high_cardinality_columns"] = high_cardinality_columns

    # НОВАЯ ЭВРИСТИКА 1: очень низкая вариативность числовых признаков
    low_variance_columns: List[str] = []
    for col in summary.columns:
        if col.is_numeric and col.std is not None and col.std < 1e-6:
            low_variance_columns.append(col.name)
    flags["has_low_variance_columns"] = len(low_variance_columns) > 0
    flags["low_variance_columns"] = low_variance_columns

    # НОВАЯ ЭВРИСТИКА 2: критически много пропусков в отдельных колонках
    critical_missing_threshold = 0.8
    critical_missing_columns: List[str] = []
    for col in summary.columns:
        if col.missing_share is not None and col.missing_share > critical_missing_threshold:
            critical_missing_columns.append(col.name)
    flags["has_critical_missing_columns"] = len(critical_missing_columns) > 0
    flags["critical_missing_columns"] = critical_missing_columns

    # Простейший «скор» качества
    score = 1.0
    score -= max_missing_share
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.1
    if flags["has_low_variance_columns"]:
        score -= 0.05
    if flags["has_critical_missing_columns"]:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


# ---------- helpers for printing ----------

def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
