"""
data_processing.py
==================
Reusable, UI-independent functions for the Sales Analytics Dashboard.

Sections
--------
1. Loading            - read CSV / Excel / JSON files
2. Validation         - dimensions, dtypes, missing values, duplicates
3. Cleaning           - missing values, duplicates, inconsistent text,
                        date formats, invalid numbers, outliers
4. Statistics & KPIs  - summaries, KPIs, aggregations for charts
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".json")

# Canonical column names the dashboard understands.
REQUIRED_COLUMNS = ["Order ID", "Order Date", "Sales", "Quantity", "Profit"]
OPTIONAL_COLUMNS = [
    "Ship Date", "Ship Mode", "Customer ID", "Segment", "Country", "City",
    "State", "Region", "Product ID", "Category", "Sub-Category",
    "Product Name", "Discount",
]
CATEGORICAL_COLUMNS = [
    "Ship Mode", "Segment", "Country", "City", "State", "Region",
    "Category", "Sub-Category", "Product Name", "Customer ID", "Product ID",
]
NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]

# Text values that really mean "no value".
_PLACEHOLDER_RE = re.compile(
    r"^(|nan|none|null|n/?a|na|nil|unknown|unknown\s.*|\?+|-+|not\s+available|"
    r"not\s+applicable|missing|undefined)$",
    re.IGNORECASE,
)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50,
}

# Date formats tried in order. Ambiguous numeric dates are read day-first.
DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y",
    "%d-%B-%Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y",
]
MAX_SHIP_DAYS = 60  # a ship date further than this after the order date is suspicious


class DataValidationError(Exception):
    """Raised for problems the user can fix (bad file, missing columns...)."""


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def get_excel_sheets(file_bytes: bytes) -> list[str]:
    """Return the sheet names of an .xlsx workbook."""
    try:
        with pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl") as xls:
            return list(xls.sheet_names)
    except Exception as exc:  # corrupt / not really xlsx
        raise DataValidationError(f"Could not read the Excel workbook: {exc}") from exc


def load_file(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV, Excel (.xlsx) or JSON file into a DataFrame.

    Raises DataValidationError with a friendly message on any problem.
    """
    if not file_bytes:
        raise DataValidationError("The uploaded file is empty.")

    name = (filename or "").lower()
    ext = next((e for e in SUPPORTED_EXTENSIONS if name.endswith(e)), None)
    if ext is None:
        raise DataValidationError(
            f"Unsupported file type '{name.rsplit('.', 1)[-1] if '.' in name else name}'. "
            f"Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    try:
        if ext == ".csv":
            df = _read_csv(file_bytes)
        elif ext == ".xlsx":
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name or 0, engine="openpyxl")
        else:
            df = _read_json(file_bytes)
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file contains no data.") from exc
    except Exception as exc:
        raise DataValidationError(f"Could not parse the {ext} file: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise DataValidationError("The file was read successfully but contains no rows.")
    return df.reset_index(drop=True)


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV, trying common encodings and auto-detecting the delimiter."""
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DataValidationError(f"Could not decode the CSV file: {last_error}")


def _read_json(file_bytes: bytes) -> pd.DataFrame:
    """Read JSON: list of records, column-oriented dict, nested {'data': [...]}, or JSON Lines."""
    text = file_bytes.decode("utf-8-sig", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:  # JSON Lines
            return pd.read_json(io.StringIO(text), lines=True)
        except ValueError as exc:
            raise DataValidationError(f"Invalid JSON: {exc}") from exc

    if isinstance(payload, dict):
        # {"data": [...records...]} or {"records": [...]}
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.json_normalize(value)
        return pd.DataFrame(payload)
    if isinstance(payload, list):
        return pd.json_normalize(payload) if payload and isinstance(payload[0], dict) else pd.DataFrame(payload)
    raise DataValidationError("JSON must contain a list of records or an object of columns.")


# --------------------------------------------------------------------------- #
# 2. Validation
# --------------------------------------------------------------------------- #
def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CANONICAL = {_key(c): c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and map variants like 'order_id' / 'ORDER ID' / 'sub category'
    to the canonical names used by the dashboard. Unknown columns are only stripped.
    """
    renamed = {c: _CANONICAL.get(_key(c), str(c).strip()) for c in df.columns}
    out = df.rename(columns=renamed)
    # If two source columns collapsed to the same name keep the first one.
    return out.loc[:, ~out.columns.duplicated()].copy()


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of required columns that are missing from *df*."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def validate_data(df: pd.DataFrame) -> dict:
    """
    Describe the uploaded (raw) data.

    Returns a dict with: rows, columns, column_names, schema (DataFrame with
    dtype / non-null / missing / missing % / unique per column), missing_total,
    duplicate_count and duplicate_rows (DataFrame).
    """
    n_rows = len(df)
    schema = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null": df.notna().sum().values,
            "Missing": df.isna().sum().values,
            "Missing %": (df.isna().sum().values / max(n_rows, 1) * 100).round(2),
            "Unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    dup_mask = df.duplicated(keep=False)
    return {
        "rows": n_rows,
        "columns": df.shape[1],
        "column_names": list(df.columns.astype(str)),
        "schema": schema,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "duplicate_rows": df[dup_mask].sort_values(list(df.columns[:1])) if dup_mask.any() else df.iloc[0:0],
    }


def detect_type_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find numeric-looking columns that hold non-numeric entries (e.g. 'unknown', '?').
    Returns a DataFrame: Column, Bad Values, Examples.
    """
    rows = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        converted = to_numeric_clean(df[col], allow_percent=(col == "Discount"))
        bad = df[col].notna() & converted.isna()
        if bad.any():
            examples = ", ".join(map(str, pd.unique(df.loc[bad, col].astype(str))[:4]))
            rows.append({"Column": col, "Bad Values": int(bad.sum()), "Examples": examples})
    return pd.DataFrame(rows, columns=["Column", "Bad Values", "Examples"])


# --------------------------------------------------------------------------- #
# 3. Cleaning helpers
# --------------------------------------------------------------------------- #
def to_numeric_clean(series: pd.Series, allow_percent: bool = False) -> pd.Series:
    """
    Convert a column to float, tolerating currency symbols, thousands separators,
    accounting negatives '(12.5)', number words ('five') and, optionally,
    percentages ('ten percent', '15%'). Anything unparseable or infinite -> NaN.
    """
    if is_bool_dtype(series):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    if is_numeric_dtype(series):
        return series.astype("float64").replace([np.inf, -np.inf], np.nan)

    def convert(value) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value) if np.isfinite(value) else np.nan
        text = str(value).strip().lower()
        if not text:
            return np.nan

        pct = re.fullmatch(r"([a-z\-]+|-?[\d.]+)\s*(%|percent|pct)", text)
        if pct:
            if not allow_percent:
                return np.nan
            base = pct.group(1)
            number = _WORD_NUMBERS.get(base)
            if number is None:
                try:
                    number = float(base)
                except ValueError:
                    return np.nan
            return number / 100.0

        if text in _WORD_NUMBERS:
            return float(_WORD_NUMBERS[text])

        cleaned = re.sub(r"[,$€£₹\s]", "", text)
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        try:
            number = float(cleaned)
        except ValueError:
            return np.nan
        if not np.isfinite(number):
            return np.nan
        return -number if negative else number

    return pd.Series([convert(v) for v in series], index=series.index, dtype="float64")


def parse_dates(series: pd.Series, formats: Optional[list[str]] = None) -> pd.Series:
    """
    Parse a column containing dates in mixed formats into datetime64.
    Each format in *formats* is tried on the values still unparsed; ambiguous
    numeric dates (03-12-2025) are interpreted day-first. Failures -> NaT.
    """
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.tz_localize(None).dt.normalize() if getattr(series.dt, "tz", None) else series.dt.normalize()

    text = pd.Series(
        [None if pd.isna(v) else str(v).strip() for v in series], index=series.index, dtype="object"
    )
    text = text.where(~text.isin(["", "nan", "NaT", "None"]), None)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for fmt in formats or DATE_FORMATS:
        todo = parsed.isna() & text.notna()
        if not todo.any():
            break
        attempt = pd.to_datetime(text[todo], format=fmt, errors="coerce")
        parsed.loc[todo] = attempt.astype("datetime64[ns]")
    return parsed.dt.normalize()


def _trim(value):
    """Trim and collapse whitespace; missing -> None."""
    return None if pd.isna(value) else re.sub(r"\s+", " ", str(value)).strip()


def _harmonise_text(series: pd.Series) -> pd.Series:
    """
    Fix inconsistent labels: trim/collapse whitespace, turn placeholders into NaN
    and unify case/spacing variants ('SameDay', 'Same Day', 'same day') into the
    most frequent spelling.
    """
    cleaned = pd.Series(
        [None if pd.isna(v) else re.sub(r"\s+", " ", str(v)).strip() for v in series],
        index=series.index,
        dtype="object",
    )
    cleaned = cleaned.where(~cleaned.map(lambda v: v is None or bool(_PLACEHOLDER_RE.match(v))), None)

    groups = cleaned.dropna().map(lambda v: re.sub(r"[^a-z0-9]", "", v.lower()))
    canonical: dict[str, str] = {}
    for key, variants in cleaned.dropna().groupby(groups):
        counts = variants.value_counts()
        # prefer the most common spelling; on ties prefer one that is not all-lower / all-upper
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].islower() or kv[0].isupper(), kv[0]))[0][0]
        canonical[key] = best
    return cleaned.map(lambda v: canonical[re.sub(r"[^a-z0-9]", "", v.lower())] if v is not None else None)


def _fill_from_mapping(df: pd.DataFrame, child: str, parent: str, min_share: float = 0.6) -> int:
    """
    Repair *parent* values using the dominant parent for each *child*
    (e.g. State -> Region, Sub-Category -> Category). Returns rows changed.
    """
    if child not in df.columns or parent not in df.columns:
        return 0
    mapping = {}
    for key, grp in df.dropna(subset=[child, parent]).groupby(child):
        counts = grp[parent].value_counts()
        if counts.iloc[0] / counts.sum() >= min_share:
            mapping[key] = counts.index[0]
    target = df[child].map(mapping)
    change = target.notna() & (df[parent] != target)
    df.loc[change, parent] = target[change]
    return int(change.sum())


def flag_outliers_iqr(series: pd.Series, multiplier: float = 3.0) -> pd.Series:
    """Boolean mask of values outside [Q1 - k*IQR, Q3 + k*IQR] (NaN are never flagged)."""
    valid = series.dropna()
    if len(valid) < 8:
        return pd.Series(False, index=series.index)
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - multiplier * iqr) | (series > q3 + multiplier * iqr)


@dataclass
class CleaningConfig:
    """User-selectable cleaning behaviour."""

    sales_profit_strategy: str = "drop"   # 'drop' rows or 'impute' median for missing/invalid Sales & Profit
    treat_outliers: bool = True           # mark extreme Sales / Quantity outliers as invalid
    outlier_multiplier: float = 10.0      # IQR multiplier for Sales ("extreme" outliers only)
    quantity_outlier_multiplier: float = 3.0


def _group_median_fill(df: pd.DataFrame, col: str, group: Optional[str]) -> int:
    """Fill NaN in *col* with the median of its group (fallback: global median). Returns #filled."""
    missing = df[col].isna()
    if not missing.any() or df[col].notna().sum() == 0:
        return 0
    global_median = df[col].median()
    if group and group in df.columns:
        fill = df.groupby(group)[col].transform("median").fillna(global_median)
    else:
        fill = pd.Series(global_median, index=df.index)
    df.loc[missing, col] = fill[missing]
    return int(missing.sum())


def clean_data(raw: pd.DataFrame, config: Optional[CleaningConfig] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean a raw sales DataFrame.

    Returns
    -------
    (clean_df, log) where *log* is a DataFrame with columns
    Step / Action / Rows Affected describing every change made.
    """
    cfg = config or CleaningConfig()
    log: list[dict] = []

    def record(step: str, action: str, n: int) -> None:
        if n:
            log.append({"Step": step, "Action": action, "Rows Affected": int(n)})

    df = standardize_column_names(raw)
    missing_cols = check_required_columns(df)
    if missing_cols:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(missing_cols)
            + ". Expected at least: " + ", ".join(REQUIRED_COLUMNS) + "."
        )
    start_rows = len(df)

    # --- empty rows & exact duplicates ------------------------------------ #
    before = len(df)
    df = df.dropna(how="all")
    record("Missing values", "Removed completely empty rows", before - len(df))

    before = len(df)
    df = df.drop_duplicates()
    record("Duplicates", "Removed exact duplicate rows", before - len(df))

    # --- inconsistent text -------------------------------------------------- #
    for col in [c for c in CATEGORICAL_COLUMNS if c in df.columns]:
        trimmed = df[col].map(_trim)
        df[col] = _harmonise_text(df[col])
        changed = trimmed.notna() & df[col].notna() & (trimmed != df[col])
        record("Inconsistent data", f"Standardised spelling/case in '{col}'", changed.sum())
    df["Order ID"] = _harmonise_text(df["Order ID"])

    # --- dates --------------------------------------------------------------- #
    raw_order = df["Order Date"].copy()
    df["Order Date"] = parse_dates(df["Order Date"])
    non_iso = raw_order.notna() & ~raw_order.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}")
    record("Date formats", "Converted non-ISO order dates (e.g. dd/mm/yyyy, dd-Mon-yyyy)",
           (non_iso & df["Order Date"].notna()).sum())

    if "Ship Date" in df.columns:
        raw_ship = df["Ship Date"].copy()
        ship = parse_dates(raw_ship)
        non_iso_s = raw_ship.notna() & ~raw_ship.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}")
        record("Date formats", "Converted non-ISO ship dates", (non_iso_s & ship.notna()).sum())

        # Resolve day/month ambiguity: if the day-first reading gives an implausible
        # ship delay, try the month-first reading and keep it when it is plausible.
        delta = (ship - df["Order Date"]).dt.days
        suspicious = ship.notna() & df["Order Date"].notna() & ((delta < 0) | (delta > MAX_SHIP_DAYS))
        if suspicious.any():
            alt = parse_dates(raw_ship[suspicious], ["%m-%d-%Y", "%m/%d/%Y"])
            alt_delta = (alt - df.loc[suspicious, "Order Date"]).dt.days
            fixed = alt.notna() & (alt_delta >= 0) & (alt_delta <= MAX_SHIP_DAYS)
            ship = ship.copy()
            fixed_idx = fixed[fixed].index
            ship.loc[fixed_idx] = alt.loc[fixed_idx]
            record("Date formats", "Re-read ambiguous ship dates as month-first (implausible as day-first)", fixed.sum())
        delta = (ship - df["Order Date"]).dt.days
        impossible = ship.notna() & df["Order Date"].notna() & (delta < 0)
        ship = ship.mask(impossible)
        record("Inconsistent data", "Ship date before order date -> set to missing", impossible.sum())
        df["Ship Date"] = ship
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # --- numbers ------------------------------------------------------------- #
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        original = df[col].copy()
        df[col] = to_numeric_clean(original, allow_percent=(col == "Discount"))
        record("Inconsistent data", f"Non-numeric text in '{col}' (e.g. '?', 'unknown') -> missing",
               (original.notna() & df[col].isna()).sum())

    # Re-check duplicates now that text/dates/numbers are standardised.
    before = len(df)
    df = df.drop_duplicates()
    record("Duplicates", "Removed duplicates revealed after standardisation", before - len(df))

    # --- rows we cannot use ---------------------------------------------------- #
    before = len(df)
    df = df[df["Order ID"].notna()]
    record("Missing values", "Dropped rows without an Order ID (cannot be counted as an order)", before - len(df))

    before = len(df)
    df = df[df["Order Date"].notna()]
    record("Missing values", "Dropped rows with a missing/unparseable Order Date", before - len(df))

    # --- derive missing categorical info from related columns ------------------ #
    n = _fill_from_mapping(df, "State", "Region")
    record("Inconsistent data", "Region corrected to match the State's dominant Region", n)
    n = _fill_from_mapping(df, "Sub-Category", "Category")
    record("Inconsistent data", "Category corrected to match the Sub-Category's dominant Category", n)

    for col in [c for c in CATEGORICAL_COLUMNS if c in df.columns]:
        n = df[col].isna().sum()
        df[col] = df[col].fillna("Unknown")
        record("Missing values", f"Filled missing '{col}' with 'Unknown'", n)

    # --- invalid values & outliers ----------------------------------------------- #
    if "Discount" in df.columns:
        invalid = (df["Discount"] < 0) | (df["Discount"] > 1)
        df.loc[invalid, "Discount"] = np.nan
        record("Inconsistent data", "Discount outside 0-100% -> missing", invalid.sum())

    invalid_q = (df["Quantity"] <= 0) | (df["Quantity"] % 1 != 0)
    df.loc[invalid_q, "Quantity"] = np.nan
    record("Inconsistent data", "Quantity <= 0 or not a whole number -> missing", invalid_q.sum())

    invalid_s = df["Sales"] <= 0
    df.loc[invalid_s, "Sales"] = np.nan
    record("Inconsistent data", "Sales <= 0 -> missing", invalid_s.sum())

    if cfg.treat_outliers:
        out_q = flag_outliers_iqr(df["Quantity"], cfg.quantity_outlier_multiplier)
        df.loc[out_q, "Quantity"] = np.nan
        record("Outliers", f"Quantity outliers (> {cfg.quantity_outlier_multiplier:g} x IQR) -> missing", out_q.sum())
        out_s = flag_outliers_iqr(df["Sales"], cfg.outlier_multiplier)
        df.loc[out_s, "Sales"] = np.nan
        record("Outliers", f"Extreme Sales outliers (> {cfg.outlier_multiplier:g} x IQR) -> missing", out_s.sum())

    # Profit cannot exceed revenue; large profits/losses are otherwise legitimate, so no IQR rule here.
    impossible_p = df["Profit"] > df["Sales"]
    df.loc[impossible_p, "Profit"] = np.nan
    record("Inconsistent data", "Profit greater than Sales (impossible) -> missing", impossible_p.sum())

    # --- missing numeric values ------------------------------------------------------ #
    group = "Sub-Category" if "Sub-Category" in df.columns else None
    if cfg.sales_profit_strategy == "impute":
        for col in ("Sales", "Profit"):
            record("Missing values", f"Imputed '{col}' with Sub-Category median", _group_median_fill(df, col, group))
    else:
        before = len(df)
        df = df.dropna(subset=["Sales", "Profit"])
        record("Missing values", "Dropped rows with missing/invalid Sales or Profit", before - len(df))

    n = _group_median_fill(df, "Quantity", group)
    record("Missing values", "Imputed 'Quantity' with Sub-Category median", n)
    if "Discount" in df.columns:
        n = _group_median_fill(df, "Discount", group)
        record("Missing values", "Imputed 'Discount' with Sub-Category median", n)

    if df.empty:
        raise DataValidationError("No usable rows remain after cleaning. Check your data or cleaning options.")

    # --- final types & derived columns ----------------------------------------------- #
    df["Quantity"] = df["Quantity"].round().astype(int)
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["Order Date"].dt.year
    df = df.sort_values("Order Date").reset_index(drop=True)

    log.append({"Step": "Summary", "Action": f"Rows: {start_rows} raw -> {len(df)} clean", "Rows Affected": start_rows - len(df)})
    return df, pd.DataFrame(log, columns=["Step", "Action", "Rows Affected"])


# --------------------------------------------------------------------------- #
# 4. Statistics, KPIs and aggregations
# --------------------------------------------------------------------------- #
def compute_kpis(df: pd.DataFrame) -> dict:
    """The five dashboard KPIs for the (possibly filtered) data."""
    total_sales = float(df["Sales"].sum()) if len(df) else 0.0
    total_profit = float(df["Profit"].sum()) if len(df) else 0.0
    return {
        "Total Sales": total_sales,
        "Total Profit": total_profit,
        "Total Orders": int(df["Order ID"].nunique()) if len(df) else 0,
        "Total Quantity Sold": int(df["Quantity"].sum()) if len(df) else 0,
        "Profit Margin (%)": (total_profit / total_sales * 100.0) if total_sales else 0.0,
    }


def numeric_summary(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Count, mean, median, std, min, quartiles, max, IQR, skew and CV for numeric columns."""
    cols = [c for c in (columns or NUMERIC_COLUMNS + ["Shipping Days"]) if c in df.columns]
    if not cols or df.empty:
        return pd.DataFrame()
    desc = df[cols].describe().T
    out = pd.DataFrame(
        {
            "Count": desc["count"],
            "Mean": desc["mean"],
            "Median": df[cols].median(),
            "Std Dev": desc["std"],
            "Min": desc["min"],
            "Q1": desc["25%"],
            "Q3": desc["75%"],
            "Max": desc["max"],
            "IQR": desc["75%"] - desc["25%"],
            "Skew": df[cols].skew(),
            "CV (%)": (desc["std"] / desc["mean"].replace(0, np.nan) * 100),
        }
    )
    return out.round(3)


def categorical_summary(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Unique count, most frequent value and its share for categorical columns."""
    cols = [c for c in (columns or ["Ship Mode", "Segment", "Region", "State", "Category", "Sub-Category"]) if c in df.columns]
    rows = []
    for col in cols:
        counts = df[col].value_counts()
        if counts.empty:
            continue
        rows.append(
            {
                "Column": col,
                "Unique": int(counts.size),
                "Most Frequent": counts.index[0],
                "Frequency": int(counts.iloc[0]),
                "Share (%)": round(counts.iloc[0] / counts.sum() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between the numeric measures."""
    cols = [c for c in ["Sales", "Quantity", "Discount", "Profit"] if c in df.columns]
    return df[cols].corr().round(3) if len(cols) > 1 and len(df) > 1 else pd.DataFrame()


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Sales, Profit, Quantity, Orders and Profit Margin grouped by one column."""
    if by not in df.columns or df.empty:
        return pd.DataFrame()
    g = df.groupby(by).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"), Orders=("Order ID", "nunique"),
    )
    g["Profit Margin (%)"] = (g["Profit"] / g["Sales"].replace(0, np.nan) * 100).round(2)
    return g.sort_values("Sales", ascending=False).round(2).reset_index()


def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[tuple] = None,
    selections: Optional[dict[str, list]] = None,
) -> pd.DataFrame:
    """
    Filter by an inclusive (start, end) Order Date range and by
    {column: [allowed values]} selections.
    """
    mask = pd.Series(True, index=df.index)
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= df["Order Date"].between(start, end)
    for col, allowed in (selections or {}).items():
        if col in df.columns:
            mask &= df[col].isin(allowed)
    return df[mask]
