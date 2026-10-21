"""Unit tests for data_processing.py  (run with: pytest -q)"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_processing as dp  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "experiment10_raw_dataset.csv"


@pytest.fixture(scope="module")
def raw():
    return dp.load_file(SAMPLE.read_bytes(), SAMPLE.name)


@pytest.fixture(scope="module")
def cleaned(raw):
    return dp.clean_data(raw)


# ---------------------------------------------------------------- loading ---
def test_load_csv_shape(raw):
    assert raw.shape == (186, 18)


def test_load_excel_and_json_roundtrip(raw):
    small = raw.head(10)
    buf = io.BytesIO()
    small.to_excel(buf, index=False)
    assert dp.load_file(buf.getvalue(), "a.xlsx").shape == small.shape
    assert dp.get_excel_sheets(buf.getvalue()) == ["Sheet1"]

    as_records = small.to_json(orient="records").encode()
    assert dp.load_file(as_records, "a.json").shape == small.shape
    nested = json.dumps({"data": json.loads(small.to_json(orient="records"))}).encode()
    assert dp.load_file(nested, "a.json").shape == small.shape
    lines = small.to_json(orient="records", lines=True).encode()
    assert dp.load_file(lines, "a.json").shape == small.shape


@pytest.mark.parametrize("payload,name", [(b"", "a.csv"), (b"a,b\n1,2", "a.txt"), (b"{bad json", "a.json"), (b"not an excel file", "a.xlsx")])
def test_load_errors_are_friendly(payload, name):
    with pytest.raises(dp.DataValidationError):
        dp.load_file(payload, name)


def test_csv_header_only_is_rejected():
    with pytest.raises(dp.DataValidationError):
        dp.load_file(b"Order ID,Sales\n", "a.csv")


# ------------------------------------------------------------- validation ---
def test_validate_data(raw):
    info = dp.validate_data(raw)
    assert (info["rows"], info["columns"]) == (186, 18)
    assert info["duplicate_count"] == 4
    assert info["missing_total"] == int(raw.isna().sum().sum())
    assert set(info["schema"].columns) >= {"Column", "Data Type", "Missing", "Missing %"}
    assert len(info["schema"]) == 18


def test_detect_type_issues(raw):
    issues = dp.detect_type_issues(raw).set_index("Column")
    assert issues.loc["Sales", "Bad Values"] == 2
    assert issues.loc["Profit", "Bad Values"] == 2
    assert issues.loc["Quantity", "Bad Values"] == 1


def test_standardize_column_names():
    df = pd.DataFrame(columns=[" order_id ", "ORDER DATE", "sub category", "Sales", "extra"])
    assert list(dp.standardize_column_names(df).columns) == ["Order ID", "Order Date", "Sub-Category", "Sales", "extra"]


def test_missing_required_columns_detected_and_raised():
    df = pd.DataFrame({"Order ID": ["A"], "Sales": [1]})
    assert set(dp.check_required_columns(df)) == {"Order Date", "Quantity", "Profit"}
    with pytest.raises(dp.DataValidationError):
        dp.clean_data(df)


# ----------------------------------------------------------- helper units ---
@pytest.mark.parametrize("value,expected", [
    ("12.5", 12.5), ("$1,200.50", 1200.5), ("(15)", -15), ("five", 5), (7, 7), ("?", np.nan),
    ("inf", np.nan), ("unknown", np.nan), (None, np.nan), ("-3", -3),
])
def test_to_numeric_clean(value, expected):
    result = dp.to_numeric_clean(pd.Series([value], dtype="object")).iloc[0]
    assert (np.isnan(result) and np.isnan(expected)) or result == expected


def test_to_numeric_percent_only_when_allowed():
    s = pd.Series(["ten percent", "15%", "0.2"], dtype="object")
    assert dp.to_numeric_clean(s, allow_percent=True).tolist() == [0.1, 0.15, 0.2]
    assert dp.to_numeric_clean(s).isna().tolist() == [True, True, False]


def test_parse_dates_mixed_formats():
    s = pd.Series(["2025-09-07", "26/12/2024", "21-Mar-2025", "03-12-2025", "not-a-date", None], dtype="object")
    out = dp.parse_dates(s)
    assert out.iloc[0] == pd.Timestamp("2025-09-07")
    assert out.iloc[1] == pd.Timestamp("2024-12-26")
    assert out.iloc[2] == pd.Timestamp("2025-03-21")
    assert out.iloc[3] == pd.Timestamp("2025-12-03")  # day-first by design
    assert out.iloc[4:].isna().all()


def test_flag_outliers_iqr():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10000.0])
    assert dp.flag_outliers_iqr(s, 3).tolist() == [False] * 9 + [True]
    assert not dp.flag_outliers_iqr(pd.Series([1.0, 2.0])).any()  # too few points


def test_harmonise_text_merges_variants():
    s = pd.Series(["Same Day"] * 5 + ["SameDay", " same day ", "Standard Class", "Standard class", "N/A", None], dtype="object")
    out = dp._harmonise_text(s)
    assert set(out.dropna()) == {"Same Day", "Standard Class"}
    assert out.isna().sum() == 2


# --------------------------------------------------------------- cleaning ---
def test_cleaning_output_is_valid(cleaned):
    df, log = cleaned
    assert not df.duplicated().any()
    for col in ["Order ID", "Order Date", "Sales", "Profit", "Quantity", "Discount"]:
        assert df[col].notna().all(), col
    assert (df["Sales"] > 0).all() and (df["Quantity"] > 0).all()
    assert df["Discount"].between(0, 1).all()
    assert (df["Profit"] <= df["Sales"]).all()
    assert df["Sales"].max() < 100_000 and df["Quantity"].max() <= 10
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert (df["Shipping Days"].dropna() >= 0).all() and df["Shipping Days"].max() <= dp.MAX_SHIP_DAYS
    assert not log.empty and {"Step", "Action", "Rows Affected"} == set(log.columns)


def test_cleaning_standardises_labels(cleaned):
    df, _ = cleaned
    assert set(df["Region"]) == {"East", "West", "South", "Central"}
    assert set(df["Category"]) == {"Furniture", "Office Supplies", "Technology"}
    assert set(df["Segment"]) <= {"Consumer", "Corporate", "Home Office", "Unknown"}
    assert set(df["Ship Mode"]) <= {"First Class", "Second Class", "Standard Class", "Same Day", "Unknown"}


def test_cleaning_fixes_ambiguous_ship_date(cleaned):
    """03-12-2025 for an order on 2025-03-08 must be read as 12 March, not 3 December."""
    df, _ = cleaned
    assert df["Shipping Days"].max() <= dp.MAX_SHIP_DAYS


def test_cleaning_does_not_remove_legit_large_profit(cleaned):
    df, _ = cleaned
    assert np.isclose(df["Profit"].max(), 854.09)


def test_junk_rows_removed(cleaned):
    df, _ = cleaned
    assert "Testville" not in set(df["City"])
    assert "Test State" not in set(df["State"])


def test_impute_strategy_keeps_more_rows(raw):
    dropped, _ = dp.clean_data(raw, dp.CleaningConfig(sales_profit_strategy="drop"))
    imputed, _ = dp.clean_data(raw, dp.CleaningConfig(sales_profit_strategy="impute"))
    assert len(imputed) > len(dropped)
    assert imputed[["Sales", "Profit"]].notna().all().all()


def test_outlier_toggle(raw):
    kept, _ = dp.clean_data(raw, dp.CleaningConfig(treat_outliers=False))
    assert kept["Sales"].max() > 1e6


def test_all_rows_invalid_raises():
    df = pd.DataFrame({"Order ID": ["A"], "Order Date": ["bad"], "Sales": [1], "Quantity": [1], "Profit": [1]})
    with pytest.raises(dp.DataValidationError):
        dp.clean_data(df)


def test_clean_works_on_native_dtypes():
    """JSON / Excel give real numbers and datetimes; cleaning must handle them too."""
    df = pd.DataFrame({
        "order_id": ["A", "B", "B"], "order_date": pd.to_datetime(["2025-01-05", "2025-02-10", "2025-02-11"]),
        "sales": [100.0, 250.5, 80.0], "quantity": [1, 2, 3], "profit": [10.0, -5.0, 8.0],
    })
    out, _ = dp.clean_data(df)
    assert len(out) == 3 and out["Order ID"].nunique() == 2


# ------------------------------------------------------------ KPIs & stats ---
def test_compute_kpis_known_values():
    df = pd.DataFrame({"Order ID": ["A", "A", "B"], "Sales": [100.0, 50.0, 50.0],
                       "Profit": [20.0, 10.0, -10.0], "Quantity": [1, 2, 3]})
    k = dp.compute_kpis(df)
    assert k["Total Sales"] == 200 and k["Total Profit"] == 20
    assert k["Total Orders"] == 2          # unique Order IDs, not rows
    assert k["Total Quantity Sold"] == 6
    assert k["Profit Margin (%)"] == pytest.approx(10.0)


def test_compute_kpis_empty_is_safe():
    k = dp.compute_kpis(pd.DataFrame(columns=["Order ID", "Sales", "Profit", "Quantity"]))
    assert k["Total Orders"] == 0 and k["Profit Margin (%)"] == 0.0


def test_apply_filters(cleaned):
    df, _ = cleaned
    east = dp.apply_filters(df, selections={"Region": ["East"]})
    assert set(east["Region"]) == {"East"} and 0 < len(east) < len(df)
    ranged = dp.apply_filters(df, date_range=("2025-01-01", "2025-03-31"))
    assert ranged["Order Date"].between("2025-01-01", "2025-03-31").all()
    assert dp.apply_filters(df, selections={"Region": []}).empty
    both = dp.apply_filters(df, ("2025-01-01", "2025-12-31"), {"Category": ["Technology"], "Region": ["West"]})
    assert set(both["Category"]) <= {"Technology"}


def test_kpis_change_with_filters(cleaned):
    df, _ = cleaned
    assert dp.compute_kpis(dp.apply_filters(df, selections={"Region": ["East"]}))["Total Sales"] < dp.compute_kpis(df)["Total Sales"]


def test_summaries(cleaned):
    df, _ = cleaned
    num = dp.numeric_summary(df)
    assert {"Mean", "Median", "Std Dev", "IQR", "Skew"} <= set(num.columns) and "Sales" in num.index
    assert num.loc["Sales", "Count"] == len(df)
    cat = dp.categorical_summary(df)
    assert "Region" in set(cat["Column"])
    corr = dp.correlation_matrix(df)
    assert corr.shape == (4, 4) and np.allclose(np.diag(corr), 1)
    grp = dp.group_summary(df, "Region")
    assert grp["Sales"].sum() == pytest.approx(df["Sales"].sum(), rel=1e-3)
    assert dp.group_summary(df, "nope").empty
