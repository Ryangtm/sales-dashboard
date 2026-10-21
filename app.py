"""
Sales Analytics Dashboard  -  Streamlit application
====================================================
Upload a CSV / Excel / JSON sales file, validate it, clean it, filter it and
explore KPIs and interactive charts.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import charts
import data_processing as dp

SAMPLE_PATH = Path(__file__).parent / "data" / "experiment10_raw_dataset.csv"

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Cached wrappers (so widgets do not re-read / re-clean the file on every click)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading file…")
def cached_load(file_bytes: bytes, filename: str, sheet: Optional[str]) -> pd.DataFrame:
    return dp.load_file(file_bytes, filename, sheet)


@st.cache_data(show_spinner="Cleaning data…")
def cached_clean(raw: pd.DataFrame, strategy: str, treat_outliers: bool):
    config = dp.CleaningConfig(sales_profit_strategy=strategy, treat_outliers=treat_outliers)
    return dp.clean_data(raw, config)


@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar: upload, cleaning options, filters
# --------------------------------------------------------------------------- #
def sidebar_data_source() -> Optional[tuple[bytes, str, Optional[str]]]:
    """Render the upload widgets; return (bytes, filename, sheet) or None."""
    st.sidebar.header("1 · Data")
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV, Excel (.xlsx) or JSON file", type=["csv", "xlsx", "json"],
        help="The file needs at least: Order ID, Order Date, Sales, Quantity, Profit.",
    )
    use_sample = st.sidebar.toggle(
        "Use bundled sample dataset", value=True, disabled=uploaded is not None,
        help="Used only when no file is uploaded.",
    )

    if uploaded is not None:
        file_bytes, name = uploaded.getvalue(), uploaded.name
    elif use_sample and SAMPLE_PATH.exists():
        file_bytes, name = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
    else:
        return None

    sheet = None
    if name.lower().endswith(".xlsx"):
        try:
            sheets = dp.get_excel_sheets(file_bytes)
        except dp.DataValidationError as exc:
            st.error(str(exc))
            st.stop()
        sheet = st.sidebar.selectbox("Excel sheet", sheets) if len(sheets) > 1 else sheets[0]
    return file_bytes, name, sheet


def sidebar_cleaning_options() -> tuple[str, bool]:
    st.sidebar.header("2 · Cleaning")
    strategy_label = st.sidebar.radio(
        "Missing / invalid Sales or Profit",
        ["Drop those rows", "Impute with Sub-Category median"],
        help="Quantity and Discount are always imputed with the Sub-Category median.",
    )
    treat_outliers = st.sidebar.checkbox(
        "Treat extreme outliers as invalid", value=True,
        help="Removes absurd values such as a Quantity of 10,000 or a single Sales value of $10 million.",
    )
    return ("drop" if strategy_label.startswith("Drop") else "impute"), treat_outliers


def _reset_filters() -> None:
    st.session_state["filter_version"] = st.session_state.get("filter_version", 0) + 1


def sidebar_filters(df: pd.DataFrame, dataset_id: str) -> tuple[pd.DataFrame, bool]:
    """Render the filters; return (filtered_df, date_range_complete)."""
    st.sidebar.header("3 · Filters")
    st.sidebar.button("↺ Reset filters", on_click=_reset_filters, width="stretch")
    version = st.session_state.get("filter_version", 0)
    key = lambda name: f"{name}|{dataset_id}|{version}"  # noqa: E731 - new key => widget resets

    min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
    date_range = st.sidebar.date_input(
        "Order date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key=key("date"),
    )
    complete = isinstance(date_range, (tuple, list)) and len(date_range) == 2
    if not complete:
        st.sidebar.info("Pick an end date to apply the date filter.")

    selections: dict[str, list] = {}
    for label, col in [("Region", "Region"), ("Category", "Category"), ("Segment", "Segment"), ("Ship mode", "Ship Mode")]:
        if col in df.columns:
            options = sorted(df[col].dropna().unique())
            selections[col] = st.sidebar.multiselect(label, options, default=options, key=key(col))
    if "State" in df.columns:
        with st.sidebar.expander("More filters"):
            options = sorted(df["State"].dropna().unique())
            selections["State"] = st.multiselect("State", options, default=options, key=key("State"))

    filtered = dp.apply_filters(df, date_range if complete else None, selections)
    return filtered, complete


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def render_validation(raw: pd.DataFrame, filename: str) -> None:
    info = dp.validate_data(raw)
    st.subheader(f"Uploaded data · {filename}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info['rows']:,}", border=True)
    c2.metric("Columns", info["columns"], border=True)
    c3.metric("Missing cells", f"{info['missing_total']:,}", border=True)
    c4.metric("Duplicate rows", info["duplicate_count"], border=True)

    st.markdown("**Column names, data types and missing values**")
    st.dataframe(
        info["schema"], hide_index=True, width="stretch",
        column_config={"Missing %": st.column_config.ProgressColumn("Missing %", format="%.1f%%", min_value=0, max_value=100)},
    )

    issues = dp.detect_type_issues(dp.standardize_column_names(raw))
    if not issues.empty:
        st.warning("Numeric columns containing non-numeric text were found. They will be converted to missing values during cleaning.")
        st.dataframe(issues, hide_index=True, width="stretch")

    if info["duplicate_count"]:
        with st.expander(f"Show duplicate records ({info['duplicate_count']} extra copies)"):
            st.dataframe(info["duplicate_rows"], width="stretch")
    else:
        st.success("No duplicate records found.")

    with st.expander("Preview of the raw data (first 20 rows)"):
        st.dataframe(raw.head(20), width="stretch")


def render_cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame, log: pd.DataFrame) -> None:
    st.subheader("What the cleaning step did")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{len(raw):,}", border=True)
    c2.metric("Clean rows", f"{len(clean):,}", border=True)
    c3.metric("Rows removed", f"{len(raw) - len(clean):,}", border=True)

    steps = ["All"] + sorted(log["Step"].unique())
    chosen = st.selectbox("Filter by step", steps)
    view = log if chosen == "All" else log[log["Step"] == chosen]
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption(
        "Orders may span several rows (one per product line), so repeated Order IDs are kept; "
        "only identical full-row duplicates are removed. Ambiguous numeric dates are read day-first."
    )
    st.download_button("⬇️ Download cleaned data (CSV)", to_csv_bytes(clean), "cleaned_sales_data.csv", "text/csv")


def render_kpis(filtered: pd.DataFrame, full: pd.DataFrame) -> None:
    k = dp.compute_kpis(filtered)
    base = dp.compute_kpis(full)
    is_filtered = len(filtered) != len(full)
    margin_delta = f"{k['Profit Margin (%)'] - base['Profit Margin (%)']:+.2f} pp vs all data" if is_filtered and len(filtered) else None

    cols = st.columns(5)
    cols[0].metric("Total Sales", fmt_money(k["Total Sales"]), border=True)
    cols[1].metric("Total Profit", fmt_money(k["Total Profit"]), border=True)
    cols[2].metric("Total Orders", f"{k['Total Orders']:,}", help="Unique Order IDs", border=True)
    cols[3].metric("Total Quantity Sold", f"{k['Total Quantity Sold']:,}", border=True)
    cols[4].metric("Profit Margin", f"{k['Profit Margin (%)']:.2f}%", delta=margin_delta, border=True,
                   help="Total Profit ÷ Total Sales × 100")
    st.caption(f"Showing **{len(filtered):,}** of **{len(full):,}** cleaned records ({len(filtered) / len(full):.0%}).")


def plot(fig, name: str) -> None:
    if fig is None:
        st.info(f"'{name}' needs columns that are not present in this dataset.")
    else:
        st.plotly_chart(fig, width="stretch")


def render_dashboard(filtered: pd.DataFrame, full: pd.DataFrame) -> None:
    render_kpis(filtered, full)
    if filtered.empty:
        st.warning("No records match the current filters. Widen the filters in the sidebar.")
        return

    plot(charts.monthly_trend(filtered), "Monthly trend")
    left, right = st.columns(2)
    with left:
        plot(charts.category_performance(filtered), "Category chart")
    with right:
        plot(charts.region_share(filtered), "Region chart")
    left, right = st.columns(2)
    with left:
        plot(charts.subcategory_profit(filtered), "Sub-category chart")
    with right:
        plot(charts.discount_vs_profit(filtered), "Discount vs Profit")
    left, right = st.columns(2)
    with left:
        plot(charts.top_products(filtered), "Top products")
    with right:
        plot(charts.margin_heatmap(filtered), "Margin heatmap")


def render_statistics(filtered: pd.DataFrame) -> None:
    st.caption("Statistics reflect the current sidebar filters.")
    if filtered.empty:
        st.warning("No records match the current filters.")
        return

    st.markdown("**Numeric summary**")
    st.dataframe(dp.numeric_summary(filtered), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Categorical summary**")
        st.dataframe(dp.categorical_summary(filtered), hide_index=True, width="stretch")
    with right:
        plot(charts.correlation_heatmap(dp.correlation_matrix(filtered)), "Correlation matrix")

    st.markdown("**Group summary**")
    options = [c for c in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode", "State", "Year"] if c in filtered.columns]
    if options:
        by = st.selectbox("Group by", options)
        st.dataframe(dp.group_summary(filtered, by), hide_index=True, width="stretch")


def render_explorer(filtered: pd.DataFrame) -> None:
    st.caption(f"{len(filtered):,} records after cleaning and filtering.")
    st.dataframe(filtered, hide_index=True, width="stretch")
    if not filtered.empty:
        st.download_button("⬇️ Download filtered data (CSV)", to_csv_bytes(filtered), "filtered_sales_data.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_welcome() -> None:
    st.title("📊 Sales Analytics Dashboard")
    st.info("👈 Upload a CSV, Excel or JSON file in the sidebar (or switch on the sample dataset) to begin.")
    st.markdown(
        "**Required columns:** " + ", ".join(f"`{c}`" for c in dp.REQUIRED_COLUMNS)
        + "\n\n**Optional columns** (enable more filters and charts): "
        + ", ".join(f"`{c}`" for c in dp.OPTIONAL_COLUMNS)
    )


def main() -> None:
    source = sidebar_data_source()
    if source is None:
        render_welcome()
        return
    file_bytes, filename, sheet = source
    strategy, treat_outliers = sidebar_cleaning_options()

    st.title("📊 Sales Analytics Dashboard")

    try:
        raw = cached_load(file_bytes, filename, sheet)
    except dp.DataValidationError as exc:
        st.error(f"❌ {exc}")
        return

    missing = dp.check_required_columns(dp.standardize_column_names(raw))
    if missing:
        st.error(f"❌ Missing required column(s): **{', '.join(missing)}**. Columns found: {', '.join(map(str, raw.columns))}")
        render_validation(raw, filename)
        return

    try:
        clean, log = cached_clean(raw, strategy, treat_outliers)
    except dp.DataValidationError as exc:
        st.error(f"❌ {exc}")
        render_validation(raw, filename)
        return

    dataset_id = f"{filename}-{len(file_bytes)}-{sheet}-{strategy}-{treat_outliers}"
    filtered, _ = sidebar_filters(clean, dataset_id)

    tab_overview, tab_cleaning, tab_dashboard, tab_stats, tab_data = st.tabs(
        ["📁 Data Overview", "🧹 Cleaning Report", "📊 Dashboard", "📈 Statistics", "🗂️ Data Explorer"]
    )
    with tab_overview:
        render_validation(raw, filename)
    with tab_cleaning:
        render_cleaning_report(raw, clean, log)
    with tab_dashboard:
        render_dashboard(filtered, clean)
    with tab_stats:
        render_statistics(filtered)
    with tab_data:
        render_explorer(filtered)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # last-resort guard so users never see a raw traceback
        st.error("Something unexpected went wrong. Try another file or reload the page.")
        with st.expander("Technical details"):
            st.exception(exc)
