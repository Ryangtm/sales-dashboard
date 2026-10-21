"""
charts.py
=========
Plotly figure builders. Every function takes the (filtered) clean DataFrame and
returns a Plotly Figure, or None when a required column is missing / data is empty,
so the UI can show a friendly message instead of crashing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = px.colors.qualitative.Set2
POSITIVE, NEGATIVE = "#2E8B57", "#D64545"


def _ok(df: pd.DataFrame, *cols: str) -> bool:
    return not df.empty and all(c in df.columns for c in cols)


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        hovermode="closest",
    )
    return fig


def monthly_trend(df: pd.DataFrame) -> Optional[go.Figure]:
    """Line chart: Sales and Profit per month."""
    if not _ok(df, "Order Month", "Sales", "Profit"):
        return None
    monthly = df.groupby("Order Month", as_index=False)[["Sales", "Profit"]].sum()
    long = monthly.melt("Order Month", var_name="Measure", value_name="Amount")
    fig = px.line(long, x="Order Month", y="Amount", color="Measure", markers=True,
                  title="Monthly Sales & Profit Trend", color_discrete_sequence=PALETTE)
    fig.update_xaxes(title="Month", rangeslider_visible=True)
    fig.update_yaxes(title="Amount ($)", tickprefix="$")
    return _style(fig, 430)


def category_performance(df: pd.DataFrame) -> Optional[go.Figure]:
    """Grouped bars: Sales vs Profit by Category."""
    if not _ok(df, "Category", "Sales", "Profit"):
        return None
    agg = df.groupby("Category", as_index=False)[["Sales", "Profit"]].sum()
    long = agg.melt("Category", var_name="Measure", value_name="Amount")
    fig = px.bar(long, x="Category", y="Amount", color="Measure", barmode="group",
                 title="Sales & Profit by Category", color_discrete_sequence=PALETTE, text_auto=".2s")
    fig.update_yaxes(title="Amount ($)", tickprefix="$")
    return _style(fig)


def region_share(df: pd.DataFrame) -> Optional[go.Figure]:
    """Donut: share of Sales by Region."""
    if not _ok(df, "Region", "Sales"):
        return None
    agg = df.groupby("Region", as_index=False)["Sales"].sum()
    fig = px.pie(agg, names="Region", values="Sales", hole=0.5, title="Sales Share by Region",
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _style(fig)


def subcategory_profit(df: pd.DataFrame) -> Optional[go.Figure]:
    """Horizontal bars: Profit by Sub-Category (green = profit, red = loss)."""
    if not _ok(df, "Sub-Category", "Profit"):
        return None
    agg = df.groupby("Sub-Category", as_index=False)["Profit"].sum().sort_values("Profit")
    colors = np.where(agg["Profit"] >= 0, POSITIVE, NEGATIVE)
    fig = go.Figure(go.Bar(x=agg["Profit"], y=agg["Sub-Category"], orientation="h", marker_color=colors,
                           hovertemplate="%{y}: $%{x:,.2f}<extra></extra>"))
    fig.update_layout(title="Profit by Sub-Category")
    fig.update_xaxes(title="Profit ($)", tickprefix="$")
    return _style(fig, 430)


def discount_vs_profit(df: pd.DataFrame) -> Optional[go.Figure]:
    """Scatter: Discount vs Profit, sized by Sales, coloured by Category."""
    if not _ok(df, "Discount", "Profit", "Sales"):
        return None
    hover = [c for c in ("Product Name", "Order ID", "Region") if c in df.columns]
    fig = px.scatter(df, x="Discount", y="Profit", size="Sales", color="Category" if "Category" in df else None,
                     hover_data=hover, title="Discount vs Profit (bubble size = Sales)",
                     color_discrete_sequence=PALETTE, opacity=0.75, size_max=28)
    fig.update_xaxes(title="Discount", tickformat=".0%")
    fig.update_yaxes(title="Profit ($)", tickprefix="$")
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    return _style(fig, 430)


def top_products(df: pd.DataFrame, n: int = 10) -> Optional[go.Figure]:
    """Horizontal bars: top-n products by Sales."""
    if not _ok(df, "Product Name", "Sales"):
        return None
    agg = df.groupby("Product Name", as_index=False)["Sales"].sum().nlargest(n, "Sales").sort_values("Sales")
    fig = px.bar(agg, x="Sales", y="Product Name", orientation="h", title=f"Top {n} Products by Sales",
                 color="Sales", color_continuous_scale="Blues")
    fig.update_xaxes(title="Sales ($)", tickprefix="$")
    fig.update_coloraxes(showscale=False)
    return _style(fig, 430)


def margin_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Heatmap: Profit margin (%) for Region x Category."""
    if not _ok(df, "Region", "Category", "Sales", "Profit"):
        return None
    pivot = df.pivot_table(index="Region", columns="Category", values=["Sales", "Profit"], aggfunc="sum")
    if pivot.empty:
        return None
    margin = (pivot["Profit"] / pivot["Sales"].replace(0, np.nan) * 100).round(2)
    fig = px.imshow(margin, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, title="Profit Margin (%) by Region & Category")
    return _style(fig)


def correlation_heatmap(corr: pd.DataFrame) -> Optional[go.Figure]:
    """Heatmap of a correlation matrix."""
    if corr is None or corr.empty:
        return None
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title="Correlation Matrix")
    return _style(fig, 360)
