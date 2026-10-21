# 📊 Sales Analytics Dashboard

A Streamlit app that uploads, validates, cleans, filters and visualises sales data.

**Live app:** `https://<your-app-name>.streamlit.app`  *(add your URL after deploying)*

## Features
| Requirement | Where |
|---|---|
| Upload **CSV / XLSX / JSON** (multi-sheet Excel supported) | sidebar → *1 · Data* |
| Validation: dimensions, column names, dtypes, missing values, duplicates, non-numeric text in numeric columns | tab **Data Overview** |
| Cleaning: missing values, duplicates, inconsistent labels, mixed date formats, invalid numbers, outliers (full audit log + download) | tab **Cleaning Report** |
| Reusable statistics functions (`numeric_summary`, `categorical_summary`, `correlation_matrix`, `group_summary`) | `data_processing.py`, tab **Statistics** |
| 6 interactive filters: date range, Region, Category, Segment, Ship Mode, State | sidebar → *3 · Filters* |
| 5 live KPIs: Total Sales, Total Profit, Total Orders (unique Order IDs), Total Quantity, Profit Margin % | tab **Dashboard** |
| 7 interactive Plotly charts (trend, category, region donut, sub-category profit, discount scatter, top products, margin heatmap) | tab **Dashboard** |

## Project structure
```
app.py                 Streamlit UI (layout, widgets, caching, error handling)
data_processing.py     Loading, validation, cleaning, statistics, KPIs  (no Streamlit code)
charts.py              Plotly figure builders
tests/                 pytest unit tests + end-to-end Streamlit tests
data/                  sample dataset (+ .xlsx / .json copies for testing uploads)
requirements.txt
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Test
```bash
pip install pytest
pytest -q
```

## Cleaning rules (all configurable/visible in the Cleaning Report)
* Column names are normalised (`order_id`, `ORDER ID` → `Order ID`).
* Empty rows and exact duplicate rows are removed. Repeated **Order IDs are kept** (an order has several product lines); *Total Orders* counts unique IDs.
* Text variants are unified (`SameDay`/`Same Day`, `consumer`/`Consumer`, `west`/`West`, `" technology "`); placeholders (`N/A`, `Unknown Segment`, blank) → `Unknown`.
* `Region` is corrected from `State`, `Category` from `Sub-Category` (dominant value wins).
* Dates: ISO, `dd/mm/yyyy`, `dd-Mon-yyyy`, `dd-mm-yyyy` parsed; ambiguous ship dates re-read month-first if day-first gives an implausible delay; ship before order → missing.
* Numbers: `'five'`→5, `'ten percent'`→0.10, `'?'`, `'unknown'`, `inf` → missing.
* Invalid: Sales ≤ 0, Quantity ≤ 0 or fractional, Discount outside 0–1, Profit > Sales. Extreme outliers (Quantity > 3×IQR, Sales > 10×IQR) are treated as missing (toggle in sidebar).
* Missing Sales/Profit: drop rows (default) or impute Sub-Category median. Missing Quantity/Discount: Sub-Category median.
* Rows without Order ID or a valid Order Date are dropped.

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public** GitHub repo (see below).
2. Go to <https://share.streamlit.io> → sign in with GitHub → **Create app** → *Deploy a public app from GitHub*.
3. Repository: `<you>/<repo>`, Branch: `main`, Main file path: `app.py` → **Deploy**.
4. Wait for the build (installs `requirements.txt`), then open the URL.

```bash
git init
git add .
git commit -m "Sales analytics dashboard"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## Post-deployment verification checklist
- [ ] App loads with the sample dataset (5 KPIs visible).
- [ ] Upload `data/sample_upload.xlsx`, then `data/sample_upload.json`, then the CSV — each loads; KPIs identical.
- [ ] Upload a `.txt` file / a CSV without `Sales` → friendly error, no crash.
- [ ] Change Region / Category / Date filters → KPIs and charts change; *Reset filters* restores them.
- [ ] Deselect every Region → "No records match" warning, no error.
- [ ] Hover, zoom and toggle legend items in the charts.
- [ ] Download the cleaned CSV from the *Cleaning Report* tab.
