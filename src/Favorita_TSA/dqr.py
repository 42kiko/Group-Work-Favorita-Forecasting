from pathlib import Path

import pandas as pd

from Favorita_TSA.utils.data_loader import parquet_loader


def df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p><em>Keine Auffälligkeiten gefunden.</em></p>"
    return df.head(max_rows).to_html(index=False, escape=True, border=0)


def build_quality_tables(df: pd.DataFrame):
    # Missing per column
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_tbl = (
        miss_pct[miss_pct > 0]
        .rename("missing_%")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "column"})
        .round(2)
    )

    # Missing per row (Top 20)
    row_miss_pct = df.isna().mean(axis=1) * 100
    rows_missing_tbl = (
        row_miss_pct[row_miss_pct > 0]
        .sort_values(ascending=False)
        .head(20)
        .to_frame("missing_%")
        .reset_index()
        .rename(columns={"index": "row_index"})
        .round(2)
    )

    # Duplicate rows
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    dup_examples = df.loc[dup_mask].head(10)

    # Dtypes
    dtypes_tbl = (
        df.dtypes.astype(str)
        .rename("dtype")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "column"})
    )

    # Numeric outliers (IQR)
    out_rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        cnt = int(mask.sum())
        if cnt:
            examples = df.loc[mask, col].head(5).tolist()
            out_rows.append(
                {"column": col, "outlier_count": cnt, "example_values": str(examples)}
            )
    outliers_tbl = pd.DataFrame(out_rows)

    # Suspicious strings (leading/trailing spaces, empty after strip)
    str_rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        has_space_issue = series.str.match(r"^\s+|\s+$").any()
        empty_after_strip = (series.str.strip() == "").any()

        if has_space_issue or empty_after_strip:
            examples = series[series.str.match(r"^\s+|\s+$")].head(5).tolist()
            str_rows.append(
                {
                    "column": col,
                    "leading/trailing_space": bool(has_space_issue),
                    "empty_after_strip": bool(empty_after_strip),
                    "example_values": str(examples),
                }
            )
    suspicious_strings_tbl = pd.DataFrame(str_rows)

    # Categorical inconsistencies (small cardinality, would change with strip+lower)
    cat_rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        vals = df[col].dropna().astype(str)
        nunique = vals.nunique()
        if nunique == 0 or nunique > 50:
            continue
        norm = vals.str.strip().str.lower()
        if (vals != norm).any():
            cat_rows.append(
                {
                    "column": col,
                    "unique_original_sample": str(vals.unique()[:10].tolist()),
                    "unique_normalized_sample": str(norm.unique()[:10].tolist()),
                    "unique_count": int(nunique),
                }
            )
    categorical_tbl = pd.DataFrame(cat_rows)

    # Possible date columns (partially parseable)
    date_rows = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        ratio = float(parsed.notna().mean())
        # "teilweise parsebar" => Hinweis auf Mixed formats / dirty dates
        if 0.3 < ratio < 0.95:
            date_rows.append(
                {
                    "column": col,
                    "parseable_ratio": round(ratio, 3),
                    "example_values": str(df[col].dropna().head(5).tolist()),
                }
            )
    dates_tbl = (
        pd.DataFrame(date_rows).sort_values("parseable_ratio", ascending=False)
        if date_rows
        else pd.DataFrame()
    )

    return {
        "missing_tbl": missing_tbl,
        "rows_missing_tbl": rows_missing_tbl,
        "dup_count": dup_count,
        "dup_examples": dup_examples,
        "dtypes_tbl": dtypes_tbl,
        "outliers_tbl": outliers_tbl,
        "suspicious_strings_tbl": suspicious_strings_tbl,
        "categorical_tbl": categorical_tbl,
        "dates_tbl": dates_tbl,
    }


def write_quality_report_html(
    df: pd.DataFrame, out_path: str = "data_quality_report.html"
) -> None:
    tables = build_quality_tables(df)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Data Quality Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 24px; line-height: 1.35; }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .meta {{ color: #444; margin-bottom: 18px; }}
    .badges span {{ display:inline-block; padding: 4px 10px; border-radius: 999px; background:#f3f4f6; margin-right: 6px; }}
    .warn {{ background: #fff3cd !important; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
    th {{ background: #fafafa; }}
    code {{ background:#f6f8fa; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Data Quality Report</h1>

  <div class="meta badges">
    <span>Rows: {df.shape[0]}</span>
    <span>Columns: {df.shape[1]}</span>
    <span class="warn">Duplicate rows: {tables["dup_count"]}</span>
  </div>

  <div class="card">
    <h2>Missing values by column (Top 50)</h2>
    {df_to_html(tables["missing_tbl"], 50)}
  </div>

  <div class="card">
    <h2>Rows with missing values (Top 20)</h2>
    {df_to_html(tables["rows_missing_tbl"], 20)}
  </div>

  <div class="card">
    <h2>Duplicate rows (examples)</h2>
    {df_to_html(tables["dup_examples"], 10)}
  </div>

  <div class="card">
    <h2>Column dtypes</h2>
    {df_to_html(tables["dtypes_tbl"], 200)}
  </div>

  <div class="card">
    <h2>Numeric outliers (IQR)</h2>
    {df_to_html(tables["outliers_tbl"], 200)}
  </div>

  <div class="card">
    <h2>Suspicious strings (spaces / empty-after-strip)</h2>
    {df_to_html(tables["suspicious_strings_tbl"], 200)}
  </div>

  <div class="card">
    <h2>Categorical inconsistencies (strip+lower would change values)</h2>
    {df_to_html(tables["categorical_tbl"], 200)}
  </div>

  <div class="card">
    <h2>Possible date columns (partially parseable)</h2>
    {df_to_html(tables["dates_tbl"], 200)}
  </div>

  <p style="color:#666;margin-top:18px;">
    Hinweis: Dieser Report ändert keine Daten. Er zeigt nur potenzielle Cleaning-Aufgaben.
  </p>
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"✅ HTML Report geschrieben: {out_path}")


def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    suf = p.suffix.lower()
    if suf in [".csv", ".txt"]:
        return pd.read_csv(p, low_memory=False)
    if suf in [".parquet"]:
        return pd.read_parquet(p)
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    raise ValueError(f"Nicht unterstütztes Format: {suf}")


# write_quality_report_html(parquet_loader("train"), "train_data_quality_report.html")
def print_data_quality_report(df: pd.DataFrame):
    tables = build_quality_tables(df)

    print("Data Quality Report")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Duplicate rows: {tables['dup_count']}\n")

    print("Missing values by column (Top 50):")
    print(tables["missing_tbl"].to_string(index=False), "\n")

    print("Rows with missing values (Top 20):")
    print(tables["rows_missing_tbl"].to_string(index=False), "\n")

    print("Duplicate rows (examples):")
    print(tables["dup_examples"].to_string(index=False), "\n")

    print("Column dtypes:")
    print(tables["dtypes_tbl"].to_string(index=False), "\n")

    print("Numeric outliers (IQR):")
    print(tables["outliers_tbl"].to_string(index=False), "\n")

    print("Suspicious strings (spaces / empty-after-strip):")
    print(tables["suspicious_strings_tbl"].to_string(index=False), "\n")

    print("Categorical inconsistencies (strip+lower would change values):")
    print(tables["categorical_tbl"].to_string(index=False), "\n")

    print("Possible date columns (partially parseable):")
    print(tables["dates_tbl"].to_string(index=False), "\n")


df_oil = parquet_loader("oil")

print_data_quality_report(df_oil)
