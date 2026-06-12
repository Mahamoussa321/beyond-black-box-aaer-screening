"""
01_make_final_features.py

Creates the final feature file for SEC AAER firm-year modeling.

Input:
    data/sec_aaer_firmyear_modeling_dataset_2009_2026.csv

Output:
    outputs/final_features/sec_aaer_features_2009_2025.csv
    outputs/final_features/sec_aaer_feature_dictionary.csv

Important:
    - Uses only fiscal years 2009-2025 for modeling because AAER release-year
      labeling gives reliable label years through 2025.
    - Computes lag/change variables using prior-year firm information only.
    - Does NOT use the future test period for imputation/scaling; those are done
      inside the modeling scripts.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sec_aaer_firmyear_modeling_dataset_2009_2026.csv"
OUTDIR = ROOT / "outputs" / "final_features"
OUTDIR.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise SystemExit(
        f"Missing input file: {DATA}\n"
        "Please copy sec_aaer_firmyear_modeling_dataset_2009_2026.csv into the data/ folder."
    )

def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b.notna()) & (b != 0), a / b, np.nan)

def add_if_missing(df, col, numerator, denominator):
    if col not in df.columns:
        df[col] = safe_div(df[numerator], df[denominator])

df = pd.read_csv(DATA)

df = df.dropna(subset=["aaer_flag", "year", "cik"]).copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df["aaer_flag"] = pd.to_numeric(df["aaer_flag"], errors="coerce").astype(int)

# Use label years 2009-2025 only. Year 2026 is incomplete for AAER labeling.
df = df[(df["year"] >= 2009) & (df["year"] <= 2025)].copy()

# Standardize identifiers.
df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
df["sic"] = pd.to_numeric(df.get("sic"), errors="coerce")
df["sic2"] = (df["sic"] // 100).astype("Int64").astype(str).replace("<NA>", np.nan)

# Numeric coercion.
base_numeric = [
    "assets", "liabilities", "revenue", "net_income", "cash", "receivables",
    "inventory", "current_assets", "current_liabilities", "debt", "equity",
    "operating_income", "gross_profit", "log_assets", "liabilities_assets",
    "roa", "net_margin", "cash_assets", "receivables_assets",
    "inventory_assets", "current_ratio", "debt_assets", "revenue_growth",
    "assets_growth"
]
for c in base_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Recompute key ratios in case any are missing.
if "log_assets" not in df.columns:
    df["log_assets"] = np.where(df["assets"] > 0, np.log(df["assets"]), np.nan)
add_if_missing(df, "liabilities_assets", "liabilities", "assets")
add_if_missing(df, "roa", "net_income", "assets")
add_if_missing(df, "net_margin", "net_income", "revenue")
add_if_missing(df, "cash_assets", "cash", "assets")
add_if_missing(df, "receivables_assets", "receivables", "assets")
add_if_missing(df, "inventory_assets", "inventory", "assets")
add_if_missing(df, "current_ratio", "current_assets", "current_liabilities")
add_if_missing(df, "debt_assets", "debt", "assets")

# Additional accounting red-flag and Beneish-style features.
df["gross_margin"] = safe_div(df["gross_profit"], df["revenue"])
df["operating_margin"] = safe_div(df["operating_income"], df["revenue"])
df["working_capital_assets"] = safe_div(df["current_assets"] - df["current_liabilities"], df["assets"])
df["asset_quality_proxy"] = 1 - safe_div(df["current_assets"], df["assets"])
df["receivables_revenue"] = safe_div(df["receivables"], df["revenue"])
df["inventory_revenue"] = safe_div(df["inventory"], df["revenue"])
df["debt_equity"] = safe_div(df["debt"], df["equity"])
df["loss_flag"] = (df["net_income"] < 0).astype(float)
df["negative_equity_flag"] = (df["equity"] < 0).astype(float)

# Prior-year firm lags only; these are safe for temporal prediction.
df = df.sort_values(["cik", "year"]).copy()
grp = df.groupby("cik", sort=False)

lag_source_cols = [
    "assets", "liabilities", "revenue", "net_income", "cash", "receivables",
    "inventory", "current_assets", "current_liabilities", "debt", "equity",
    "gross_profit", "gross_margin", "asset_quality_proxy",
    "receivables_revenue", "liabilities_assets", "debt_assets", "roa"
]

for c in lag_source_cols:
    if c in df.columns:
        df[f"lag_{c}"] = grp[c].shift(1)

df["dsri_proxy"] = safe_div(df["receivables_revenue"], df["lag_receivables_revenue"])
df["gmi_proxy"] = safe_div(df["lag_gross_margin"], df["gross_margin"])
df["aqi_proxy"] = safe_div(df["asset_quality_proxy"], df["lag_asset_quality_proxy"])
df["sgi"] = safe_div(df["revenue"], df["lag_revenue"])
df["lvgi"] = safe_div(df["liabilities_assets"], df["lag_liabilities_assets"])
df["debt_assets_change"] = df["debt_assets"] - df["lag_debt_assets"]
df["roa_change"] = df["roa"] - df["lag_roa"]
df["gross_margin_change"] = df["gross_margin"] - df["lag_gross_margin"]
df["receivables_growth"] = safe_div(df["receivables"] - df["lag_receivables"], np.abs(df["lag_receivables"]))
df["inventory_growth"] = safe_div(df["inventory"] - df["lag_inventory"], np.abs(df["lag_inventory"]))
df["debt_growth"] = safe_div(df["debt"] - df["lag_debt"], np.abs(df["lag_debt"]))

df["delta_current_assets"] = df["current_assets"] - df["lag_current_assets"]
df["delta_cash"] = df["cash"] - df["lag_cash"]
df["delta_current_liabilities"] = df["current_liabilities"] - df["lag_current_liabilities"]
df["total_accruals_assets_proxy"] = safe_div(
    (df["delta_current_assets"] - df["delta_cash"]) - df["delta_current_liabilities"],
    df["assets"]
)

# Replace inf with nan; imputation/scaling is fit later using training data only.
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

# Do not include raw lag columns by default in the final paper feature list, but keep
# them in the file for optional sensitivity checks.
out = OUTDIR / "sec_aaer_features_2009_2025.csv"
df.to_csv(out, index=False)

feature_dictionary = [
    ("log_assets", "Firm size: log total assets."),
    ("liabilities_assets", "Leverage: liabilities divided by assets."),
    ("roa", "Profitability: net income divided by assets."),
    ("net_margin", "Net income divided by revenue."),
    ("cash_assets", "Cash intensity: cash divided by assets."),
    ("receivables_assets", "Receivables divided by assets."),
    ("inventory_assets", "Inventory divided by assets."),
    ("current_ratio", "Current assets divided by current liabilities."),
    ("debt_assets", "Debt divided by assets."),
    ("revenue_growth", "Year-over-year revenue growth from SEC panel."),
    ("assets_growth", "Year-over-year asset growth from SEC panel."),
    ("gross_margin", "Gross profit divided by revenue."),
    ("operating_margin", "Operating income divided by revenue."),
    ("working_capital_assets", "Working capital divided by assets."),
    ("asset_quality_proxy", "Proxy for non-current/less liquid asset share."),
    ("receivables_revenue", "Receivables divided by revenue."),
    ("inventory_revenue", "Inventory divided by revenue."),
    ("debt_equity", "Debt divided by equity."),
    ("loss_flag", "Indicator for negative net income."),
    ("negative_equity_flag", "Indicator for negative equity."),
    ("dsri_proxy", "Beneish-style days-sales-in-receivables proxy."),
    ("gmi_proxy", "Beneish-style gross-margin index proxy."),
    ("aqi_proxy", "Beneish-style asset-quality index proxy."),
    ("sgi", "Beneish-style sales-growth index."),
    ("lvgi", "Beneish-style leverage index."),
    ("debt_assets_change", "Change in debt-to-assets."),
    ("roa_change", "Change in return on assets."),
    ("gross_margin_change", "Change in gross margin."),
    ("receivables_growth", "Receivables growth."),
    ("inventory_growth", "Inventory growth."),
    ("debt_growth", "Debt growth."),
    ("total_accruals_assets_proxy", "Working-capital accruals proxy divided by assets."),
]
pd.DataFrame(feature_dictionary, columns=["feature", "description"]).to_csv(
    OUTDIR / "sec_aaer_feature_dictionary.csv", index=False
)

print("Saved:", out)
print("Rows:", len(df))
print("AAER class counts:")
print(df["aaer_flag"].value_counts())
print("Years:", df["year"].min(), "-", df["year"].max())
