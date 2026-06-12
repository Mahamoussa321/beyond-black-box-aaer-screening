"""
04_make_publication_tables_and_figures.py

Combines baseline, AI teacher, and AI-guided VC results into paper-ready tables
and standard figures. The companion script 05_make_highres_publication_figures.py
creates the prettier high-resolution PNG/PDF figure set for the manuscript.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "baseline_models"
AIVC = ROOT / "outputs" / "ai_teacher_guided_vc_publication"
OUT = ROOT / "outputs" / "paper_tables_figures"
OUT.mkdir(parents=True, exist_ok=True)

baseline_metrics = BASE / "baseline_auc_metrics.csv"
baseline_top = BASE / "baseline_top_screening_metrics.csv"
ai_metrics = AIVC / "ai_vc_test_metrics.csv"
ai_top = AIVC / "ai_vc_top_screening_metrics.csv"
teacher_metrics = AIVC / "teacher_test_metrics.csv"
ai_predictions = AIVC / "ai_vc_test_predictions.csv"

missing = [p for p in [baseline_metrics, baseline_top, ai_metrics, ai_top, teacher_metrics] if not p.exists()]
if missing:
    raise SystemExit("Missing files. Run earlier scripts first:\n" + "\n".join(map(str, missing)))


def top_screening_metrics(y_true, prob, model_name):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    n = len(y_true)
    total_pos = int(y_true.sum())
    prevalence = total_pos / n if n else np.nan
    order = np.argsort(-prob)
    rules = [
        ("Top-K=positives", total_pos),
        ("Top 1%", max(1, int(round(0.01 * n)))),
        ("Top 5%", max(1, int(round(0.05 * n)))),
        ("Top 10%", max(1, int(round(0.10 * n)))),
        ("Top 100", min(100, n)),
        ("Top 250", min(250, n)),
        ("Top 500", min(500, n)),
    ]
    rows = []
    for label, k in rules:
        k = max(1, int(k))
        idx = order[:k]
        tp = int(y_true[idx].sum())
        precision = tp / k
        recall = tp / total_pos if total_pos else np.nan
        lift = precision / prevalence if prevalence else np.nan
        rows.append({
            "model": model_name,
            "screening_rule": label,
            "selected_n": k,
            "true_positives_found": tp,
            "precision": precision,
            "recall": recall,
            "lift_over_random": lift,
        })
    return rows


bm = pd.read_csv(baseline_metrics)
am = pd.read_csv(ai_metrics)
tm = pd.read_csv(teacher_metrics)

# Standardize columns.
bm["model_family"] = "Baseline"
am["model_family"] = "AI-guided VC"
tm["model_family"] = "AI teacher"

for df in [bm, am, tm]:
    if "n_test" not in df.columns and "n" in df.columns:
        df["n_test"] = df["n"]
    if "positives_test" not in df.columns and "positives" in df.columns:
        df["positives_test"] = df["positives"]

model_table = pd.concat([tm, am, bm], ignore_index=True, sort=False)
model_table = model_table.sort_values("pr_auc_average_precision", ascending=False)
keep_cols = [
    "model", "model_family", "roc_auc", "pr_auc_average_precision", "brier_score",
    "n_train", "n_test", "positives_train", "positives_test", "lambda_ai",
    "pos_weight", "degree", "teacher_selected_on_validation", "selection_metric"
]
keep_cols = [c for c in keep_cols if c in model_table.columns]
model_table[keep_cols].to_csv(OUT / "table_sec_model_comparison.csv", index=False)

bt = pd.read_csv(baseline_top)
at = pd.read_csv(ai_top)

# Add AI-teacher top-screening performance from the saved teacher probabilities.
teacher_top = pd.DataFrame()
if ai_predictions.exists():
    pred = pd.read_csv(ai_predictions)
    if "teacher_probability" in pred.columns:
        one = pred.drop_duplicates(subset=[c for c in ["cik", "year", "aaer_flag"] if c in pred.columns]).copy()
        teacher_top = pd.DataFrame(top_screening_metrics(one["aaer_flag"].values, one["teacher_probability"].values, "LightGBM Teacher"))

top_table = pd.concat([teacher_top, bt, at], ignore_index=True, sort=False)
top_table.to_csv(OUT / "table_sec_top_screening_full.csv", index=False)
compact = top_table[top_table["screening_rule"].isin(["Top 1%", "Top 5%", "Top 10%", "Top 100", "Top 250", "Top 500"])].copy()
compact.to_csv(OUT / "table_sec_top_screening.csv", index=False)

# Standard model-comparison figure. High-resolution prettier figures are made by script 05.
plot_df = model_table[keep_cols].copy().dropna(subset=["roc_auc", "pr_auc_average_precision"])
labels = plot_df["model"].astype(str).tolist()
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.8))
ax.bar(x - width/2, plot_df["roc_auc"], width, label="ROC-AUC")
ax.bar(x + width/2, plot_df["pr_auc_average_precision"], width, label="PR-AUC")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_ylabel("Metric value")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figure_sec_model_auc_pr.png", dpi=300)
plt.close(fig)

# Standard top-screening recall figure.
screen_df = compact[compact["screening_rule"].isin(["Top 1%", "Top 5%", "Top 10%"])].copy()
fig, ax = plt.subplots(figsize=(9, 4.8))
for model in screen_df["model"].unique():
    sub = screen_df[screen_df["model"] == model]
    ax.plot(sub["screening_rule"], sub["recall"], marker="o", label=model)
ax.set_ylabel("Recall of AAER-positive firm-years")
ax.set_xlabel("Screening budget")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figure_sec_top_screening_recall.png", dpi=300)
plt.close(fig)

# Standard AI-VC index weights. Pretty color version is created by script 05.
wpath = AIVC / "ai_vc_index_weights.csv"
if wpath.exists():
    weights = pd.read_csv(wpath).sort_values("abs_alpha_weight", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(weights["index_feature"][::-1], weights["alpha_weight"][::-1])
    ax.set_xlabel("Index weight")
    fig.tight_layout()
    fig.savefig(OUT / "figure_ai_vc_index_weights.png", dpi=300)
    plt.close(fig)

# Standard AI-VC coefficient curves. Pretty color version is created by script 05.
cpath = AIVC / "ai_vc_coefficient_curves.csv"
if cpath.exists():
    coef = pd.read_csv(cpath)
    candidate_cols = [c for c in coef.columns if c not in ["index_grid", "intercept"]]
    variation = coef[candidate_cols].std().sort_values(ascending=False).head(8).index.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in variation:
        ax.plot(coef["index_grid"], coef[c], label=c)
    ax.set_xlabel("Learned firm-risk context index")
    ax.set_ylabel("Varying coefficient")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure_ai_vc_coefficient_curves.png", dpi=300)
    plt.close(fig)

summary = []
summary.append("SEC AAER publication results summary")
summary.append("====================================")
summary.append("")
summary.append("Model comparison:")
summary.append(model_table[keep_cols].to_string(index=False))
summary.append("")
summary.append("Top-screening table:")
summary.append(compact.to_string(index=False))
(OUT / "RUN_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")

print("\nModel comparison:")
print(model_table[keep_cols].to_string(index=False))
print("\nTop screening summary:")
print(compact.to_string(index=False))
print("\nSaved paper tables and figures to:", OUT)
