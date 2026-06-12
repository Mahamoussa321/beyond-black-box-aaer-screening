"""
05_make_highres_publication_figures.py

Creates high-resolution, colorful, journal-ready figures for the SEC AAER
AI-teacher / AI-guided VC paper. These are intended for the final manuscript.

Outputs are saved as PNG and PDF in:
    outputs/publication_figures_highres

A copy of the key PNG files is also saved in:
    outputs/paper_tables_figures
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "baseline_models"
AIVC = ROOT / "outputs" / "ai_teacher_guided_vc_publication"
PAPER = ROOT / "outputs" / "paper_tables_figures"
FEATURES = ROOT / "outputs" / "final_features" / "sec_aaer_features_2009_2025.csv"

OUT = ROOT / "outputs" / "publication_figures_highres"
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "axes.linewidth": 0.9,
    "savefig.dpi": 450,
})

# High-contrast, publication-friendly palette.
COLORS = {
    "LightGBM Teacher": "#E85D04",                    # orange
    "AI-guided VC Student": "#0077B6",                # strong blue
    "VC Student (validation-selected)": "#8338EC",   # purple
    "LightGBM": "#F48C06",                            # amber
    "Logistic Regression": "#495057",                 # charcoal
    "Random Forest": "#2A9D8F",                       # teal
    "Extra Trees": "#6A4C93",                         # violet
    "Gradient Boosting": "#D00000",                   # red
}

POS_PALETTE = ["#0077B6", "#00B4D8", "#06D6A0", "#118AB2", "#3A86FF", "#2EC4B6", "#43AA8B", "#90E0EF"]
NEG_PALETTE = ["#EF476F", "#F78C6B", "#FF7096", "#C77DFF", "#F28482", "#FF6B6B", "#FF9770", "#D65DB1"]
CURVE_PALETTE = ["#0077B6", "#EF476F", "#06D6A0", "#F78C6B", "#3A86FF", "#C77DFF", "#FFBE0B", "#2A9D8F"]


def clean_label(x):
    return str(x).replace("VC Student (validation-selected)", "VC Student")


def style_axis(ax):
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="right")


def save_all(fig, name):
    # Save in high-resolution figure folder.
    fig.savefig(OUT / f"{name}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        # Also keep an easy PNG copy in the paper tables/figures folder.
    fig.savefig(PAPER / f"{name}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure 1: Study workflow
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

boxes = [
    (0.55, 5.2, 2.55, 1.0, "SEC financial\nstatement data\n2009-2026", "#E0F7FA"),
    (0.55, 3.7, 2.55, 1.0, "SEC AAER\nenforcement\nreleases", "#FFE5EC"),
    (4.0, 4.45, 2.85, 1.1, "Firm-year panel\n+ AAER risk proxy", "#FFF3BF"),
    (7.6, 5.3, 3.1, 1.0, "AI teacher\nLightGBM / MLP", "#FFE8CC"),
    (7.6, 3.65, 3.1, 1.0, "AI-guided\nsingle-index VC\nstudent", "#D0EBFF"),
    (4.0, 1.6, 2.85, 1.0, "Temporal split\nTrain 2009-2018\nValidation 2019-2021\nTest 2022-2025", "#E6FCF5"),
    (7.6, 1.6, 3.1, 1.0, "Evaluation\nROC-AUC, PR-AUC\nTop-k recall, lift", "#F3D9FA"),
]
for x, y, w, h, txt, face in boxes:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.10",
                           linewidth=1.25, edgecolor="#343A40", facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=11, fontweight="semibold", color="#212529")

for start, end in [
    ((3.1, 5.7), (4.0, 5.05)), ((3.1, 4.2), (4.0, 4.85)),
    ((6.85, 5.0), (7.6, 5.8)), ((6.85, 4.7), (7.6, 4.15)),
    ((9.15, 5.3), (9.15, 4.65)), ((5.45, 4.45), (5.45, 2.6)),
    ((6.85, 2.1), (7.6, 2.1)), ((9.15, 3.65), (9.15, 2.6)),
]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=17, linewidth=1.5, color="#495057"))
save_all(fig, "fig01_sec_aaer_workflow_pretty")


# =============================================================================
# Figure 2: Class imbalance by year
# =============================================================================
if FEATURES.exists():
    df = pd.read_csv(FEATURES)
    df["year"] = df["year"].astype(int)
    yearly = df.groupby("year").agg(
        firm_years=("aaer_flag", "size"),
        aaer_positives=("aaer_flag", "sum")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(yearly["year"], yearly["firm_years"], color="#90E0EF", alpha=0.95, label="Firm-years", edgecolor="white")
    ax1.set_ylabel("Total firm-years")
    ax1.set_xlabel("Fiscal year")
    style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["aaer_positives"], color="#EF476F", marker="o", linewidth=2.5,
             label="AAER-positive firm-years")
    ax2.set_ylabel("AAER-positive firm-years")
    ax2.spines["top"].set_visible(False)

    total_pos = int(df["aaer_flag"].sum())
    total_n = len(df)
    prevalence = total_pos / total_n * 100
    ax1.text(0.02, 0.95, f"Overall prevalence: {total_pos:,}/{total_n:,} = {prevalence:.3f}%",
             transform=ax1.transAxes, ha="left", va="top", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CED4DA"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.02, 0.84), frameon=True)
    fig.tight_layout()
    save_all(fig, "fig02_class_imbalance_by_year_pretty")


# =============================================================================
# Collect prediction curves
# =============================================================================
curve_items = []
base_pred_path = BASE / "baseline_test_predictions.csv"
if base_pred_path.exists():
    bp = pd.read_csv(base_pred_path)
    for model in ["Logistic Regression", "LightGBM", "Random Forest", "Extra Trees"]:
        sub = bp[bp["model"] == model].copy()
        if not sub.empty:
            curve_items.append((model, sub["aaer_flag"].values, sub["predicted_probability"].values))

ai_pred_path = AIVC / "ai_vc_test_predictions.csv"
if ai_pred_path.exists():
    ap = pd.read_csv(ai_pred_path)
    first_model = ap["model"].iloc[0] if len(ap) else None
    if first_model is not None:
        sub = ap[ap["model"] == first_model].copy()
        if "teacher_probability" in sub.columns:
            curve_items.append(("LightGBM Teacher", sub["aaer_flag"].values, sub["teacher_probability"].values))
    for model in ["AI-guided VC Student", "VC Student (validation-selected)"]:
        sub = ap[ap["model"] == model].copy()
        if not sub.empty and "student_probability" in sub.columns:
            curve_items.append((model, sub["aaer_flag"].values, sub["student_probability"].values))

priority = {"LightGBM Teacher": 0, "AI-guided VC Student": 1, "VC Student (validation-selected)": 2,
            "LightGBM": 3, "Logistic Regression": 4, "Random Forest": 5, "Extra Trees": 6}
curve_items = sorted(curve_items, key=lambda x: priority.get(x[0], 99))


# =============================================================================
# Figure 3: ROC curves
# =============================================================================
if curve_items:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for model, y, p in curve_items:
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = roc_auc_score(y, p)
        ax.plot(fpr, tpr, linewidth=2.4, color=COLORS.get(model, None), label=f"{clean_label(model)} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#ADB5BD", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_axis(ax)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    save_all(fig, "fig03_roc_curves_pretty")


# =============================================================================
# Figure 4: Precision-recall curves
# =============================================================================
if curve_items:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    y0 = curve_items[0][1]
    prevalence = np.mean(y0)
    ax.axhline(prevalence, linestyle="--", linewidth=1.2, color="#ADB5BD", label=f"Random baseline ({prevalence:.4f})")
    for model, y, p in curve_items:
        prec, rec, _ = precision_recall_curve(y, p)
        apv = average_precision_score(y, p)
        ax.plot(rec, prec, linewidth=2.4, color=COLORS.get(model, None), label=f"{clean_label(model)} (AP={apv:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.055, ax.get_ylim()[1]))
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_all(fig, "fig04_precision_recall_curves_pretty")


# =============================================================================
# Figure 5: model comparison bar chart
# =============================================================================
model_table_path = PAPER / "table_sec_model_comparison.csv"
if model_table_path.exists():
    mt = pd.read_csv(model_table_path).dropna(subset=["roc_auc", "pr_auc_average_precision"]).copy()
    mt["plot_label"] = mt["model"].map(clean_label)
    mt = mt.sort_values("roc_auc", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    bar_colors = [COLORS.get(m, "#0077B6") for m in mt["model"]]
    axes[0].barh(mt["plot_label"], mt["roc_auc"], color=bar_colors, alpha=0.92, edgecolor="white")
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_xlim(0.45, max(0.78, mt["roc_auc"].max() + 0.03))
    style_axis(axes[0]); add_panel_label(axes[0], "A")

    axes[1].barh(mt["plot_label"], mt["pr_auc_average_precision"], color=bar_colors, alpha=0.92, edgecolor="white")
    axes[1].set_xlabel("PR-AUC / average precision")
    axes[1].set_xlim(0, max(0.035, mt["pr_auc_average_precision"].max() * 1.15))
    style_axis(axes[1]); add_panel_label(axes[1], "B")
    fig.tight_layout()
    save_all(fig, "fig05_model_comparison_auc_pr_pretty")


# =============================================================================
# Figure 6: top-screening recall and lift
# =============================================================================
top_path = PAPER / "table_sec_top_screening.csv"
if top_path.exists():
    top = pd.read_csv(top_path)
    top = top[top["screening_rule"].isin(["Top 1%", "Top 5%", "Top 10%"])].copy()
    keep_models = ["LightGBM Teacher", "AI-guided VC Student", "VC Student (validation-selected)", "LightGBM", "Logistic Regression"]
    top = top[top["model"].isin(keep_models)].copy()
    top["model_label"] = top["model"].map(clean_label)
    order = ["Top 1%", "Top 5%", "Top 10%"]
    top["screening_rule"] = pd.Categorical(top["screening_rule"], categories=order, ordered=True)
    top = top.sort_values(["model_label", "screening_rule"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for model in top["model"].unique():
        sub = top[top["model"] == model]
        axes[0].plot(sub["screening_rule"], sub["recall"], marker="o", linewidth=2.6,
                     markersize=7, color=COLORS.get(model, None), label=clean_label(model))
        axes[1].plot(sub["screening_rule"], sub["lift_over_random"], marker="o", linewidth=2.6,
                     markersize=7, color=COLORS.get(model, None), label=clean_label(model))
    axes[0].set_ylabel("Recall of AAER-positive firm-years")
    axes[0].set_xlabel("Screening budget")
    axes[0].set_ylim(0, max(0.40, top["recall"].max() * 1.2))
    style_axis(axes[0]); add_panel_label(axes[0], "A")
    axes[1].set_ylabel("Lift over random screening")
    axes[1].set_xlabel("Screening budget")
    axes[1].set_ylim(0, max(18, top["lift_over_random"].max() * 1.2))
    style_axis(axes[1]); add_panel_label(axes[1], "B")
    axes[1].legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_all(fig, "fig06_top_screening_recall_lift_pretty")


# =============================================================================
# Figure 7: AI-VC index weights with pretty sign-aware colors
# =============================================================================
weights_path = AIVC / "ai_vc_index_weights.csv"
if weights_path.exists():
    w = pd.read_csv(weights_path).sort_values("abs_alpha_weight", ascending=False).head(18).copy()
    w["feature"] = w["index_feature"].astype(str).str.replace("_", " ")
    w = w.iloc[::-1].copy()
    vals = w["alpha_weight"].values
    labels = w["feature"].values

    pos_i, neg_i = 0, 0
    bar_colors = []
    for v in vals:
        if v >= 0:
            bar_colors.append(POS_PALETTE[pos_i % len(POS_PALETTE)]); pos_i += 1
        else:
            bar_colors.append(NEG_PALETTE[neg_i % len(NEG_PALETTE)]); neg_i += 1

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    bars = ax.barh(labels, vals, color=bar_colors, alpha=0.96, edgecolor="white", linewidth=1.0)
    ax.axvline(0, color="#343A40", linewidth=1.2)
    ax.set_xlabel("Single-index weight")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin, xmax = vals.min(), vals.max()
    pad = 0.02 * (xmax - xmin)
    for bar, v in zip(bars, vals):
        y = bar.get_y() + bar.get_height() / 2
        if v >= 0:
            ax.text(v + pad, y, f"{v:.2f}", va="center", ha="left", fontsize=10, color="#212529")
        else:
            ax.text(v - pad, y, f"{v:.2f}", va="center", ha="right", fontsize=10, color="#212529")
    ax.set_xlim(xmin - 0.08, xmax + 0.08)
    fig.tight_layout()
    save_all(fig, "fig07_ai_vc_index_weights_pretty")


# =============================================================================
# Figure 8: AI-VC coefficient curves with pretty colors
# =============================================================================
coef_path = AIVC / "ai_vc_coefficient_curves.csv"
if coef_path.exists():
    coef = pd.read_csv(coef_path)
    candidate_cols = [c for c in coef.columns if c not in ["index_grid", "intercept"]]
    variation = coef[candidate_cols].std().sort_values(ascending=False).head(7).index.tolist()
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for i, c in enumerate(variation):
        ax.plot(coef["index_grid"], coef[c], linewidth=2.6, color=CURVE_PALETTE[i % len(CURVE_PALETTE)],
                label=c.replace("_", " "))
    ax.axhline(0, color="#6C757D", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Learned firm-risk context index")
    ax.set_ylabel("Index-varying coefficient")
    style_axis(ax)
    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.95)
    fig.tight_layout()
    save_all(fig, "fig08_ai_vc_coefficient_curves_pretty")


# =============================================================================
# Figure 9: predicted risk distribution
# =============================================================================
if ai_pred_path.exists():
    ap = pd.read_csv(ai_pred_path)
    sub = ap[ap["model"] == "AI-guided VC Student"].copy()
    if not sub.empty and "student_probability" in sub.columns:
        fig, ax = plt.subplots(figsize=(8.5, 5.4))
        neg = sub[sub["aaer_flag"] == 0]["student_probability"].values
        pos = sub[sub["aaer_flag"] == 1]["student_probability"].values
        ax.hist(neg, bins=45, density=True, alpha=0.62, label="Non-AAER firm-years", color="#90E0EF", edgecolor="white")
        ax.hist(pos, bins=18, density=True, alpha=0.78, label="AAER-positive firm-years", color="#EF476F", edgecolor="white")
        ax.set_xlabel("Predicted AAER risk")
        ax.set_ylabel("Density")
        style_axis(ax)
        ax.legend(frameon=True)
        fig.tight_layout()
        save_all(fig, "fig09_ai_vc_predicted_risk_distribution_pretty")

print("\nHigh-resolution colorful publication figures saved to:")
print(OUT)
print("\nPNG copies also saved to:")
print(PAPER)
print("\nFiles created:")
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)
