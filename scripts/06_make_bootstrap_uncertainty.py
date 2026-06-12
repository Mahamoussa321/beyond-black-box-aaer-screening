"""
06_make_bootstrap_uncertainty.py

Creates bootstrap uncertainty table for the SEC AAER manuscript.

Input:
    outputs/ai_teacher_guided_vc_publication/ai_vc_test_predictions.csv

Outputs:
    outputs/paper_tables_figures/table_bootstrap_uncertainty.csv
    outputs/paper_tables_figures/table_bootstrap_uncertainty.tex

Run from the main project folder:
    python scripts/06_make_bootstrap_uncertainty.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
AIVC = ROOT / "outputs" / "ai_teacher_guided_vc_publication"
OUT = ROOT / "outputs" / "paper_tables_figures"
OUT.mkdir(parents=True, exist_ok=True)

PRED = AIVC / "ai_vc_test_predictions.csv"
if not PRED.exists():
    raise SystemExit(
        f"Missing prediction file:\n{PRED}\n\n"
        "Please run the main publication pipeline first:\n"
        "python scripts/00_run_publication_pipeline.py"
    )

df = pd.read_csv(PRED)


def metric_values(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    pos = int(y.sum())

    if len(np.unique(y)) < 2:
        return {
            "ROC-AUC": np.nan,
            "PR-AUC": np.nan,
            "Top-5% Recall": np.nan,
            "Top-5% Lift": np.nan,
        }

    k = int(round(0.05 * n))
    k = max(1, min(k, n))
    order = np.argsort(-p)
    top = order[:k]
    tp = int(y[top].sum())
    precision = tp / k
    recall = tp / pos if pos > 0 else np.nan
    prevalence = pos / n if n > 0 else np.nan
    lift = precision / prevalence if prevalence and prevalence > 0 else np.nan

    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "Top-5% Recall": recall,
        "Top-5% Lift": lift,
    }


def bootstrap_ci(y, p, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)

    vals = {
        "ROC-AUC": [],
        "PR-AUC": [],
        "Top-5% Recall": [],
        "Top-5% Lift": [],
    }

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yb = y[idx]
        pb = p[idx]

        if len(np.unique(yb)) < 2:
            continue

        m = metric_values(yb, pb)
        for key in vals:
            vals[key].append(m[key])

    point = metric_values(y, p)
    out = {}
    for key in vals:
        arr = np.asarray(vals[key], dtype=float)
        out[key] = {
            "point": point[key],
            "lo": np.nanpercentile(arr, 2.5),
            "hi": np.nanpercentile(arr, 97.5),
        }
    return out


models = {}

# Teacher probability appears duplicated for each student-model row.
# Use one row per firm-year for the teacher.
dedup_cols = [c for c in ["cik", "year", "aaer_flag"] if c in df.columns]
teacher = df.drop_duplicates(subset=dedup_cols).copy()

if "teacher_probability" not in teacher.columns:
    raise SystemExit("Could not find column teacher_probability in ai_vc_test_predictions.csv")

models["LightGBM Teacher"] = (
    teacher["aaer_flag"].values,
    teacher["teacher_probability"].values,
)

for model_name in ["VC Student (validation-selected)", "AI-guided VC Student"]:
    sub = df[df["model"] == model_name].copy()
    if sub.empty:
        raise SystemExit(f"Could not find predictions for model: {model_name}")
    if "student_probability" not in sub.columns:
        raise SystemExit("Could not find column student_probability in ai_vc_test_predictions.csv")
    models[model_name] = (
        sub["aaer_flag"].values,
        sub["student_probability"].values,
    )

rows = []
for model_name, (y, p) in models.items():
    ci = bootstrap_ci(y, p, n_boot=1000, seed=42)
    for metric, values in ci.items():
        rows.append({
            "model": model_name,
            "metric": metric,
            "point": values["point"],
            "ci_low": values["lo"],
            "ci_high": values["hi"],
        })

res = pd.DataFrame(rows)
csv_out = OUT / "table_bootstrap_uncertainty.csv"
tex_out = OUT / "table_bootstrap_uncertainty.tex"
res.to_csv(csv_out, index=False)


def fmt(metric, point, lo, hi):
    if "Recall" in metric:
        return f"{100*point:.1f}\\% [{100*lo:.1f}\\%, {100*hi:.1f}\\%]"
    if "Lift" in metric:
        return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"
    if metric == "PR-AUC":
        return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


wide = {}
for metric in ["ROC-AUC", "PR-AUC", "Top-5% Recall", "Top-5% Lift"]:
    wide[metric] = {}
    for model in models:
        row = res[(res["model"] == model) & (res["metric"] == metric)].iloc[0]
        wide[metric][model] = fmt(metric, row["point"], row["ci_low"], row["ci_high"])

latex = []
latex.append(r"\begin{table}[!ht]")
latex.append(r"\centering")
latex.append(r"\caption{Percentile bootstrap uncertainty summary for key screening metrics}")
latex.append(r"\label{tab:bootstrap}")
latex.append(r"\begin{tabular}{lccc}")
latex.append(r"\toprule")
latex.append(r"\textbf{Metric} & \textbf{LightGBM Teacher} & \textbf{Unguided VC Student} & \textbf{AI-Guided VC Student} \\")
latex.append(r"\midrule")
for metric in ["ROC-AUC", "PR-AUC", "Top-5% Recall", "Top-5% Lift"]:
    latex.append(
        f"{metric} & "
        f"{wide[metric]['LightGBM Teacher']} & "
        f"{wide[metric]['VC Student (validation-selected)']} & "
        f"{wide[metric]['AI-guided VC Student']} \\\\"
    )
latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

tex_out.write_text("\n".join(latex), encoding="utf-8")

print("Saved bootstrap uncertainty outputs:")
print(csv_out)
print(tex_out)
print()
print(res.to_string(index=False))
