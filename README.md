# SEC AAER AI-Teacher / AI-Guided VC Publication Package

This package contains the final reproducible code for the SEC AAER external-validation study. It trains baseline audit-risk models, selects an AI teacher on validation years, trains an AI-guided single-index varying-coefficient (VC) student, evaluates once on temporal test years, and creates colorful high-resolution publication figures.

## Data

Place the SEC AAER firm-year modeling dataset here:

```text
data/sec_aaer_firmyear_modeling_dataset_2009_2026.csv
```

The package includes the dataset used in the verified run. If you want to rerun with a newly rebuilt dataset, replace the CSV in `data/` before running the pipeline.

## Temporal design

- Train years: 2009-2018
- Validation years: 2019-2021
- Test years: 2022-2025

Model and hyperparameter selection are made on the validation period. The test period is used only for final evaluation.

## Models

- Logistic Regression baseline
- Random Forest baseline
- Extra Trees baseline
- LightGBM baseline
- AI teacher candidates: LightGBM and neural MLP
- AI-guided single-index VC student
- Unguided VC student for ablation

## Run everything

In PowerShell:

```powershell
cd "C:\Users\maham\Desktop\sec_aaer_publication_ai_teacher_vc_PRETTY_FINAL\sec_aaer_publication_ai_teacher_vc_PRETTY_FINAL"

$py = "C:\Users\maham\anaconda3\python.exe"

& $py -m pip install -r requirements.txt

& $py ".\scripts\00_run_publication_pipeline.py" *>&1 | Tee-Object ".\publication_pipeline_log.txt"
```

## Main outputs

Tables:

```text
outputs/paper_tables_figures/table_sec_model_comparison.csv
outputs/paper_tables_figures/table_sec_top_screening.csv
outputs/paper_tables_figures/table_sec_top_screening_full.csv
```

Standard figures:

```text
outputs/paper_tables_figures/figure_sec_model_auc_pr.png
outputs/paper_tables_figures/figure_sec_top_screening_recall.png
outputs/paper_tables_figures/figure_ai_vc_index_weights.png
outputs/paper_tables_figures/figure_ai_vc_coefficient_curves.png
```

Colorful high-resolution manuscript figures:

```text
outputs/publication_figures_highres/fig01_sec_aaer_workflow_pretty.png/pdf
outputs/publication_figures_highres/fig02_class_imbalance_by_year_pretty.png/pdf
outputs/publication_figures_highres/fig03_roc_curves_pretty.png/pdf
outputs/publication_figures_highres/fig04_precision_recall_curves_pretty.png/pdf
outputs/publication_figures_highres/fig05_model_comparison_auc_pr_pretty.png/pdf
outputs/publication_figures_highres/fig06_top_screening_recall_lift_pretty.png/pdf
outputs/publication_figures_highres/fig07_ai_vc_index_weights_pretty.png/pdf
outputs/publication_figures_highres/fig08_ai_vc_coefficient_curves_pretty.png/pdf
outputs/publication_figures_highres/fig09_ai_vc_predicted_risk_distribution_pretty.png/pdf
```

## Recommended main-paper figures

1. Workflow diagram
2. Class imbalance by year
3. Precision-recall curves
4. Top-screening recall and lift
5. AI-guided VC index weights
6. AI-guided VC coefficient curves

Use ROC curves, model comparison bars, and predicted-risk distributions as optional or supplementary figures.
