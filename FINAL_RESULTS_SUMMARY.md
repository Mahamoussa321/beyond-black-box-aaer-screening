# Verified results summary

The package was tested on the uploaded SEC AAER 2009--2026 dataset.

Main modeling sample after excluding fiscal year 2026:
- 101,103 firm-year observations
- 226 AAER-positive firm-years
- 100,877 AAER-negative firm-years
- Train: 2009--2018; Validation: 2019--2021; Test: 2022--2025

AI teacher selection:
- Candidate teachers: LightGBM teacher and neural MLP teacher
- Selection criterion: validation PR-AUC
- Selected teacher in the tested run: LightGBM Teacher

Final test results from the validation-selected publication pipeline:
- AI-guided VC Student: ROC-AUC 0.7282, PR-AUC 0.0064, Top-5% recall 25.0%, Top-5% lift 5.00
- VC Student without AI distillation: ROC-AUC 0.7192, PR-AUC 0.0052, Top-5% recall 20.0%, Top-5% lift 4.00
- LightGBM baseline: ROC-AUC 0.7041, PR-AUC 0.0045, Top-5% recall 15.0%, Top-5% lift 3.00

Use these as reproducible preliminary results. For final manuscript submission, report confidence intervals via bootstrap if time permits.
