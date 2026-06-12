"""
02_train_baseline_models.py

Fast, reproducible baseline models for the SEC AAER paper.
Train+validation years are used for the final test evaluation; year is not a predictor.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
try:
    import lightgbm as lgb
except Exception:
    lgb = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "final_features" / "sec_aaer_features_2009_2025.csv"
OUT = ROOT / "outputs" / "baseline_models"
OUT.mkdir(parents=True, exist_ok=True)

NUM_FEATURES = [
    "log_assets", "liabilities_assets", "roa", "net_margin", "cash_assets",
    "receivables_assets", "inventory_assets", "current_ratio", "debt_assets",
    "revenue_growth", "assets_growth", "gross_margin", "operating_margin",
    "working_capital_assets", "asset_quality_proxy", "receivables_revenue",
    "inventory_revenue", "debt_equity", "loss_flag", "negative_equity_flag",
    "dsri_proxy", "gmi_proxy", "aqi_proxy", "sgi", "lvgi",
    "debt_assets_change", "roa_change", "gross_margin_change",
    "receivables_growth", "inventory_growth", "debt_growth",
    "total_accruals_assets_proxy",
]
CAT_FEATURES = ["sic2"]

def onehot():
    return OneHotEncoder(handle_unknown="ignore", min_frequency=20)

def winsorize(train, test, cols):
    train = train.copy(); test = test.copy()
    for c in cols:
        lo, hi = train[c].quantile([0.005, 0.995])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            train[c] = train[c].clip(lo, hi); test[c] = test[c].clip(lo, hi)
    return train, test

def eval_row(y, p, name):
    return {"model": name, "roc_auc": roc_auc_score(y, p), "pr_auc_average_precision": average_precision_score(y, p), "brier_score": brier_score_loss(y, np.clip(p,0,1))}

def top_metrics(y, p, name):
    y = np.asarray(y).astype(int); p = np.asarray(p); n = len(y); pos = int(y.sum()); prev = pos/n; order = np.argsort(-p)
    rules = [("Top-K=positives", pos), ("Top 1%", max(1, round(.01*n))), ("Top 5%", max(1, round(.05*n))), ("Top 10%", max(1, round(.10*n))), ("Top 100", min(100,n)), ("Top 250", min(250,n)), ("Top 500", min(500,n))]
    rows=[]
    for label,k in rules:
        k=int(max(1,k)); idx=order[:k]; tp=int(y[idx].sum()); prec=tp/k; rec=tp/pos if pos else np.nan; lift=prec/prev if prev else np.nan
        rows.append({"model":name,"screening_rule":label,"selected_n":k,"true_positives_found":tp,"precision":prec,"recall":rec,"lift_over_random":lift})
    return rows

if not DATA.exists():
    raise SystemExit("Run scripts/01_make_final_features.py first.")

df = pd.read_csv(DATA).dropna(subset=["aaer_flag","year"]).copy()
df["year"] = df["year"].astype(int); df["aaer_flag"] = df["aaer_flag"].astype(int)
num = [c for c in NUM_FEATURES if c in df.columns]
cat = [c for c in CAT_FEATURES if c in df.columns]
features = num + cat
for c in num: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in cat: df[c] = df[c].astype(str).replace("nan", np.nan)
train = df[df.year <= 2021].copy(); test = df[df.year >= 2022].copy()
train, test = winsorize(train, test, num)
ytr = train.aaer_flag.values.astype(int); yte = test.aaer_flag.values.astype(int)
print("Temporal split:", flush=True)
print("Train+valid:", len(train), "positives:", int(ytr.sum()), flush=True)
print("Test:", len(test), "positives:", int(yte.sum()), flush=True)
print("Features:", len(features), flush=True)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", onehot())]), cat),
])
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
    "Random Forest": RandomForestClassifier(n_estimators=50, min_samples_leaf=10, class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, min_samples_leaf=10, class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1),
}
if lgb is not None:
    pos_ratio = (len(ytr)-ytr.sum())/max(1,ytr.sum())
    models["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, num_leaves=4, learning_rate=0.02, min_child_samples=50, reg_lambda=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=np.sqrt(pos_ratio), n_jobs=4, random_state=42, verbose=-1)
rows=[]; top=[]; preds=[]
for name, model in models.items():
    print("Training:", name, flush=True)
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(train[features], ytr)
    p = pipe.predict_proba(test[features])[:,1]
    row = eval_row(yte, p, name); row.update({"n_train":len(train),"n_test":len(test),"positives_train":int(ytr.sum()),"positives_test":int(yte.sum())}); rows.append(row)
    top.extend(top_metrics(yte, p, name))
    tmp = test[[c for c in ["cik","name","sic","sic2","year","aaer_flag"] if c in test.columns]].copy(); tmp["model"]=name; tmp["predicted_probability"]=p; preds.append(tmp)
metrics = pd.DataFrame(rows).sort_values("pr_auc_average_precision", ascending=False)
topdf = pd.DataFrame(top)
pred = pd.concat(preds, ignore_index=True)
metrics.to_csv(OUT/"baseline_auc_metrics.csv", index=False)
topdf.to_csv(OUT/"baseline_top_screening_metrics.csv", index=False)
pred.to_csv(OUT/"baseline_test_predictions.csv", index=False)
print("AUC / PR-AUC:", flush=True); print(metrics.to_string(index=False), flush=True)
print("Saved outputs to:", OUT, flush=True)
