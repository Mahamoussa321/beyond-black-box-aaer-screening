"""
03_train_ai_teacher_guided_vc_validation_selected.py

Publication-oriented AI-teacher-guided single-index varying-coefficient logistic model.

Key safeguards:
    - Train years: 2009-2018.
    - Validation years: 2019-2021.
    - Test years: 2022-2025.
    - Hyperparameters for the teacher and VC student are selected on validation years only.
    - The final teacher and final VC student are refit on train+validation and evaluated once on the test years.
    - The year variable is never used as an input predictor.

The AI teacher is explicit: the script compares a LightGBM teacher and a small neural-network
MLP teacher, chooses the better one on validation PR-AUC, and uses the selected teacher's
probabilities as the distillation target for the interpretable VC student.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import random
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

try:
    import lightgbm as lgb
except Exception as exc:
    raise SystemExit("lightgbm is required. Install with: pip install lightgbm") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit("torch is required. Install with: pip install torch") from exc

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "final_features" / "sec_aaer_features_2009_2025.csv"
OUTDIR = ROOT / "outputs" / "ai_teacher_guided_vc_publication"
OUTDIR.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise SystemExit("Run scripts/01_make_final_features.py first.")

X_FEATURES = [
    "log_assets", "liabilities_assets", "roa", "net_margin",
    "cash_assets", "receivables_assets", "inventory_assets",
    "current_ratio", "debt_assets", "revenue_growth", "assets_growth",
    "gross_margin", "working_capital_assets", "asset_quality_proxy",
    "receivables_revenue", "debt_equity", "loss_flag",
    "dsri_proxy", "gmi_proxy", "sgi", "lvgi",
    "debt_assets_change", "roa_change", "gross_margin_change",
    "receivables_growth", "inventory_growth", "debt_growth",
    "total_accruals_assets_proxy",
]

Z_NUM = [
    "assets", "liabilities", "revenue", "net_income", "cash", "receivables",
    "inventory", "current_assets", "current_liabilities", "debt", "equity",
    "operating_income", "gross_profit", "log_assets", "liabilities_assets",
    "roa", "net_margin", "cash_assets", "receivables_assets",
    "inventory_assets", "current_ratio", "debt_assets", "revenue_growth",
    "assets_growth", "gross_margin", "operating_margin", "working_capital_assets",
    "asset_quality_proxy", "receivables_revenue", "inventory_revenue",
    "debt_equity", "loss_flag", "negative_equity_flag", "dsri_proxy",
    "gmi_proxy", "aqi_proxy", "sgi", "lvgi", "debt_assets_change",
    "roa_change", "gross_margin_change", "receivables_growth", "inventory_growth",
    "debt_growth", "total_accruals_assets_proxy",
]

CAT = ["sic2"]


def onehot_encoder_dense(min_frequency=20):
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse=False)


def onehot_encoder_sparse(min_frequency=20):
    return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)


def winsorize_from_train(train, frames, cols, lower=0.005, upper=0.995):
    train = train.copy()
    frames = [x.copy() for x in frames]
    for c in cols:
        if c not in train.columns:
            continue
        lo, hi = train[c].quantile([lower, upper])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            train[c] = train[c].clip(lo, hi)
            for j in range(len(frames)):
                frames[j][c] = frames[j][c].clip(lo, hi)
    return train, frames


def evaluate(y_true, prob, name):
    prob = np.asarray(prob).astype(float)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc_average_precision": average_precision_score(y_true, prob),
        "brier_score": brier_score_loss(y_true, np.clip(prob, 0, 1)),
        "positives": int(np.sum(y_true)),
        "n": int(len(y_true)),
    }


def top_screening_metrics(y_true, prob, model_name):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)
    n = len(y_true)
    total_pos = int(y_true.sum())
    prevalence = total_pos / n
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


class MLPTeacher(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


class VCStudent(nn.Module):
    def __init__(self, px, pz, degree=3):
        super().__init__()
        self.degree = int(degree)
        self.alpha_raw = nn.Parameter(torch.randn(pz) * 0.02)
        self.gamma = nn.Parameter(torch.randn(self.degree + 1, px + 1) * 0.02)

    def alpha(self):
        a = self.alpha_raw / (torch.norm(self.alpha_raw) + 1e-8)
        # Identify sign so the first component is nonnegative.
        sign = torch.where(a[0] < 0, torch.tensor(-1.0, device=a.device), torch.tensor(1.0, device=a.device))
        return a * sign

    def basis(self, I):
        return torch.stack([I ** d for d in range(self.degree + 1)], dim=1)

    def forward(self, X, Z):
        I = torch.clamp(Z @ self.alpha(), -4, 4)
        beta = self.basis(I) @ self.gamma
        logit = beta[:, 0] + (beta[:, 1:] * X).sum(1)
        return logit, I, beta

    def smooth_penalty(self):
        if self.degree < 2:
            return torch.tensor(0.0)
        return (self.gamma[2:, :] ** 2).mean()


def train_mlp_teacher(X_train, y_train, X_eval, epochs=40, seed=42):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X_train.astype("float32"))
    ytr = torch.tensor(y_train.astype("float32"))
    Xev = torch.tensor(X_eval.astype("float32"))

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    model = MLPTeacher(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(max(1.0, (len(y_train) - y_train.sum()) / max(1, y_train.sum())) ** 0.5)

    for _ in range(epochs):
        for _b in range(6):
            pi = np.random.choice(pos_idx, size=min(len(pos_idx), 256), replace=True)
            ni = np.random.choice(neg_idx, size=min(len(neg_idx), 2048), replace=False)
            idx = np.concatenate([pi, ni])
            np.random.shuffle(idx)
            logit = model(Xtr[idx])
            loss = F.binary_cross_entropy_with_logits(logit, ytr[idx], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    with torch.no_grad():
        prob = torch.sigmoid(model(Xev)).numpy()
        train_prob = torch.sigmoid(model(Xtr)).numpy()
    return model, train_prob.astype("float32"), prob.astype("float32")


def fit_teacher(train, valid, test, features, cat_features, y_col="aaer_flag"):
    num_features = [c for c in features if c not in cat_features]
    y_train = train[y_col].values.astype(int)
    y_valid = valid[y_col].values.astype(int)

    sparse_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", onehot_encoder_sparse(20))]), cat_features),
    ])
    Xtr = sparse_pre.fit_transform(train[features])
    Xva = sparse_pre.transform(valid[features])

    pos_ratio = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    lgb_grid = [
        {"n_estimators": 250, "num_leaves": 4, "learning_rate": 0.02, "reg_lambda": 5, "scale_pos_weight": np.sqrt(pos_ratio)},
        {"n_estimators": 400, "num_leaves": 8, "learning_rate": 0.015, "reg_lambda": 10, "scale_pos_weight": np.sqrt(pos_ratio)},
        {"n_estimators": 300, "num_leaves": 4, "learning_rate": 0.025, "reg_lambda": 5, "scale_pos_weight": 1.0},
    ]
    teacher_rows = []
    best = None
    best_ap = -np.inf
    for j, params in enumerate(lgb_grid):
        model = lgb.LGBMClassifier(
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            random_state=SEED + j,
            verbose=-1,
            **params,
        )
        model.fit(Xtr, y_train)
        pva = model.predict_proba(Xva)[:, 1]
        row = evaluate(y_valid, pva, f"LightGBM Teacher cfg{j}")
        row.update(params)
        teacher_rows.append(row)
        if row["pr_auc_average_precision"] > best_ap:
            best_ap = row["pr_auc_average_precision"]
            best = ("LightGBM Teacher", params)

    # MLP teacher candidate on the same teacher feature set.
    dense_pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", onehot_encoder_dense(20))]), cat_features),
    ])
    Xtr_dense = dense_pre.fit_transform(train[features])
    Xva_dense = dense_pre.transform(valid[features])
    if hasattr(Xtr_dense, "toarray"):
        Xtr_dense = Xtr_dense.toarray()
        Xva_dense = Xva_dense.toarray()
    _, _, pva_mlp = train_mlp_teacher(Xtr_dense, y_train, Xva_dense, epochs=8)
    row = evaluate(y_valid, pva_mlp, "Neural MLP Teacher")
    teacher_rows.append(row)
    if row["pr_auc_average_precision"] > best_ap:
        best_ap = row["pr_auc_average_precision"]
        best = ("Neural MLP Teacher", None)

    pd.DataFrame(teacher_rows).sort_values("pr_auc_average_precision", ascending=False).to_csv(
        OUTDIR / "teacher_validation_grid.csv", index=False
    )
    print("Teacher validation grid:")
    print(pd.DataFrame(teacher_rows).sort_values("pr_auc_average_precision", ascending=False).to_string(index=False))
    print("Selected teacher:", best[0])
    return best


def final_teacher_probabilities(train_final, test, features, cat_features, selected_teacher):
    num_features = [c for c in features if c not in cat_features]
    y_train = train_final["aaer_flag"].values.astype(int)
    y_test = test["aaer_flag"].values.astype(int)
    name, params = selected_teacher

    if name == "LightGBM Teacher":
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", onehot_encoder_sparse(20))]), cat_features),
        ])
        Xtr = pre.fit_transform(train_final[features])
        Xte = pre.transform(test[features])
        model = lgb.LGBMClassifier(
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
            random_state=SEED,
            verbose=-1,
            **params,
        )
        model.fit(Xtr, y_train)
        ptr = model.predict_proba(Xtr)[:, 1].astype("float32")
        pte = model.predict_proba(Xte)[:, 1].astype("float32")
    else:
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_features),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", onehot_encoder_dense(20))]), cat_features),
        ])
        Xtr = pre.fit_transform(train_final[features])
        Xte = pre.transform(test[features])
        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray(); Xte = Xte.toarray()
        _, ptr, pte = train_mlp_teacher(Xtr, y_train, Xte, epochs=12)
    pd.DataFrame([evaluate(y_test, pte, name)]).to_csv(OUTDIR / "teacher_test_metrics.csv", index=False)
    return ptr, pte, name


def make_student_arrays(train, valid_or_test, x_features, z_num, cat):
    xpre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    zpre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), z_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", onehot_encoder_dense(20))]), cat),
    ])
    Xtr = xpre.fit_transform(train[x_features]).astype("float32")
    Xev = xpre.transform(valid_or_test[x_features]).astype("float32")
    Ztr = zpre.fit_transform(train[z_num + cat])
    Zev = zpre.transform(valid_or_test[z_num + cat])
    if hasattr(Ztr, "toarray"):
        Ztr = Ztr.toarray(); Zev = Zev.toarray()
    Ztr = Ztr.astype("float32"); Zev = Zev.astype("float32")
    z_names = list(z_num)
    if cat:
        enc = zpre.named_transformers_["cat"].named_steps["oh"]
        z_names += enc.get_feature_names_out(cat).tolist()
    return Xtr, Ztr, Xev, Zev, z_names


def train_student(X, Z, y, teacher_prob, cfg, epochs=40):
    y = np.asarray(y).astype(int)
    teacher_prob = np.asarray(teacher_prob).astype("float32")
    model = VCStudent(X.shape[1], Z.shape[1], degree=cfg["degree"])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    posw = torch.tensor(float(cfg["pos_weight"]))

    for _epoch in range(epochs):
        for _b in range(5):
            pi = np.random.choice(pos_idx, size=min(len(pos_idx), 384), replace=True)
            ni = np.random.choice(neg_idx, size=min(len(neg_idx), 1536), replace=False)
            idx = np.concatenate([pi, ni])
            np.random.shuffle(idx)
            xb = torch.tensor(X[idx])
            zb = torch.tensor(Z[idx])
            yb = torch.tensor(y[idx].astype("float32"))
            tb = torch.tensor(teacher_prob[idx])
            logit, _, _ = model(xb, zb)
            cls = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=posw)
            # AI-guidance: match the teacher probabilities through soft-label distillation.
            ai = F.binary_cross_entropy_with_logits(logit, tb)
            loss = cls + cfg["lambda_ai"] * ai + 1e-4 * torch.mean(torch.abs(model.alpha())) + 1e-3 * model.smooth_penalty()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def predict_student(model, X, Z):
    with torch.no_grad():
        logit, I, beta = model(torch.tensor(X), torch.tensor(Z))
        prob = torch.sigmoid(logit).numpy()
    return prob, I.numpy(), beta.numpy()


def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["aaer_flag", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df["aaer_flag"] = df["aaer_flag"].astype(int)
    df = df[(df["year"] >= 2009) & (df["year"] <= 2025)].copy()

    x_features = [c for c in X_FEATURES if c in df.columns]
    z_num = [c for c in Z_NUM if c in df.columns]
    cat = [c for c in CAT if c in df.columns]
    all_numeric = sorted(set(x_features + z_num))
    for c in all_numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat:
        df[c] = df[c].astype(str).replace("nan", np.nan)

    train = df[df["year"] <= 2018].copy()
    valid = df[(df["year"] >= 2019) & (df["year"] <= 2021)].copy()
    test = df[df["year"] >= 2022].copy()

    train, [valid, test] = winsorize_from_train(train, [valid, test], all_numeric)
    train_final = pd.concat([train, valid], ignore_index=True)

    print("Split:")
    print("Train:", len(train), "positives:", int(train.aaer_flag.sum()))
    print("Validation:", len(valid), "positives:", int(valid.aaer_flag.sum()))
    print("Test:", len(test), "positives:", int(test.aaer_flag.sum()))
    print("X features:", len(x_features), "Z/context features before one-hot:", len(z_num) + len(cat))

    teacher_features = sorted(set(x_features + z_num + cat))
    selected_teacher = fit_teacher(train, valid, test, teacher_features, cat)

    # Validation probabilities for teacher, needed for student validation selection.
    # Fit selected teacher on train only and generate train/validation probabilities.
    ptr_train, pva_teacher, teacher_name_for_valid = final_teacher_probabilities(train, valid, teacher_features, cat, selected_teacher)

    Xtr, Ztr, Xva, Zva, _z_names_valid = make_student_arrays(train, valid, x_features, z_num, cat)
    ytr = train["aaer_flag"].values.astype(int)
    yva = valid["aaer_flag"].values.astype(int)

    configs = []
    for lambda_ai in [0.0, 0.05, 0.20]:
        for pos_weight in [5.0, 10.0, 20.0]:
            configs.append({"lambda_ai": lambda_ai, "pos_weight": pos_weight, "degree": 3})

    grid_rows = []
    best_cfg = None
    best_ap = -np.inf
    for cfg in configs:
        t0 = time.time()
        model = train_student(Xtr, Ztr, ytr, ptr_train, cfg, epochs=20)
        pva, _Iva, _bva = predict_student(model, Xva, Zva)
        row = evaluate(yva, pva, "AI-guided VC Student")
        row.update(cfg)
        row["train_seconds"] = time.time() - t0
        grid_rows.append(row)
        print("Student validation cfg:", row, flush=True)
        if row["pr_auc_average_precision"] > best_ap:
            best_ap = row["pr_auc_average_precision"]
            best_cfg = cfg

    grid_df = pd.DataFrame(grid_rows).sort_values("pr_auc_average_precision", ascending=False)
    grid_df.to_csv(OUTDIR / "ai_vc_validation_grid.csv", index=False)

    best_overall_cfg = grid_df.iloc[0][["lambda_ai", "pos_weight", "degree"]].to_dict()
    best_overall_cfg = {"lambda_ai": float(best_overall_cfg["lambda_ai"]), "pos_weight": float(best_overall_cfg["pos_weight"]), "degree": int(best_overall_cfg["degree"])}

    ai_grid = grid_df[grid_df["lambda_ai"] > 0].copy()
    if len(ai_grid) > 0:
        best_ai_cfg = ai_grid.iloc[0][["lambda_ai", "pos_weight", "degree"]].to_dict()
        best_ai_cfg = {"lambda_ai": float(best_ai_cfg["lambda_ai"]), "pos_weight": float(best_ai_cfg["pos_weight"]), "degree": int(best_ai_cfg["degree"])}
    else:
        best_ai_cfg = best_overall_cfg

    print("Selected overall VC configuration from validation:", best_overall_cfg, flush=True)
    print("Selected AI-guided VC configuration from validation (lambda_ai > 0):", best_ai_cfg, flush=True)

    # Refit selected teacher on train+validation.
    ptr_final, pte_teacher, selected_teacher_name = final_teacher_probabilities(train_final, test, teacher_features, cat, selected_teacher)
    Xtf, Ztf, Xte, Zte, z_names = make_student_arrays(train_final, test, x_features, z_num, cat)
    ytf = train_final["aaer_flag"].values.astype(int)
    yte = test["aaer_flag"].values.astype(int)

    final_models = [
        ("VC Student (validation-selected)", best_overall_cfg),
        ("AI-guided VC Student", best_ai_cfg),
    ]

    metric_rows = []
    top_rows = []
    pred_frames = []
    ai_model_for_interpretation = None
    ai_prob = ai_index = ai_beta = None

    for model_label, cfg in final_models:
        final_model = train_student(Xtf, Ztf, ytf, ptr_final, cfg, epochs=30)
        pte, Ite, betate = predict_student(final_model, Xte, Zte)
        metrics = evaluate(yte, pte, model_label)
        metrics.update(cfg)
        metrics.update({
            "n_train": len(train_final),
            "positives_train": int(ytf.sum()),
            "teacher_selected_on_validation": selected_teacher_name,
            "selection_metric": "validation PR-AUC",
        })
        metric_rows.append(metrics)
        top_rows.extend(top_screening_metrics(yte, pte, model_label))

        pred_cols = [c for c in ["cik", "name", "sic", "sic2", "year", "aaer_flag"] if c in test.columns]
        pred = test[pred_cols].copy()
        pred["model"] = model_label
        pred["student_probability"] = pte
        pred["student_index"] = Ite
        pred["teacher_probability"] = pte_teacher
        pred_frames.append(pred)

        if model_label == "AI-guided VC Student":
            ai_model_for_interpretation = final_model
            ai_prob, ai_index, ai_beta = pte, Ite, betate

    metrics_df = pd.DataFrame(metric_rows).sort_values("pr_auc_average_precision", ascending=False)
    top_df = pd.DataFrame(top_rows)
    pred_df = pd.concat(pred_frames, ignore_index=True)

    metrics_df.to_csv(OUTDIR / "ai_vc_test_metrics.csv", index=False)
    top_df.to_csv(OUTDIR / "ai_vc_top_screening_metrics.csv", index=False)
    pred_df.to_csv(OUTDIR / "ai_vc_test_predictions.csv", index=False)

    # Interpretability outputs are reported for the explicitly AI-guided model (lambda_ai > 0).
    final_model = ai_model_for_interpretation
    if final_model is None:
        final_model = train_student(Xtf, Ztf, ytf, ptr_final, best_ai_cfg, epochs=30)

    alpha = final_model.alpha().detach().numpy()
    alpha_df = pd.DataFrame({"index_feature": z_names, "alpha_weight": alpha})
    alpha_df["abs_alpha_weight"] = alpha_df["alpha_weight"].abs()
    alpha_df.sort_values("abs_alpha_weight", ascending=False).to_csv(OUTDIR / "ai_vc_index_weights.csv", index=False)

    grid = np.linspace(-3, 3, 121).astype("float32")
    with torch.no_grad():
        coef = (final_model.basis(torch.tensor(grid)) @ final_model.gamma).numpy()
    coef_df = pd.DataFrame(coef, columns=["intercept"] + x_features)
    coef_df.insert(0, "index_grid", grid)
    coef_df.to_csv(OUTDIR / "ai_vc_coefficient_curves.csv", index=False)

    print("\nFinal VC / AI-guided VC test metrics:")
    print(metrics_df.to_string(index=False))
    print("\nTop screening:")
    print(top_df.to_string(index=False))
    print("\nSaved outputs to:", OUTDIR)


if __name__ == "__main__":
    main()
