#!/usr/bin/env python3
# compare_shallow_vs_nn_rf_shap.py
# Pick a CSV, run nested CV (outer=5, inner=3), compare 2 shallow + RF vs 2 NNs,
# enforce positive predictions via log(y+eps), save metrics/plots,
# then refit BEST model and generate SHAP plots.

import warnings, math, os, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- File picker --------------------
csv_path = None
try:
    import tkinter as tk
    from tkinter import filedialog
    tk.Tk().withdraw()
    csv_path = filedialog.askopenfilename(
        title="Choose your CSV",
        filetypes=[("CSV files","*.csv"), ("All files","*.*")]
    )
except Exception:
    pass
if not csv_path:
    csv_path = input("Paste CSV path and press Enter:\n").strip()
csv_path = Path(csv_path).expanduser().resolve()
assert csv_path.exists(), f"File not found: {csv_path}"

# -------------------- Helpers --------------------
def to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "ifc":
        return s
    s2 = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    s2 = s2.str.replace(r"[^0-9eE\.\-\+]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def infer_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    lower = {c: c.lower() for c in cols}
    tgt_keys = ["activity", "u/l", "ul", "response", "y"]
    tgt_candidates = [c for c in cols if any(k in lower[c] for k in tgt_keys)]
    ycol = tgt_candidates[0] if tgt_candidates else cols[-1]
    xcols = [c for c in cols if c != ycol and df[c].dtype.kind in "ifc"]
    xcols = [c for c in xcols if not c.lower().startswith("unnamed")]
    if not xcols:
        raise ValueError("No numeric features found after cleaning.")
    return xcols, ycol

def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def ci95(values, n_boot=1000, rng=0):
    rs = np.random.RandomState(rng)
    vals = np.asarray(values, dtype=float)
    boots = [np.mean(vals[rs.randint(0, len(vals), len(vals))]) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return lo, hi

# -------------------- Load & clean --------------------
raw = pd.read_csv(csv_path)
df = raw.copy()
for c in df.columns:
    df[c] = to_numeric_series(df[c])

xcols, ycol = infer_columns(df)
X = df[xcols].astype(float).to_numpy()
y_raw = df[ycol].astype(float).to_numpy()

EPS = 0.1
if (y_raw <= -EPS).any():
    raise ValueError("Target has values <= -EPS; increase EPS or fix data.")
y = np.log(y_raw + EPS)  # positivity-preserving

# -------------------- Models & CV --------------------
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

n_features = X.shape[1]

prep_poly = ColumnTransformer(
    [("poly_scale", Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                              ("scale", StandardScaler())]), list(range(n_features)))],
    remainder="drop"
)
prep_std = ColumnTransformer(
    [("scale", StandardScaler(), list(range(n_features)))],
    remainder="drop"
)

models = [
    ("Ridge(poly2) [shallow]",
     Pipeline([("prep", prep_poly), ("mdl", Ridge(random_state=0))]),
     {"mdl__alpha": [0.1, 1.0, 10.0]}),

    ("SVR(RBF) [shallow]",
     Pipeline([("prep", prep_std), ("mdl", SVR())]),
     {"mdl__C": [1.0, 10.0],
      "mdl__gamma": ["scale", 0.1],
      "mdl__epsilon": [0.1]}),

    ("RandomForest [shallow]",
     Pipeline([("prep", "passthrough"),
               ("mdl", RandomForestRegressor(random_state=0))]),
     {"mdl__n_estimators": [300, 600],
      "mdl__max_depth": [None, 6, 12],
      "mdl__min_samples_leaf": [1, 2, 3]}),

    ("MLP small (16,8) [nn]",
     Pipeline([("prep", prep_std),
               ("mdl", MLPRegressor(hidden_layer_sizes=(16,8),
                                    activation="relu",
                                    alpha=1e-3,
                                    learning_rate_init=1e-3,
                                    early_stopping=True,
                                    random_state=0,
                                    max_iter=3000))]),
     {"mdl__alpha": [1e-4, 1e-3]}),

    ("MLP medium (32,16) [nn]",
     Pipeline([("prep", prep_std),
               ("mdl", MLPRegressor(hidden_layer_sizes=(32,16),
                                    activation="relu",
                                    alpha=1e-3,
                                    learning_rate_init=1e-3,
                                    early_stopping=True,
                                    random_state=0,
                                    max_iter=3000))]),
     {"mdl__alpha": [1e-4, 1e-3]}),
]

outer = KFold(n_splits=5, shuffle=True, random_state=42)
def inner_cv(): return KFold(n_splits=3, shuffle=True, random_state=123)

metrics_rows, pred_rows = [], []

for name, pipe, grid in models:
    fold_r2, fold_rmse = [], []
    for k, (tr, te) in enumerate(outer.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        gs = GridSearchCV(pipe, grid, cv=inner_cv(), scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_

        yhat = best.predict(Xte)
        yte_bt  = np.exp(yte)  - EPS
        yhat_bt = np.maximum(np.exp(yhat) - EPS, 0.0)

        fold_r2.append(r2_score(yte_bt, yhat_bt))
        fold_rmse.append(rmse(yte_bt, yhat_bt))

        for idx, yy, pp in zip(te, yte_bt, yhat_bt):
            pred_rows.append({"model": name, "outer_fold": k, "row_index": int(idx),
                              "observed": float(yy), "predicted": float(pp)})

    rmse_lo, rmse_hi = ci95(fold_rmse)
    r2_lo, r2_hi     = ci95(fold_r2)

    metrics_rows.append({
        "model": name,
        "category": "shallow" if "[shallow]" in name else "nn",
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_sd": float(np.std(fold_rmse)),
        "rmse_ci95_lo": rmse_lo,
        "rmse_ci95_hi": rmse_hi,
        "r2_mean": float(np.mean(fold_r2)),
        "r2_sd": float(np.std(fold_r2)),
        "r2_ci95_lo": r2_lo,
        "r2_ci95_hi": r2_hi
    })

# -------------------- Save CV outputs --------------------
metrics = pd.DataFrame(metrics_rows).sort_values(["category","rmse_mean"])
preds   = pd.DataFrame(pred_rows)

out_dir = csv_path.parent
metrics_path = out_dir / "metrics.csv"
preds_path   = out_dir / "predictions.csv"
metrics.to_csv(metrics_path, index=False)
preds.to_csv(preds_path, index=False)

# -------------------- Plots --------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# RMSE bar (mean±SD)
labels = metrics["model"].tolist()
means = metrics["rmse_mean"].values
errs  = metrics["rmse_sd"].values

plt.figure(figsize=(8.4,4.8))
plt.bar(range(len(labels)), means, yerr=errs, capsize=5)
plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
plt.ylabel("RMSE (outer-CV, original scale)")
plt.title("Shallow vs Neural Nets — RMSE (mean ± SD)")
plt.tight_layout()
bar_path = out_dir / "rmse_bar.png"
plt.savefig(bar_path, dpi=160)

# Parity for best model (lowest RMSE over all outer predictions)
def _rmse_subset(d: pd.DataFrame) -> float:
    return rmse(d["observed"].values, d["predicted"].values)

best_model = None; best_r = 1e18
for m in preds["model"].unique():
    d = preds[preds["model"] == m]
    r = _rmse_subset(d)
    if r < best_r:
        best_model, best_r = m, r
d = preds[preds["model"] == best_model]
obs = d["observed"].values; pred = d["predicted"].values
lim_hi = max(obs.max(), pred.max()) * 1.05 if len(obs) else 1.0
plt.figure(figsize=(6.2,6.2))
plt.scatter(obs, pred, alpha=0.75, label=f"{best_model}")
plt.plot([0, lim_hi], [0, lim_hi], linestyle="--", linewidth=1.0)
plt.xlabel("Observed (U/L)"); plt.ylabel("Predicted (U/L)")
plt.title("Parity plot (outer-CV) — best model")
plt.xlim(0, lim_hi); plt.ylim(0, lim_hi); plt.legend(); plt.tight_layout()
parity_path = out_dir / "parity_best.png"
plt.savefig(parity_path, dpi=160)

# -------------------- SHAP for the BEST model --------------------
# Refit best model on FULL data with cross-validated hyperparams, then explain.
# Note: SHAP explains model output; here we explain BACK-TRANSFORMED predictions.

best_spec = [s for s in models if s[0] == best_model][0]
name, pipe, grid = best_spec

cv_full = GridSearchCV(pipe, grid, cv=inner_cv(), scoring="neg_mean_squared_error", n_jobs=-1)
cv_full.fit(X, y)
best_pipe_full = cv_full.best_estimator_

# Prepare a function that outputs back-transformed predictions

def f_predict_bt(Xarr: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(best_pipe_full.predict(Xarr)) - EPS, 0.0)



# Build a DataFrame of features for SHAP (keeps names)
Xdf = pd.DataFrame(df[xcols].astype(float).values, columns=xcols)

try:
    import shap
    # Choose explainer:
    # - If RandomForest (tree model): TreeExplainer on the RF itself (pipeline prep is passthrough)
    # - Else (SVR, Ridge, MLP): KernelExplainer on the full pipeline via f_predict_bt
    if "RandomForest" in name:
        rf = best_pipe_full.named_steps["mdl"]  # prep is passthrough
        explainer = shap.TreeExplainer(rf, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(Xdf)
    else:
        # Small background sample for speed
        BG = shap.utils.sample(Xdf, min(30, len(Xdf)), random_state=0)
        explainer = shap.KernelExplainer(f_predict_bt, BG)
        # nsamples kept modest for small N; increase if you want smoother plots
        shap_vals = explainer.shap_values(Xdf, nsamples=200)

    # Save SHAP summary (beeswarm)
    plt.figure()
    shap.summary_plot(shap_vals, Xdf, show=False)
    shap_sum_path = out_dir / "shap_summary.png"
    plt.tight_layout(); plt.savefig(shap_sum_path, dpi=160); plt.close()

    # Bar plot of mean |SHAP|
    plt.figure()
    shap.summary_plot(shap_vals, Xdf, plot_type="bar", show=False)
    shap_bar_path = out_dir / "shap_bar.png"
    plt.tight_layout(); plt.savefig(shap_bar_path, dpi=160); plt.close()

    print("WROTE:", shap_sum_path)
    print("WROTE:", shap_bar_path)
except ImportError:
    print("SHAP not installed. Install it with: pip install shap")
except Exception as e:
    print("SHAP computation failed:", repr(e))

# -------------------- Report file locations --------------------
print("WROTE:", metrics_path)
print("WROTE:", preds_path)
print("WROTE:", bar_path)
print("WROTE:", parity_path)
print("Best model (for SHAP):", best_model)
